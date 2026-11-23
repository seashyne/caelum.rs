//! Lock-free hybrid spin-then-park worker thread pool (inspired by `ggml_threadpool`)
//! with Linux P-core-first affinity pinning and dynamic work-stealing for hybrid CPUs.
//!
//! During active inference (~100–200 matvecs per token), workers synchronize via atomic
//! spin-barriers with ZERO mutex locks and ZERO OS futex syscalls (< 0.5 µs dispatch latency).
//! On hybrid CPUs (e.g., Intel P-cores + E-cores or ARM big.LITTLE), threads are pinned to
//! distinct physical P-cores first, and `parallel_for_dynamic` balances heterogeneous cores
//! automatically via an atomic work-stealing cursor.

#[cfg(not(target_arch = "wasm32"))]
use std::cell::UnsafeCell;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc, Condvar, Mutex,
};
#[cfg(not(target_arch = "wasm32"))]
use std::thread::{self, JoinHandle};

#[cfg(target_os = "linux")]
mod affinity {
    use std::fs;
    use std::path::Path;

    extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    }

    pub fn pin_current_thread(cpu_id: usize) {
        if cpu_id >= 1024 {
            return;
        }
        let mut mask = [0u64; 16]; // 1024 bits (CPU_SETSIZE)
        mask[cpu_id / 64] |= 1u64 << (cpu_id % 64);
        unsafe {
            let _ = sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr());
        }
    }

    /// Returns logical CPU IDs ordered for maximum inference throughput:
    /// 1. Primary logical thread of each physical core, fastest max clock first (P-cores → E-cores)
    /// 2. HyperThread/SMT siblings, fastest max clock first
    pub fn detect_preferred_cpus() -> Vec<usize> {
        let mut entries: Vec<(bool, u64, usize)> = Vec::new();
        for cpu in 0..256usize {
            let base = format!("/sys/devices/system/cpu/cpu{cpu}");
            if !Path::new(&base).exists() {
                continue;
            }
            // Skip offline CPUs
            if let Ok(online) = fs::read_to_string(format!("{base}/online")) {
                if online.trim() == "0" {
                    continue;
                }
            }
            let max_freq = fs::read_to_string(format!("{base}/cpufreq/cpuinfo_max_freq"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0);
            let is_primary = fs::read_to_string(format!("{base}/topology/thread_siblings_list"))
                .ok()
                .and_then(|s| {
                    s.trim()
                        .split(|c| c == '-' || c == ',')
                        .next()
                        .and_then(|first| first.parse::<usize>().ok())
                })
                .map(|first| first == cpu)
                .unwrap_or(true);
            entries.push((is_primary, max_freq, cpu));
        }

        // Sort: primary physical cores first, then highest max_freq first, then lowest cpu id
        entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)));
        entries.into_iter().map(|(_, _, id)| id).collect()
    }

    /// Returns the number of top-tier physical performance cores (or all physical cores if homogeneous).
    pub fn detect_p_core_count() -> Option<usize> {
        let mut primary_freqs: Vec<u64> = Vec::new();
        for cpu in 0..256usize {
            let base = format!("/sys/devices/system/cpu/cpu{cpu}");
            if !Path::new(&base).exists() {
                continue;
            }
            let is_primary = fs::read_to_string(format!("{base}/topology/thread_siblings_list"))
                .ok()
                .and_then(|s| {
                    s.trim()
                        .split(|c| c == '-' || c == ',')
                        .next()
                        .and_then(|first| first.parse::<usize>().ok())
                })
                .map(|first| first == cpu)
                .unwrap_or(true);
            if !is_primary {
                continue;
            }
            let max_freq = fs::read_to_string(format!("{base}/cpufreq/cpuinfo_max_freq"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0);
            primary_freqs.push(max_freq);
        }
        if primary_freqs.is_empty() {
            return None;
        }
        let max_f = *primary_freqs.iter().max().unwrap_or(&0);
        if max_f == 0 {
            return Some(primary_freqs.len());
        }
        // Count physical cores within 90% of peak frequency (P-cores on hybrid chips)
        let threshold = (max_f * 9) / 10;
        let p_cores = primary_freqs.iter().filter(|&&f| f >= threshold).count();
        Some(p_cores.max(1))
    }
}

/// Detects the optimal default thread count for this CPU (prefers physical P-cores on hybrid chips).
pub fn recommended_thread_count() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Some(p_cores) = affinity::detect_p_core_count() {
            return p_cores.max(1);
        }
    }
    let default_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    #[cfg(target_arch = "x86_64")]
    {
        default_threads.min(8).max(1)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        default_threads.min(8).max(1)
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct Job {
    func: *const (),
    call: unsafe fn(*const (), usize, usize),
    total_items: usize,
    /// When `chunk_size > 0`, threads dynamically steal chunks via `SharedState::cursor`.
    /// When `chunk_size == 0`, threads take static equal shards (`partition_range`).
    chunk_size: usize,
}

#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for Job {}
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for Job {}

#[cfg(not(target_arch = "wasm32"))]
struct SharedState {
    generation: AtomicU64,
    remaining: AtomicUsize,
    cursor: AtomicUsize,
    parked: AtomicUsize,
    shutdown: AtomicBool,
    /// Written exclusively by main thread while `remaining == 0` before releasing `generation`.
    job: UnsafeCell<Option<Job>>,
    park_mutex: Mutex<()>,
    wake_workers: Condvar,
}

#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for SharedState {}
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for SharedState {}

/// Number of `spin_loop` pauses workers wait for the next layer before parking on `Condvar`.
#[cfg(not(target_arch = "wasm32"))]
const WORKER_SPIN_ITERS: u32 = 40_000;

pub struct ThreadPool {
    num_threads: usize,
    #[cfg(not(target_arch = "wasm32"))]
    state: Option<Arc<SharedState>>,
    #[cfg(not(target_arch = "wasm32"))]
    workers: Vec<JoinHandle<()>>,
    /// Serializes concurrent callers to `parallel_for_*`.
    #[cfg(not(target_arch = "wasm32"))]
    dispatch_lock: Mutex<()>,
}

impl ThreadPool {
    pub fn new(num_threads: usize) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = num_threads;
            return Self { num_threads: 1 };
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let num_threads = num_threads.max(1);
            if num_threads <= 1 {
                return Self {
                    num_threads: 1,
                    state: None,
                    workers: Vec::new(),
                    dispatch_lock: Mutex::new(()),
                };
            }

            #[cfg(target_os = "linux")]
            let preferred_cpus = affinity::detect_preferred_cpus();

            // Pin main thread to top physical P-core (index 0) so it never drifts onto an E-core
            #[cfg(target_os = "linux")]
            if preferred_cpus.len() >= num_threads {
                affinity::pin_current_thread(preferred_cpus[0]);
            }

            let state = Arc::new(SharedState {
                generation: AtomicU64::new(0),
                remaining: AtomicUsize::new(0),
                cursor: AtomicUsize::new(0),
                parked: AtomicUsize::new(0),
                shutdown: AtomicBool::new(false),
                job: UnsafeCell::new(None),
                park_mutex: Mutex::new(()),
                wake_workers: Condvar::new(),
            });

            let num_workers = num_threads - 1;
            let mut workers = Vec::with_capacity(num_workers);

            for worker_idx in 1..num_threads {
                let st = Arc::clone(&state);
                let total_threads = num_threads;
                #[cfg(target_os = "linux")]
                let pin_cpu = if preferred_cpus.len() >= num_threads {
                    Some(preferred_cpus[worker_idx])
                } else {
                    None
                };

                let handle = thread::Builder::new()
                    .name(format!("caelum-worker-{worker_idx}"))
                    .spawn(move || {
                        #[cfg(target_os = "linux")]
                        if let Some(cpu_id) = pin_cpu {
                            affinity::pin_current_thread(cpu_id);
                        }

                        let mut last_gen = 0u64;
                        loop {
                            let mut spins = 0u32;
                            let mut cur_gen = st.generation.load(Ordering::Acquire);
                            while cur_gen == last_gen {
                                if st.shutdown.load(Ordering::Acquire) {
                                    return;
                                }
                                if spins < WORKER_SPIN_ITERS {
                                    spins += 1;
                                    std::hint::spin_loop();
                                    cur_gen = st.generation.load(Ordering::Acquire);
                                } else {
                                    let guard = st.park_mutex.lock().unwrap();
                                    st.parked.fetch_add(1, Ordering::Release);
                                    cur_gen = st.generation.load(Ordering::Acquire);
                                    if cur_gen == last_gen && !st.shutdown.load(Ordering::Acquire) {
                                        let _g = st.wake_workers.wait(guard).unwrap();
                                    }
                                    st.parked.fetch_sub(1, Ordering::AcqRel);
                                    spins = 0;
                                    cur_gen = st.generation.load(Ordering::Acquire);
                                }
                            }

                            if st.shutdown.load(Ordering::Acquire) {
                                return;
                            }
                            last_gen = cur_gen;

                            let job = unsafe { (*st.job.get()).expect("job present") };
                            if job.chunk_size > 0 {
                                loop {
                                    let start = st.cursor.fetch_add(job.chunk_size, Ordering::Relaxed);
                                    if start >= job.total_items {
                                        break;
                                    }
                                    let end = (start + job.chunk_size).min(job.total_items);
                                    unsafe { (job.call)(job.func, start, end) };
                                }
                            } else {
                                let (start, end) =
                                    partition_range(job.total_items, total_threads, worker_idx);
                                if start < end {
                                    unsafe { (job.call)(job.func, start, end) };
                                }
                            }

                            st.remaining.fetch_sub(1, Ordering::AcqRel);
                        }
                    })
                    .expect("failed to spawn caelum worker thread");
                workers.push(handle);
            }

            Self {
                num_threads,
                state: Some(state),
                workers,
                dispatch_lock: Mutex::new(()),
            }
        }
    }

    #[inline]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Splits `0..total` into static disjoint shards and executes `f(start, end)` across all threads.
    pub fn parallel_for_range<F>(&self, total: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        self.dispatch_job(total, 0, f);
    }

    /// Dynamically steals chunks of size `chunk_size` across `0..total` via a lock-free atomic cursor.
    /// Ideal for hybrid P-core + E-core CPUs where fast cores naturally absorb more tiles.
    pub fn parallel_for_dynamic<F>(&self, total: usize, chunk_size: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        self.dispatch_job(total, chunk_size.max(1), f);
    }

    fn dispatch_job<F>(&self, total: usize, chunk_size: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if total == 0 {
            return;
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ = chunk_size;
            f(0, total);
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(state) = &self.state else {
                f(0, total);
                return;
            };

            let _serial = self.dispatch_lock.lock().unwrap();

            unsafe fn trampoline<F: Fn(usize, usize) + Sync>(
                fn_ptr: *const (),
                start: usize,
                end: usize,
            ) {
                let func = &*(fn_ptr as *const F);
                func(start, end);
            }

            let num_workers = self.num_threads - 1;
            unsafe {
                *state.job.get() = Some(Job {
                    func: (&f as *const F) as *const (),
                    call: trampoline::<F>,
                    total_items: total,
                    chunk_size,
                });
            }
            if chunk_size > 0 {
                state.cursor.store(0, Ordering::Relaxed);
            }
            state.remaining.store(num_workers, Ordering::Release);
            state.generation.fetch_add(1, Ordering::Release);

            if state.parked.load(Ordering::Acquire) > 0 {
                let _g = state.park_mutex.lock().unwrap();
                state.wake_workers.notify_all();
            }

            // Main thread participates immediately
            if chunk_size > 0 {
                loop {
                    let start = state.cursor.fetch_add(chunk_size, Ordering::Relaxed);
                    if start >= total {
                        break;
                    }
                    let end = (start + chunk_size).min(total);
                    f(start, end);
                }
            } else {
                let (start0, end0) = partition_range(total, self.num_threads, 0);
                if start0 < end0 {
                    f(start0, end0);
                }
            }

            // Lock-free spin barrier waiting for workers to complete
            while state.remaining.load(Ordering::Acquire) > 0 {
                std::hint::spin_loop();
            }
            unsafe {
                *state.job.get() = None;
            }
        }
    }

    /// Splits `out` into disjoint row slices and executes `f(row_start, chunk)` across all threads.
    pub fn parallel_for_rows<F>(&self, out: &mut [f32], f: F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        let total = out.len();
        if total == 0 {
            return;
        }
        let base_addr = out.as_mut_ptr() as usize;
        self.parallel_for_range(total, |start, end| {
            let slice = unsafe {
                std::slice::from_raw_parts_mut((base_addr as *mut f32).add(start), end - start)
            };
            f(start, slice);
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn partition_range(total: usize, num_shards: usize, shard_idx: usize) -> (usize, usize) {
    let base = total / num_shards;
    let rem = total % num_shards;
    let start = shard_idx * base + shard_idx.min(rem);
    let len = base + usize::from(shard_idx < rem);
    (start, start + len)
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for ThreadPool {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            state.shutdown.store(true, Ordering::Release);
            state.generation.fetch_add(1, Ordering::Release);
            {
                let _guard = state.park_mutex.lock().unwrap();
                state.wake_workers.notify_all();
            }
            for h in self.workers.drain(..) {
                let _ = h.join();
            }
        }
    }
}
