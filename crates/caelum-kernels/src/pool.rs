//! Lock-free hybrid spin-then-park worker thread pool (inspired by `ggml_threadpool`).
//!
//! During active inference (~200 matvecs per token), workers synchronize via atomic spin-barriers
//! with ZERO mutex locks and ZERO OS futex syscalls (< 0.5 µs dispatch latency). When idle between
//! prompts, workers park cleanly on a `Condvar` at 0% CPU usage.

#[cfg(not(target_arch = "wasm32"))]
use std::cell::UnsafeCell;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc, Condvar, Mutex,
};
#[cfg(not(target_arch = "wasm32"))]
use std::thread::{self, JoinHandle};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct Job {
    func: *const (),
    call: unsafe fn(*const (), usize, usize),
    total_items: usize,
}

#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for Job {}
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for Job {}

#[cfg(not(target_arch = "wasm32"))]
struct SharedState {
    generation: AtomicU64,
    remaining: AtomicUsize,
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
const WORKER_SPIN_ITERS: u32 = 30_000;

pub struct ThreadPool {
    num_threads: usize,
    #[cfg(not(target_arch = "wasm32"))]
    state: Option<Arc<SharedState>>,
    #[cfg(not(target_arch = "wasm32"))]
    workers: Vec<JoinHandle<()>>,
    /// Serializes concurrent callers to `parallel_for_range`.
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

            let state = Arc::new(SharedState {
                generation: AtomicU64::new(0),
                remaining: AtomicUsize::new(0),
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
                let handle = thread::Builder::new()
                    .name(format!("caelum-worker-{worker_idx}"))
                    .spawn(move || {
                        let mut last_gen = 0u64;
                        loop {
                            // 1. Fast lock-free spin wait for the next matvec/matmul
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
                                    // 2. Idle fallback: park cleanly on Condvar (0% CPU when idle)
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

                            // Safety: `generation` Acquire synchronizes with main thread's Release
                            // after writing `st.job`, and `st.job` remains valid until `remaining` reaches 0.
                            let job = unsafe { (*st.job.get()).expect("job present") };
                            let (start, end) =
                                partition_range(job.total_items, total_threads, worker_idx);
                            if start < end {
                                unsafe { (job.call)(job.func, start, end) };
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

    /// Splits `0..total` into disjoint index ranges and executes `f(start, end)` across all threads.
    pub fn parallel_for_range<F>(&self, total: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if total == 0 {
            return;
        }

        #[cfg(target_arch = "wasm32")]
        {
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
                });
            }
            state.remaining.store(num_workers, Ordering::Release);
            state.generation.fetch_add(1, Ordering::Release);

            // Only touch Condvar if at least one worker actually went to sleep while idle
            if state.parked.load(Ordering::Acquire) > 0 {
                let _g = state.park_mutex.lock().unwrap();
                state.wake_workers.notify_all();
            }

            // Main thread executes shard 0 immediately
            let (start0, end0) = partition_range(total, self.num_threads, 0);
            if start0 < end0 {
                f(start0, end0);
            }

            // Lock-free spin barrier waiting for workers to complete their shards
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
