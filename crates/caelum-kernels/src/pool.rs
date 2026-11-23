//! Zero-dependency persistent worker thread pool optimized for low-latency LLM row-parallel matvec.
//!
//! Avoids per-matvec thread spawn overhead across hundreds of transformer projections per token.
//! Falls back cleanly to single-threaded execution on `wasm32` or when `num_threads <= 1`.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Condvar, Mutex,
};
#[cfg(not(target_arch = "wasm32"))]
use std::thread::{self, JoinHandle};

#[cfg(not(target_arch = "wasm32"))]
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
    mutex: Mutex<PoolControl>,
    wake_workers: Condvar,
    wake_main: Condvar,
    remaining: AtomicUsize,
}

#[cfg(not(target_arch = "wasm32"))]
struct PoolControl {
    generation: u64,
    shutdown: bool,
    job: Option<Job>,
}

pub struct ThreadPool {
    num_threads: usize,
    #[cfg(not(target_arch = "wasm32"))]
    state: Option<Arc<SharedState>>,
    #[cfg(not(target_arch = "wasm32"))]
    workers: Vec<JoinHandle<()>>,
    /// Serializes concurrent callers to `parallel_for_rows` safely.
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
                mutex: Mutex::new(PoolControl {
                    generation: 0,
                    shutdown: false,
                    job: None,
                }),
                wake_workers: Condvar::new(),
                wake_main: Condvar::new(),
                remaining: AtomicUsize::new(0),
            });

            let num_workers = num_threads - 1; // Main thread also participates as shard 0
            let mut workers = Vec::with_capacity(num_workers);

            for worker_idx in 1..num_threads {
                let st = Arc::clone(&state);
                let total_threads = num_threads;
                let handle = thread::Builder::new()
                    .name(format!("caelum-worker-{worker_idx}"))
                    .spawn(move || {
                        let mut last_gen = 0u64;
                        loop {
                            let (func, call, total_items) = {
                                let mut guard = st.mutex.lock().unwrap();
                                while guard.generation == last_gen && !guard.shutdown {
                                    guard = st.wake_workers.wait(guard).unwrap();
                                }
                                if guard.shutdown {
                                    return;
                                }
                                last_gen = guard.generation;
                                let j = guard.job.as_ref().expect("job present on generation bump");
                                (j.func, j.call, j.total_items)
                            };

                            let (start, end) = partition_range(total_items, total_threads, worker_idx);
                            if start < end {
                                unsafe { call(func, start, end) };
                            }

                            if st.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                                let _guard = st.mutex.lock().unwrap();
                                st.wake_main.notify_one();
                            }
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
            state.remaining.store(num_workers, Ordering::Release);

            {
                let mut guard = state.mutex.lock().unwrap();
                guard.job = Some(Job {
                    func: (&f as *const F) as *const (),
                    call: trampoline::<F>,
                    total_items: total,
                });
                guard.generation = guard.generation.wrapping_add(1);
                state.wake_workers.notify_all();
            }

            let (start0, end0) = partition_range(total, self.num_threads, 0);
            if start0 < end0 {
                f(start0, end0);
            }

            if state.remaining.load(Ordering::Acquire) > 0 {
                let mut guard = state.mutex.lock().unwrap();
                while state.remaining.load(Ordering::Acquire) > 0 {
                    guard = state.wake_main.wait(guard).unwrap();
                }
                guard.job = None;
            } else {
                let mut guard = state.mutex.lock().unwrap();
                guard.job = None;
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
            let slice =
                unsafe { std::slice::from_raw_parts_mut((base_addr as *mut f32).add(start), end - start) };
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
            {
                let mut guard = state.mutex.lock().unwrap();
                guard.shutdown = true;
                state.wake_workers.notify_all();
            }
            for h in self.workers.drain(..) {
                let _ = h.join();
            }
        }
    }
}
