#ifndef TURBOCORE_DETAIL_PARALLEL_HPP
#define TURBOCORE_DETAIL_PARALLEL_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace turbocore::detail {

/// Number of threads to use for `tasks` independent tasks. `requested` == 0
/// means one per hardware thread. Never more threads than tasks, never zero.
[[nodiscard]] inline unsigned resolve_thread_count(unsigned requested, std::size_t tasks) noexcept {
  unsigned threads = requested != 0 ? requested : std::thread::hardware_concurrency();
  threads = std::max(threads, 1u);
  if (tasks < threads) {
    threads = static_cast<unsigned>(std::max<std::size_t>(tasks, 1));
  }
  return threads;
}

/// Runs task(index) for every index in [0, count) on `threads` threads
/// (resolved with resolve_thread_count).
///
/// `make_worker` is called once on each thread and returns the callable that
/// thread will use, so each thread owns its scratch state and tasks need no
/// locking as long as they write to disjoint outputs.
///
/// Indices are handed out in increasing order. If a task throws, no further
/// indices are handed out, the tasks already running are allowed to finish,
/// and the exception of the lowest failing index is rethrown on the calling
/// thread. Because every index below a failing one has already been started,
/// that is the same exception a sequential loop would have raised, regardless
/// of thread timing.
template <typename MakeWorker>
void parallel_for(std::size_t count, unsigned threads, MakeWorker&& make_worker) {
  threads = resolve_thread_count(threads, count);

  if (threads == 1) {
    auto task = make_worker();
    for (std::size_t index = 0; index < count; ++index) {
      task(index);
    }
    return;
  }

  std::atomic<std::size_t> next{0};
  std::atomic<bool> stop{false};
  std::mutex failure_mutex;
  std::size_t failed_index = count;
  std::exception_ptr failure;

  const auto record_failure = [&](std::size_t index) {
    const std::lock_guard lock(failure_mutex);
    if (index < failed_index) {
      failed_index = index;
      failure = std::current_exception();
    }
    stop.store(true);
  };

  const auto run = [&] {
    std::size_t index = count;
    try {
      auto task = make_worker();
      while (!stop.load()) {
        index = next.fetch_add(1);
        if (index >= count) {
          break;
        }
        task(index);
      }
    } catch (...) {
      // A failure in make_worker() has no index of its own; rank it last so
      // that a real task failure takes precedence.
      record_failure(std::min(index, count - 1));
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(threads - 1);
  try {
    for (unsigned t = 1; t < threads; ++t) {
      workers.emplace_back(run);
    }
  } catch (...) {
    // Could not start a thread: finish with the ones that did start.
  }
  run();  // the calling thread is one of the workers
  for (std::thread& worker : workers) {
    worker.join();
  }

  if (failure) {
    std::rethrow_exception(failure);
  }
}

}  // namespace turbocore::detail

#endif  // TURBOCORE_DETAIL_PARALLEL_HPP
