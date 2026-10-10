#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "detail/parallel.hpp"

namespace {

using turbocore::detail::parallel_for;
using turbocore::detail::resolve_thread_count;

TEST(Parallel, ThreadCountIsClampedToTheWork) {
  EXPECT_EQ(resolve_thread_count(4, 100), 4u);
  EXPECT_EQ(resolve_thread_count(8, 3), 3u);
  EXPECT_EQ(resolve_thread_count(8, 0), 1u);
  EXPECT_GE(resolve_thread_count(0, 1000), 1u);
}

TEST(Parallel, EveryIndexRunsExactlyOnce) {
  for (const unsigned threads : {1u, 2u, 3u, 8u, 0u}) {
    for (const std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{1000}}) {
      std::vector<std::atomic<int>> runs(count);
      parallel_for(count, threads, [&] {
        return [&](std::size_t index) { runs[index].fetch_add(1); };
      });
      for (std::size_t i = 0; i < count; ++i) {
        ASSERT_EQ(runs[i].load(), 1) << "threads=" << threads << " count=" << count << " index=" << i;
      }
    }
  }
}

TEST(Parallel, EachThreadGetsItsOwnWorkerState) {
  std::atomic<int> workers_created{0};
  std::atomic<std::size_t> total{0};
  parallel_for(500, 4, [&] {
    workers_created.fetch_add(1);
    // `scratch` is per worker: unsynchronised use must be safe (TSan would object otherwise).
    return [&, scratch = std::vector<std::size_t>()](std::size_t index) mutable {
      scratch.push_back(index);
      total.fetch_add(index);
    };
  });
  EXPECT_LE(workers_created.load(), 4);
  EXPECT_GE(workers_created.load(), 1);
  EXPECT_EQ(total.load(), std::size_t{500} * 499 / 2);
}

// Several tasks fail; whichever thread gets there first, the caller must see
// the failure a sequential loop would have hit first.
TEST(Parallel, LowestFailingIndexIsReportedRegardlessOfTiming) {
  for (int repeat = 0; repeat < 200; ++repeat) {
    for (const unsigned threads : {1u, 2u, 8u}) {
      try {
        parallel_for(64, threads, [] {
          return [](std::size_t index) {
            if (index == 13 || index == 14 || index == 40 || index == 63) {
              throw std::runtime_error(std::to_string(index));
            }
          };
        });
        FAIL() << "expected an exception";
      } catch (const std::runtime_error& error) {
        ASSERT_STREQ(error.what(), "13") << "threads=" << threads;
      }
    }
  }
}

TEST(Parallel, WorkStopsAfterAFailure) {
  std::atomic<std::size_t> started{0};
  EXPECT_THROW(parallel_for(100000, 4,
                            [&] {
                              return [&](std::size_t index) {
                                started.fetch_add(1);
                                if (index == 5) {
                                  throw std::runtime_error("boom");
                                }
                              };
                            }),
               std::runtime_error);
  EXPECT_LT(started.load(), std::size_t{100000});
}

}  // namespace
