// Runs tasks on worker threads and hands the results to one merge function in
// the order the tasks were submitted, however they finish.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace cta {

// `In` is a task, `Out` its result. At most `max_in_flight` tasks (queued,
// running, or finished and waiting for their turn to merge) exist at a time,
// so memory stays bounded when tasks arrive faster than they finish.
//
// Each worker thread calls `make_work()` once and uses the function it gets
// for all its tasks, which is where per-thread state (a parser, a buffer)
// lives. The first exception from a task or from `merge` stops the pool and is
// rethrown by `submit` or `finish`.
template <typename In, typename Out>
class OrderedPool {
 public:
  using Work = std::function<Out(In&)>;

  OrderedPool(int workers, std::function<Work()> make_work, std::function<void(Out&&)> merge,
              std::size_t max_in_flight)
      : max_in_flight_(max_in_flight), make_work_(std::move(make_work)), merge_(std::move(merge)) {
    for (int i = 0; i < workers; ++i) workers_.emplace_back([this] { work(); });
    merge_thread_ = std::thread([this] { merge_loop(); });
  }
  ~OrderedPool() { stop(); }
  OrderedPool(const OrderedPool&) = delete;
  OrderedPool& operator=(const OrderedPool&) = delete;

  // Queues a task; blocks while the pool is full.
  void submit(In&& task) {
    std::unique_lock<std::mutex> lock(m_);
    room_.wait(lock, [&] { return aborted_ || next_submit_ - next_merge_ < max_in_flight_; });
    if (aborted_) {
      if (error_) std::rethrow_exception(error_);
      throw std::runtime_error("worker pool stopped");
    }
    queue_.emplace_back(next_submit_++, std::move(task));
    work_.notify_one();
  }

  // Waits until every task has been merged.
  void finish() {
    {
      std::lock_guard<std::mutex> lock(m_);
      closed_ = true;
    }
    work_.notify_all();
    result_.notify_all();
    for (auto& t : workers_) t.join();
    merge_thread_.join();
    if (error_) std::rethrow_exception(error_);
  }

 private:
  void fail(std::exception_ptr e) {
    {
      std::lock_guard<std::mutex> lock(m_);
      if (!error_) error_ = e;
      aborted_ = true;
    }
    work_.notify_all();
    room_.notify_all();
    result_.notify_all();
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(m_);
      aborted_ = true;
      closed_ = true;
    }
    work_.notify_all();
    room_.notify_all();
    result_.notify_all();
    for (auto& t : workers_)
      if (t.joinable()) t.join();
    if (merge_thread_.joinable()) merge_thread_.join();
  }

  void work() {
    Work run;
    try {
      run = make_work_();
    } catch (...) {
      fail(std::current_exception());
      return;
    }
    while (true) {
      std::pair<std::size_t, In> task;
      {
        std::unique_lock<std::mutex> lock(m_);
        work_.wait(lock, [&] { return aborted_ || !queue_.empty() || closed_; });
        if (aborted_ || queue_.empty()) return;
        task = std::move(queue_.front());
        queue_.pop_front();
      }
      try {
        Out out = run(task.second);
        task.second = In();  // free the input before the result waits to merge
        {
          std::lock_guard<std::mutex> lock(m_);
          done_.emplace(task.first, std::move(out));
        }
        result_.notify_all();
      } catch (...) {
        fail(std::current_exception());
        return;
      }
    }
  }

  void merge_loop() {
    while (true) {
      std::optional<Out> out;
      {
        std::unique_lock<std::mutex> lock(m_);
        result_.wait(lock, [&] {
          return aborted_ || done_.count(next_merge_) > 0 || (closed_ && next_merge_ == next_submit_);
        });
        if (aborted_) return;
        auto it = done_.find(next_merge_);
        if (it == done_.end()) return;  // closed, and everything is merged
        out.emplace(std::move(it->second));
        done_.erase(it);
        ++next_merge_;
      }
      room_.notify_all();
      try {
        merge_(std::move(*out));
      } catch (...) {
        fail(std::current_exception());
        return;
      }
    }
  }

  const std::size_t max_in_flight_;
  std::function<Work()> make_work_;
  std::function<void(Out&&)> merge_;
  std::mutex m_;
  std::condition_variable work_, room_, result_;
  std::deque<std::pair<std::size_t, In>> queue_;
  std::map<std::size_t, Out> done_;
  std::size_t next_submit_ = 0, next_merge_ = 0;
  bool closed_ = false, aborted_ = false;
  std::exception_ptr error_;
  std::vector<std::thread> workers_;
  std::thread merge_thread_;
};

}  // namespace cta
