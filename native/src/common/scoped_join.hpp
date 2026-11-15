#pragma once

#include <thread>
#include <utility>
#include <vector>

namespace cta {

// Threads that are joined when it goes out of scope, so an exception in the
// code that started them does not leave a joinable std::thread to terminate.
class ScopedJoin {
 public:
  ScopedJoin() = default;
  ~ScopedJoin() { join_all(); }
  ScopedJoin(const ScopedJoin&) = delete;
  ScopedJoin& operator=(const ScopedJoin&) = delete;

  template <typename F, typename... Args>
  void emplace(F&& f, Args&&... args) {
    threads_.emplace_back(std::forward<F>(f), std::forward<Args>(args)...);
  }

  void join_all() {
    for (auto& t : threads_)
      if (t.joinable()) t.join();
  }

 private:
  std::vector<std::thread> threads_;
};

}  // namespace cta
