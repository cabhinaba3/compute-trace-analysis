#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "util/util.hpp"

namespace cta {

// ---------------------------------------------------------------------------
// Open-addressing map from a correlation id to the CUDA runtime call that
// carried it. `put` lets a later call with the same id replace the earlier
// one; `put_if_absent` keeps the first.
class LaunchMap {
 public:
  void put(std::int64_t key, std::int64_t start, std::int64_t thread) {
    if (key == kEmpty) return;
    if ((size_ + 1) * 2 > slots_.size()) grow();
    Slot& s = slots_[probe(key)];
    if (s.key == kEmpty) {
      s.key = key;
      ++size_;
    }
    s.start = start;
    s.thread = thread;
  }
  struct Hit {
    std::int64_t start;
    std::int64_t thread;
  };
  void put_if_absent(std::int64_t key, std::int64_t start, std::int64_t thread) {
    Hit existing{};
    if (!find(key, existing)) put(key, start, thread);
  }
  bool find(std::int64_t key, Hit& hit) const {
    if (slots_.empty() || key == kEmpty) return false;
    const Slot& s = slots_[probe(key)];
    if (s.key == kEmpty) return false;
    hit = {s.start, s.thread};
    return true;
  }

 private:
  static constexpr std::int64_t kEmpty = std::numeric_limits<std::int64_t>::min();
  struct Slot {
    std::int64_t key = kEmpty;
    std::int64_t start = 0;
    std::int64_t thread = 0;
  };
  std::size_t probe(std::int64_t key) const {
    const std::size_t mask = slots_.size() - 1;
    std::size_t i = mix64(static_cast<std::uint64_t>(key)) & mask;
    while (slots_[i].key != kEmpty && slots_[i].key != key) i = (i + 1) & mask;
    return i;
  }
  void grow() {
    std::vector<Slot> old = std::move(slots_);
    slots_.assign(old.empty() ? 1024 : old.size() * 2, Slot{});
    size_ = 0;
    for (const Slot& s : old) {
      if (s.key != kEmpty) {
        slots_[probe(s.key)] = s;
        ++size_;
      }
    }
  }
  std::vector<Slot> slots_;
  std::size_t size_ = 0;
};

}  // namespace cta
