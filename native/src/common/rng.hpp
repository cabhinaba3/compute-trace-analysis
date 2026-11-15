// A deterministic random number generator.
#pragma once

#include <cstdint>

#include "common/util.hpp"

namespace cta {

// xoshiro256** seeded through splitmix64. Range mapping is done by hand
// (std::*_distribution output differs between standard libraries), so a seed
// gives the same bytes on every platform.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) {
    for (auto& w : s_) {
      seed += 0x9e3779b97f4a7c15ULL;
      w = mix64(seed);
    }
  }
  std::uint64_t next() {
    const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }
  // Uniform integer in [lo, hi].
  std::int64_t range(std::int64_t lo, std::int64_t hi) {
    return lo + static_cast<std::int64_t>(next() % static_cast<std::uint64_t>(hi - lo + 1));
  }
  // Uniform double in [0, 1).
  double unit() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

 private:
  static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  std::uint64_t s_[4];
};

}  // namespace cta
