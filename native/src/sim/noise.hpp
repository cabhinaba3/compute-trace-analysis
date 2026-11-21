// OS noise: detours arrive on each CPU as a Poisson process and stretch the
// CPU activity they fall in (detours in idle time cost nothing). A CPU's
// detours depend only on the seed, so runs are reproducible.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "sim/types.hpp"
#include "util/util.hpp"

namespace cta::sim {

class OsNoise {
 public:
  // frequency: detours per ns; duration: mean length.
  OsNoise(double frequency, double duration, bool exponential, std::uint64_t seed, std::size_t cpus)
      : frequency_(frequency), duration_(duration), exponential_(exponential), seed_(seed),
        cpus_(frequency > 0 ? cpus : 0) {}

  // The detour time added to an activity of `length` starting at `start`;
  // starts on one CPU must not decrease.
  Time stretch(std::size_t cpu, Time start, Time length) {
    if (cpus_.empty()) return 0;
    Cpu& c = cpus_[cpu];
    if (c.k == 0) advance(cpu, c);
    while (c.at < static_cast<double>(start)) advance(cpu, c);
    double extra = 0;
    for (; c.at < static_cast<double>(start + length) + extra; advance(cpu, c)) extra += c.length;
    return std::llround(extra);
  }

 private:
  struct Cpu {
    double at = 0, length = 0;  // the next detour
    std::uint64_t k = 0;
  };

  double uniform(std::size_t cpu, std::uint64_t k, std::uint64_t which) const {  // in (0, 1]
    const std::uint64_t x = mix64(seed_ ^ mix64((static_cast<std::uint64_t>(cpu) << 32) ^ (k * 2 + which)));
    return (static_cast<double>(x >> 11) + 1.0) * 0x1.0p-53;
  }
  void advance(std::size_t cpu, Cpu& c) {
    c.at += -std::log(uniform(cpu, c.k, 0)) / frequency_;
    c.length = exponential_ ? -std::log(uniform(cpu, c.k, 1)) * duration_ : duration_;
    ++c.k;
  }

  double frequency_, duration_;
  bool exponential_;
  std::uint64_t seed_;
  std::vector<Cpu> cpus_;
};

}  // namespace cta::sim
