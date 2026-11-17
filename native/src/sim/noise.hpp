// OS noise: each CPU suffers detours (interrupts, daemons) that arrive as a
// Poisson process. A CPU activity is stretched by the detours that start
// while it runs, including during the stretch itself; detours that fall in
// idle time cost nothing. The detours of a CPU depend only on the seed and
// the CPU, not on what else the simulation does, so runs are reproducible.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "sim/params.hpp"
#include "sim/types.hpp"
#include "util/util.hpp"

namespace cta::sim {

class OsNoise {
 public:
  OsNoise(const NoiseParams& params, std::uint64_t seed, std::size_t streams)
      : p_(params), seed_(seed), streams_(params.frequency > 0 ? streams : 0) {}

  bool enabled() const { return !streams_.empty(); }

  // Extra time detours add to an activity of `length` ns that starts at
  // `start` on CPU `stream`. Starts on one stream must not decrease.
  Time stretch(std::size_t stream, Time start, Time length) {
    if (streams_.empty()) return 0;
    Stream& s = streams_[stream];
    if (s.k == 0) advance(stream, s);
    while (s.at < static_cast<double>(start)) advance(stream, s);
    double extra = 0;
    while (s.at < static_cast<double>(start + length) + extra) {
      extra += s.length;
      advance(stream, s);
    }
    return static_cast<Time>(std::llround(extra));
  }

 private:
  struct Stream {
    double at = 0;      // start of the next detour
    double length = 0;  // and its length
    std::uint64_t k = 0;
  };

  // A uniform number in (0, 1] from (stream, k, which).
  double uniform(std::size_t stream, std::uint64_t k, std::uint64_t which) const {
    const std::uint64_t x = mix64(seed_ ^ mix64((static_cast<std::uint64_t>(stream) << 32) ^ (k * 2 + which)));
    return (static_cast<double>(x >> 11) + 1.0) * 0x1.0p-53;
  }
  void advance(std::size_t stream, Stream& s) {
    s.at += -std::log(uniform(stream, s.k, 0)) / p_.frequency;
    s.length = p_.exponential ? -std::log(uniform(stream, s.k, 1)) * p_.duration : p_.duration;
    ++s.k;
  }

  NoiseParams p_;
  std::uint64_t seed_;
  std::vector<Stream> streams_;
};

}  // namespace cta::sim
