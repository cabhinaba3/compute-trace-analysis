// Small helpers shared by the readers and writers.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace cta {

// A 64-bit integer mixer (splitmix64's finalizer), for hashing and seeding.
inline std::uint64_t mix64(std::uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Appends one vector to the end of another.
template <typename T>
void append_vector(std::vector<T>& to, const std::vector<T>& from) {
  to.insert(to.end(), from.begin(), from.end());
}

// Seconds in a steady_clock duration.
inline double duration_seconds(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double>(d).count();
}

// Whether a name contains "nccl" in any letter case: an NCCL kernel.
inline bool contains_nccl(std::string_view s) {
  for (std::size_t i = 0; i + 4 <= s.size(); ++i) {
    auto low = [&](std::size_t j) { return static_cast<char>(s[i + j] | 0x20); };
    if (low(0) == 'n' && low(1) == 'c' && low(2) == 'c' && low(3) == 'l') return true;
  }
  return false;
}

}  // namespace cta
