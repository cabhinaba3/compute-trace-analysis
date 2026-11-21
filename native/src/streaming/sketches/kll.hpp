// KLL Sketch: Optimal Quantile Approximation in Streams
// (Zohar Karnin, Kevin Lang, Edo Liberty — FOCS 2016).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "streaming/sketches/quantile.hpp"
#include "util/util.hpp"

namespace cta::streaming::sketches {

// Space-optimal randomized rank-error quantile sketch with deterministic PRNG
// seeding for reproducible runs. Guarantees rank error <= eps with high prob.
class KLLSketch final : public QuantileSketch {
 public:
  explicit KLLSketch(double eps = 0.005, std::uint64_t seed = 0x51a7c4e9ULL)
      : eps_(eps), k_(compute_k(eps)), rng_state_(seed) {
    if (!(eps > 0.0 && eps < 0.5)) throw std::invalid_argument("KLL eps must be in (0, 0.5)");
    levels_.resize(1);
  }

  void add(double value, std::uint64_t count = 1) override {
    if (count == 0 || !std::isfinite(value)) return;
    // Decompose multiplicity into binary powers so weighted insertion stays
    // O(log count) instead of looping `count` times.
    std::size_t h = 0;
    std::uint64_t rem = count;
    while (rem > 0) {
      if (rem & 1ULL) {
        if (h >= levels_.size()) levels_.resize(h + 1);
        levels_[h].push_back(value);
      }
      rem >>= 1;
      ++h;
    }
    total_ += count;
    if (value < min_val_) min_val_ = value;
    if (value > max_val_) max_val_ = value;
    compress();
  }

  double quantile(double q) const override {
    if (total_ == 0) return 0.0;
    const double qc = std::clamp(q, 0.0, 1.0);
    if (qc == 0.0) return min_val_;
    if (qc == 1.0) return max_val_;

    std::vector<std::pair<double, std::uint64_t>> items;
    for (std::size_t h = 0; h < levels_.size(); ++h) {
      const std::uint64_t w = std::uint64_t{1} << h;
      for (double v : levels_[h]) items.emplace_back(v, w);
    }
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    const std::uint64_t target = static_cast<std::uint64_t>(std::floor(qc * static_cast<double>(total_ - 1)));
    std::uint64_t acc = 0;
    for (const auto& [v, w] : items) {
      acc += w;
      if (target < acc) return v;
    }
    return max_val_;
  }

  std::uint64_t count() const override { return total_; }
  std::string_view name() const override { return "kll"; }
  double error_bound() const override { return eps_; }

 private:
  static std::size_t compute_k(double eps) {
    // Practical KLL sizing: k ~= 2.5 / eps gives <= eps rank error with failure prob < 1e-4.
    return std::max<std::size_t>(16, static_cast<std::size_t>(std::ceil(2.5 / eps)));
  }

  std::size_t level_capacity(std::size_t h) const {
    const std::size_t H = levels_.size();
    const std::size_t depth = H - 1 - h;
    const double cap = static_cast<double>(k_) * std::pow(2.0 / 3.0, static_cast<double>(depth));
    return std::max<std::size_t>(2, static_cast<std::size_t>(std::ceil(cap)));
  }

  std::uint64_t next_bit() {
    rng_state_ += 0x9e3779b97f4a7c15ULL;
    return mix64(rng_state_) & 1ULL;
  }

  void compress() {
    for (std::size_t h = 0; h < levels_.size(); ++h) {
      if (levels_[h].size() < level_capacity(h)) continue;
      if (h + 1 == levels_.size()) levels_.emplace_back();
      auto& cur = levels_[h];
      std::sort(cur.begin(), cur.end());
      const std::size_t offset = static_cast<std::size_t>(next_bit());
      auto& next = levels_[h + 1];
      for (std::size_t i = offset; i < cur.size(); i += 2) next.push_back(cur[i]);
      cur.clear();
    }
  }

  double eps_;
  std::size_t k_;
  std::uint64_t rng_state_;
  std::uint64_t total_ = 0;
  double min_val_ = 1e300;
  double max_val_ = -1e300;
  std::vector<std::vector<double>> levels_;
};

}  // namespace cta::streaming::sketches
