// Pluggable streaming quantile sketch interface and Hybrid Exact-First wrapper.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cta::streaming::sketches {

// Abstract interface implemented by DDSketch, KLL, and Greenwald-Khanna.
class QuantileSketch {
 public:
  virtual ~QuantileSketch() = default;
  virtual void add(double value, std::uint64_t count = 1) = 0;
  virtual double quantile(double q) const = 0;
  virtual std::uint64_t count() const = 0;
  virtual std::string_view name() const = 0;
  virtual double error_bound() const = 0;
};

// Whether make_sketch knows `kind` (checked up front: a typo must not go unnoticed
// until the exact map overflows, which may never happen on a small trace).
inline bool is_known_sketch(const std::string& kind) {
  return kind == "ddsketch" || kind == "kll" || kind == "gk" || kind == "greenwald_khanna";
}

// Factory implemented after concrete sketch headers (see make_quantile_sketch below).
std::unique_ptr<QuantileSketch> make_sketch(const std::string& kind, double accuracy);

// Exact frequency map up to `exact_cap` distinct uint64 values; seamlessly
// spills to a pluggable QuantileSketch if distinct cardinality exceeds cap.
// While exact, matches numpy.percentile(..., method="linear") bit-for-bit.
class HybridQuantile {
 public:
  explicit HybridQuantile(std::string sketch_kind = "ddsketch", std::size_t exact_cap = 65536, double accuracy = 0.001)
      : sketch_kind_(std::move(sketch_kind)), exact_cap_(exact_cap), accuracy_(accuracy) {
    if (!is_known_sketch(sketch_kind_))
      throw std::invalid_argument("unknown quantile_sketch '" + sketch_kind_ + "'; use ddsketch, kll, or gk");
    if (exact_cap_ == 0) {
      sketch_ = make_sketch(sketch_kind_, accuracy_);
      spilled_ = true;
    }
  }

  void add(std::uint64_t value, std::uint64_t weight = 1) {
    if (weight == 0) return;
    total_ += weight;
    if (!spilled_) {
      auto it = exact_.find(value);
      if (it != exact_.end()) {
        it->second += weight;
        return;
      }
      if (exact_.size() < exact_cap_) {
        exact_.emplace(value, weight);
        return;
      }
      spill();
    }
    sketch_->add(static_cast<double>(value), weight);
  }

  double quantile(double q) const {
    if (total_ == 0) return 0.0;
    const double qc = std::clamp(q, 0.0, 1.0);
    if (spilled_) return sketch_->quantile(qc);

    std::vector<std::pair<std::uint64_t, std::uint64_t>> sorted(exact_.begin(), exact_.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    if (total_ == 1) return static_cast<double>(sorted.front().first);
    const double pos = qc * static_cast<double>(total_ - 1);
    const std::uint64_t lo_idx = static_cast<std::uint64_t>(std::floor(pos));
    const std::uint64_t hi_idx = static_cast<std::uint64_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo_idx);

    auto value_at = [&](std::uint64_t target) -> double {
      std::uint64_t acc = 0;
      for (const auto& [val, cnt] : sorted) {
        acc += cnt;
        if (target < acc) return static_cast<double>(val);
      }
      return static_cast<double>(sorted.back().first);
    };
    const double lo_val = value_at(lo_idx);
    if (lo_idx == hi_idx) return lo_val;
    const double hi_val = value_at(hi_idx);
    return lo_val + frac * (hi_val - lo_val);
  }

  std::uint64_t count() const { return total_; }
  bool is_exact() const { return !spilled_; }
  std::string_view sketch_name() const { return spilled_ ? sketch_->name() : std::string_view("exact"); }
  double error_bound() const { return spilled_ ? sketch_->error_bound() : 0.0; }

 private:
  void spill() {
    sketch_ = make_sketch(sketch_kind_, accuracy_);
    for (const auto& [val, cnt] : exact_) sketch_->add(static_cast<double>(val), cnt);
    exact_.clear();
    exact_.rehash(0);
    spilled_ = true;
  }

  std::string sketch_kind_;
  std::size_t exact_cap_ = 65536;
  double accuracy_ = 0.001;
  std::uint64_t total_ = 0;
  bool spilled_ = false;
  std::unordered_map<std::uint64_t, std::uint64_t> exact_;
  std::unique_ptr<QuantileSketch> sketch_;
};

}  // namespace cta::streaming::sketches
