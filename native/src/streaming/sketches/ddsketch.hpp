// DDSketch: A Fast and Fully-Mergeable Quantile Sketch with Relative-Error Guarantees
// (Charles Masson, Jee E. Rim, Homin K. Lee — PVLDB 2019).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "streaming/sketches/quantile.hpp"

namespace cta::streaming::sketches {

// Deterministic relative-error quantile sketch for non-negative values.
// Guarantees |q_hat - q| <= alpha * q for every value above min_value_.
class DDSketch final : public QuantileSketch {
 public:
  explicit DDSketch(double alpha = 0.001, double min_value = 1e-9)
      : alpha_(alpha),
        min_value_(min_value),
        gamma_((1.0 + alpha) / (1.0 - alpha)),
        inv_log_gamma_(1.0 / std::log(gamma_)) {
    if (!(alpha > 0.0 && alpha < 1.0)) throw std::invalid_argument("DDSketch alpha must be in (0, 1)");
  }

  void add(double value, std::uint64_t count = 1) override {
    if (count == 0 || !std::isfinite(value)) return;
    total_ += count;
    if (value <= min_value_) {
      zero_count_ += count;
      return;
    }
    const std::int64_t key = key_of(value);
    if (bins_.empty()) {
      min_key_ = key;
      bins_.assign(1, count);
      return;
    }
    if (key < min_key_) {
      const std::size_t shift = static_cast<std::size_t>(min_key_ - key);
      bins_.insert(bins_.begin(), shift, 0);
      min_key_ = key;
      bins_[0] += count;
      return;
    }
    const std::size_t idx = static_cast<std::size_t>(key - min_key_);
    if (idx >= bins_.size()) bins_.resize(idx + 1, 0);
    bins_[idx] += count;
  }

  double quantile(double q) const override {
    if (total_ == 0) return 0.0;
    const double qc = std::clamp(q, 0.0, 1.0);
    const std::uint64_t target = static_cast<std::uint64_t>(std::floor(qc * static_cast<double>(total_ - 1)));
    if (target < zero_count_) return 0.0;
    std::uint64_t acc = zero_count_;
    for (std::size_t i = 0; i < bins_.size(); ++i) {
      acc += bins_[i];
      if (target < acc) return value_of(min_key_ + static_cast<std::int64_t>(i));
    }
    return bins_.empty() ? 0.0 : value_of(min_key_ + static_cast<std::int64_t>(bins_.size() - 1));
  }

  void merge(const DDSketch& other) {
    if (other.total_ == 0) return;
    total_ += other.total_;
    zero_count_ += other.zero_count_;
    if (other.bins_.empty()) return;
    if (bins_.empty()) {
      min_key_ = other.min_key_;
      bins_ = other.bins_;
      return;
    }
    const std::int64_t new_min = std::min(min_key_, other.min_key_);
    const std::int64_t new_max = std::max(min_key_ + static_cast<std::int64_t>(bins_.size()),
                                          other.min_key_ + static_cast<std::int64_t>(other.bins_.size()));
    std::vector<std::uint64_t> merged(static_cast<std::size_t>(new_max - new_min), 0);
    const std::size_t off_a = static_cast<std::size_t>(min_key_ - new_min);
    const std::size_t off_b = static_cast<std::size_t>(other.min_key_ - new_min);
    for (std::size_t i = 0; i < bins_.size(); ++i) merged[off_a + i] += bins_[i];
    for (std::size_t i = 0; i < other.bins_.size(); ++i) merged[off_b + i] += other.bins_[i];
    min_key_ = new_min;
    bins_ = std::move(merged);
  }

  std::uint64_t count() const override { return total_; }
  std::string_view name() const override { return "ddsketch"; }
  double error_bound() const override { return alpha_; }

 private:
  std::int64_t key_of(double v) const { return static_cast<std::int64_t>(std::ceil(std::log(v) * inv_log_gamma_)); }

  double value_of(std::int64_t key) const {
    return 2.0 * std::pow(gamma_, static_cast<double>(key)) / (gamma_ + 1.0);
  }

  double alpha_;
  double min_value_;
  double gamma_;
  double inv_log_gamma_;
  std::uint64_t total_ = 0;
  std::uint64_t zero_count_ = 0;
  std::int64_t min_key_ = 0;
  std::vector<std::uint64_t> bins_;
};

}  // namespace cta::streaming::sketches
