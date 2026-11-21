// Space-Efficient Online Computation of Quantile Summaries
// (Michael Greenwald, Sanjeev Khanna — ACM SIGMOD 2001).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "streaming/sketches/quantile.hpp"

namespace cta::streaming::sketches {

// Deterministic worst-case eps-rank-error quantile summary with zero randomness.
// Separates exact multiplicity (`own`) from compressed predecessor gap (`g`) so
// the GK invariant `g + delta <= floor(2 * eps * N)` holds unconditionally even
// when a single value has huge multiplicity (e.g., repeated collective sizes).
class GKSketch final : public QuantileSketch {
 public:
  explicit GKSketch(double eps = 0.005) : eps_(eps) {
    if (!(eps > 0.0 && eps < 0.5)) throw std::invalid_argument("GK eps must be in (0, 0.5)");
    compress_every_ = std::max<std::uint64_t>(16, static_cast<std::uint64_t>(std::floor(1.0 / (2.0 * eps_))));
  }

  void add(double value, std::uint64_t count = 1) override {
    if (count == 0 || !std::isfinite(value)) return;
    if (entries_.empty()) {
      entries_.push_back({value, count, 0, 0});
      total_ += count;
      return;
    }
    const std::uint64_t threshold = max_span(total_ + count);
    auto it = std::lower_bound(entries_.begin(), entries_.end(), value,
                               [](const Tuple& t, double v) { return t.v < v; });
    if (it != entries_.end() && it->v == value) {
      it->own += count;
    } else if (it == entries_.begin() || it == entries_.end()) {
      entries_.insert(it, Tuple{value, count, 0, 0});
    } else {
      // Only compressed uncertainty (`it->g + it->delta`) can sit below `value`.
      const std::uint64_t parent_unc = it->g + it->delta;
      const std::uint64_t delta = parent_unc > 0 ? std::min<std::uint64_t>(threshold, parent_unc) : 0;
      entries_.insert(it, Tuple{value, count, 0, delta});
    }
    total_ += count;
    if (++inserts_ >= compress_every_) {
      inserts_ = 0;
      compress();
    }
  }

  double quantile(double q) const override {
    if (entries_.empty() || total_ == 0) return 0.0;
    const double qc = std::clamp(q, 0.0, 1.0);
    if (qc == 0.0) return entries_.front().v;
    if (qc == 1.0) return entries_.back().v;

    const std::uint64_t target = static_cast<std::uint64_t>(std::floor(qc * static_cast<double>(total_ - 1))) + 1;
    const std::uint64_t margin = static_cast<std::uint64_t>(std::ceil(eps_ * static_cast<double>(total_)));

    std::uint64_t cum = 0;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      const Tuple& t = entries_[i];
      const std::uint64_t exact_lo = cum + t.g + 1;
      const std::uint64_t exact_hi = cum + t.g + t.own;
      const std::uint64_t rmax = exact_hi + t.delta;
      // Target lands directly inside this value's exact multiplicity band.
      if (target >= exact_lo && target <= exact_hi) return t.v;
      // Target lands inside the compressed uncertainty band bounded by eps * N.
      if (rmax >= target && (exact_hi + margin >= target)) return t.v;
      cum = exact_hi;
    }
    return entries_.back().v;
  }

  std::uint64_t count() const override { return total_; }
  std::string_view name() const override { return "gk"; }
  double error_bound() const override { return eps_; }

 private:
  struct Tuple {
    double v = 0.0;
    std::uint64_t own = 0;    // exact copies of `v`
    std::uint64_t g = 0;      // compressed elements <= `v` merged from left
    std::uint64_t delta = 0;  // max rank uncertainty
  };

  std::uint64_t max_span(std::uint64_t n) const {
    return static_cast<std::uint64_t>(std::floor(2.0 * eps_ * static_cast<double>(n)));
  }

  static std::uint64_t band_of(std::uint64_t delta, std::uint64_t p) {
    if (delta >= p) return 0;
    const std::uint64_t diff = p - delta;
    std::uint64_t b = 1;
    std::uint64_t limit = 1;
    while (limit < diff) {
      limit = (limit << 1) | 1ULL;
      ++b;
    }
    return b;
  }

  void compress() {
    if (entries_.size() < 3) return;
    const std::uint64_t cap = max_span(total_);
    if (cap == 0) return;

    std::vector<Tuple> out;
    out.reserve(entries_.size());
    out.push_back(entries_.back());
    // Merge `cur` into `right`'s compressed gap only if total compressed span
    // stays within floor(2 * eps * N), preserving the deterministic GK bound.
    for (std::size_t i = entries_.size() - 1; i-- > 1;) {
      Tuple& right = out.back();
      const Tuple& cur = entries_[i];
      const std::uint64_t cur_mass = cur.own + cur.g;
      if (out.size() >= 2 && band_of(cur.delta, cap) <= band_of(right.delta, cap) &&
          cur_mass + right.g + right.delta <= cap) {
        right.g += cur_mass;
      } else {
        out.push_back(cur);
      }
    }
    out.push_back(entries_.front());
    std::reverse(out.begin(), out.end());
    entries_ = std::move(out);
  }

  double eps_;
  std::uint64_t compress_every_ = 100;
  std::uint64_t inserts_ = 0;
  std::uint64_t total_ = 0;
  std::vector<Tuple> entries_;
};

}  // namespace cta::streaming::sketches
