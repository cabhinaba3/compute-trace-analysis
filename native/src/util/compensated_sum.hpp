// Neumaier (1974) improved Kahan-Babushka compensated summation for
// order-stable floating-point accumulation across out-of-order streaming passes.
#pragma once

#include <cmath>

namespace cta {

class NeumaierSum {
 public:
  void add(long double x) {
    const long double t = sum_ + x;
    if (std::fabsl(sum_) >= std::fabsl(x)) {
      c_ += (sum_ - t) + x;
    } else {
      c_ += (x - t) + sum_;
    }
    sum_ = t;
  }

  long double value() const { return sum_ + c_; }
  double as_double() const { return static_cast<double>(sum_ + c_); }

 private:
  long double sum_ = 0.0L;
  long double c_ = 0.0L;
};

}  // namespace cta
