#include "intervals.hpp"

namespace cta {

std::vector<std::int64_t> nesting_parents(const std::int64_t* starts, const std::int64_t* ends,
                                          std::size_t n) {
  std::vector<std::int64_t> parent(n, -1);
  std::vector<std::size_t> open;
  for (std::size_t i = 0; i < n; ++i) {
    while (!open.empty() && ends[open.back()] <= starts[i]) open.pop_back();
    if (!open.empty()) parent[i] = static_cast<std::int64_t>(open.back());
    open.push_back(i);
  }
  return parent;
}

}  // namespace cta
