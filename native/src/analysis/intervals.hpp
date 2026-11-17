// Interval helpers shared by the trace readers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cta::analysis {

// For ranges sorted by start (longer first on equal starts), returns the index
// of each range's enclosing range: the latest earlier range that has not ended
// by this range's start, or -1.
std::vector<std::int64_t> nesting_parents(const std::int64_t* starts, const std::int64_t* ends,
                                          std::size_t n);

}  // namespace cta::analysis
