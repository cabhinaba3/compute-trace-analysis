// Replaces each collective op with its rank's actions of the collective's
// algorithm, so a Chakra graph runs at the level of a GOAL schedule.
#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "sim/collectives.hpp"
#include "sim/graph.hpp"
#include "sim/params.hpp"

namespace cta::sim {

// Lowered messages' tags: the top bit, the instance (43 bits), the seq (20).
inline constexpr std::uint64_t kCollectiveTagBit = std::uint64_t{1} << 63;

struct LoweringStats {
  std::uint64_t collectives = 0, fallbacks = 0, messages = 0, bytes = 0;

  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("collectives", collectives);
    visit("fallbacks", fallbacks);
    visit("messages", messages);
    visit("bytes", bytes);
  }
};

std::array<Algorithm, kNumCollTypes> algorithms_of(const SimParams& p);

// Members per node when the members come in node-aligned runs of one length
// (as hierarchical algorithms need), else 0.
std::uint32_t node_size_of(std::span<const std::uint32_t> members, std::uint32_t ranks_per_node);

// A collective becomes an entry NOP (taking its incoming edges and those of
// ops that irequire it), its rank's actions in its place, and an exit NOP
// (giving its requires edges) that waits for the rank's last actions, or for
// its outputs with completion "outputs".
GraphColumns lower_collectives(const GraphView& g, const Groups& groups, const SimParams& params,
                               LoweringStats& stats);

}  // namespace cta::sim
