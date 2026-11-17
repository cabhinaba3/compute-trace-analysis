// Lowering: replaces each collective op with the sends, recvs and reductions
// its algorithm gives its rank, so a Chakra graph runs at the same level as a
// GOAL schedule.
#pragma once

#include <cstdint>
#include <span>

#include "sim/graph.hpp"
#include "sim/params.hpp"

namespace cta::sim {

// The top bit marks the tags of lowered messages, so they never match a
// message of the trace. Below it: the collective instance (43 bits) and the
// message's seq (20 bits).
inline constexpr std::uint64_t kCollectiveTagBit = std::uint64_t{1} << 63;

struct LoweringStats {
  std::uint64_t collectives = 0;  // collective ops replaced
  std::uint64_t fallbacks = 0;    // of which ran the type's default algorithm instead
  std::uint64_t messages = 0;     // sends added
  std::uint64_t bytes = 0;        // bytes of the sends added

  template <typename Visit>
  void for_each_column(Visit&& visit) {
    visit("collectives", collectives);
    visit("fallbacks", fallbacks);
    visit("messages", messages);
    visit("bytes", bytes);
  }
};

// Members per node of a group, for hierarchical algorithms: n when the
// members, in group order, come in runs of n ranks that share a node and no
// node appears in two runs; 0 when they do not.
std::uint32_t node_size_of(std::span<const std::uint32_t> members, std::uint32_t ranks_per_node);

// The lowered graph. Each collective becomes an entry NOP (which takes the
// collective's incoming edges and the edges of ops that irequire it), its
// rank's actions in schedule order, and an exit NOP (which gives the
// collective's requires edges); the actions sit where the collective sat, so
// the order of ops per rank is kept. Ops without dependencies inside the
// schedule wait for the entry; the exit waits for the rank's last actions (or
// its outputs, with data_completion).
//
// Throws std::invalid_argument for an invalid graph, a collective whose rank
// is not in its group, or a collective with more than 2^20 messages between
// two members.
GraphColumns lower_collectives(const GraphView& g, const Groups& groups, const CollectiveParams& params,
                               std::uint32_t ranks_per_node, LoweringStats& stats);

}  // namespace cta::sim
