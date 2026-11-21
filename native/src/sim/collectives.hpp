// Collective algorithms as per-member schedules of sends, receives and
// reductions. Lowering inlines them; the analytical level evaluates them.
//
// `bytes` is Chakra's comm_size: the whole buffer of an all-reduce, broadcast
// or reduce, the whole output of an all-gather or gather, the whole input of a
// reduce-scatter or scatter, and the whole send buffer of an all-to-all (B/P
// per peer, or B with all_to_all_per_peer). Chunk c of B bytes in P chunks is
// B/P bytes, plus one if c < B % P.
#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "sim/types.hpp"

namespace cta::sim {

enum class Algorithm : std::uint8_t {
  kRing,               // all-to-all: pairwise exchange, one peer per step
  kRecursiveDoubling,  // barrier: dissemination
  kHalvingDoubling,    // Rabenseifner
  kBinomialTree,
  kDoubleBinaryTree,   // NCCL's, pipelined in `chunks`
  kDirect,
  kHierarchical,       // intra-node ring, inter-node ring, intra-node ring
};
inline constexpr int kNumAlgorithms = 7;

// Throws std::invalid_argument for an unknown name.
Algorithm parse_algorithm(std::string_view name);

enum class ActionKind : std::uint8_t { kSend, kRecv, kReduce };

struct Action {
  ActionKind kind;
  std::uint32_t peer = 0;  // position in the group
  std::uint64_t bytes = 0;
  // The k-th message between two positions in one direction has seq k, at
  // both ends.
  std::uint32_t seq = 0;
  std::uint32_t deps_begin = 0, deps_end = 0;  // into Schedule::deps; earlier actions
};

struct Schedule {
  std::vector<Action> actions;
  std::vector<std::uint32_t> deps;
  std::vector<std::uint32_t> outputs;  // the actions that leave the member its result

  std::span<const std::uint32_t> deps_of(std::size_t a) const {
    return std::span<const std::uint32_t>(deps).subspan(actions[a].deps_begin,
                                                        actions[a].deps_end - actions[a].deps_begin);
  }
  std::vector<std::uint32_t> sinks() const;  // actions nothing waits for
};

struct CollSpec {
  CollType type = CollType::kAllReduce;
  Algorithm algorithm = Algorithm::kRing;
  std::uint32_t size = 1;
  std::uint64_t bytes = 0;
  std::uint32_t root = 0;
  std::uint32_t node_size = 1;  // hierarchical: positions [k n, (k + 1) n) share node k
  std::uint32_t chunks = 1;
  bool reduce = false;
  bool all_to_all_per_peer = false;
};

bool supports(CollType type, Algorithm algorithm, std::uint32_t size, std::uint32_t node_size);
// `wanted` if it supports the collective, else the type's default, which
// supports every size.
Algorithm resolve_algorithm(CollType type, Algorithm wanted, std::uint32_t size, std::uint32_t node_size,
                            bool* fell_back);

// Throws std::invalid_argument when the algorithm does not support the spec.
void build_schedule(const CollSpec& spec, std::uint32_t me, Schedule& out);

// Contention-free message costs; src and dst are global ranks.
class CostModel {
 public:
  virtual ~CostModel() = default;
  virtual Time send_busy(std::uint32_t src, std::uint32_t dst, std::uint64_t bytes) const = 0;
  virtual Time deliver(std::uint32_t src, std::uint32_t dst, std::uint64_t bytes) const = 0;
  virtual Time reduce(std::uint64_t bytes) const = 0;
};

// When each member finishes, all starting at 0, without contention: after its
// last action, or its outputs with `data_completion`. Above `exact_limit`
// actions in all, only member 0 is evaluated, assuming each message's sender
// starts when its receiver posts it, and every member gets its time.
std::vector<Time> evaluate_collective(const CollSpec& spec, std::span<const std::uint32_t> members,
                                      const CostModel& cost, bool data_completion,
                                      std::uint64_t exact_limit = 1u << 22);

}  // namespace cta::sim
