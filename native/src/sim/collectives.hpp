// Collective algorithms as per-rank schedules of sends, receives and
// reductions, the one description both lowering (collectives become sends and
// recvs in the graph) and the analytical cost model (collectives stay whole)
// work from.
//
// A collective has a group of P members, numbered by their position 0..P-1 in
// the group. `bytes` is the buffer size as Chakra's comm_size gives it: the
// whole buffer of an all-reduce, broadcast or reduce, the whole output of an
// all-gather or gather, the whole input of a reduce-scatter or scatter, and
// the whole send buffer of an all-to-all (B/P to each peer). A buffer split
// into P chunks gives chunk c B/P bytes plus one if c < B % P, so no byte is
// lost or invented.
#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "sim/types.hpp"

namespace cta::sim {

enum class Algorithm : std::uint8_t {
  kRing = 0,               // rings (all-to-all: pairwise exchange, one peer per step)
  kRecursiveDoubling = 1,  // log2(P) exchange steps (barrier: dissemination)
  kHalvingDoubling = 2,    // Rabenseifner: recursive halving, then doubling
  kBinomialTree = 3,
  kDoubleBinaryTree = 4,   // NCCL's two complementary binary trees, pipelined
  kDirect = 5,             // every message at once, no steps
  kHierarchical = 6,       // intra-node ring, inter-node ring, intra-node ring
};
inline constexpr int kNumAlgorithms = 7;

const char* algorithm_name(Algorithm a);
const char* coll_type_name(CollType t);
// Throws std::invalid_argument for an unknown name.
Algorithm parse_algorithm(std::string_view name);

enum class ActionKind : std::uint8_t { kSend = 0, kRecv = 1, kReduce = 2 };

struct Action {
  ActionKind kind;
  std::uint32_t peer = 0;   // position of the other member (sends and recvs)
  std::uint64_t bytes = 0;  // bytes moved, or bytes reduced
  // A send and a recv pair up when they join the same two positions in the
  // same direction and have the same seq: the k-th message from A to B.
  std::uint32_t seq = 0;
  std::uint32_t deps_begin = 0, deps_end = 0;  // into Schedule::deps
};

// The actions of one member. An action waits for the actions listed as its
// deps (earlier indices only); actions without deps start with the collective.
struct Schedule {
  std::vector<Action> actions;
  std::vector<std::uint32_t> deps;
  // The actions that leave the member with its result. Empty when it has the
  // result from the start (the root of a broadcast).
  std::vector<std::uint32_t> outputs;

  void clear();
  std::span<const std::uint32_t> deps_of(std::size_t a) const {
    return std::span<const std::uint32_t>(deps).subspan(actions[a].deps_begin,
                                                        actions[a].deps_end - actions[a].deps_begin);
  }
  // Actions no other action waits for.
  std::vector<std::uint32_t> sinks() const;
};

struct CollSpec {
  CollType type = CollType::kAllReduce;
  Algorithm algorithm = Algorithm::kRing;
  std::uint32_t size = 1;       // P
  std::uint64_t bytes = 0;
  std::uint32_t root = 0;       // position of the root of rooted collectives
  // Members per node for kHierarchical; positions must be node-major
  // (positions [k * node_size, (k + 1) * node_size) share node k).
  std::uint32_t node_size = 1;
  std::uint32_t chunks = 1;     // pipeline chunks (ring broadcast/reduce, double binary tree)
  bool reduce = false;          // add a reduction action after each received partial result
};

bool supports(CollType type, Algorithm algorithm, std::uint32_t size, std::uint32_t node_size);
Algorithm default_algorithm(CollType type);
// `wanted` if it supports the collective, else the type's default (which
// supports every size); sets *fell_back when it had to fall back.
Algorithm resolve_algorithm(CollType type, Algorithm wanted, std::uint32_t size,
                            std::uint32_t node_size, bool* fell_back);

// Builds member `me`'s schedule into `out`. Throws std::invalid_argument when
// the algorithm does not support the collective (see resolve_algorithm).
void build_schedule(const CollSpec& spec, std::uint32_t me, Schedule& out);

// Contention-free costs of single messages, in nanoseconds; src and dst are
// global ranks.
class CostModel {
 public:
  virtual ~CostModel() = default;
  // From a send's start until it completes locally.
  virtual Time send_busy(std::uint32_t src, std::uint32_t dst, std::uint64_t bytes) const = 0;
  // From a send's start until the receiver has the message.
  virtual Time deliver(std::uint32_t src, std::uint32_t dst, std::uint64_t bytes) const = 0;
  virtual Time reduce(std::uint64_t bytes) const = 0;
};

// When each member finishes the collective, all of them starting at time 0,
// with no contention between messages. `members` maps positions to global
// ranks. A member finishes with its last action, or with its outputs when
// `data_completion` is set. Groups whose schedules would hold more than
// `exact_limit` actions in all are evaluated for member 0 alone, assuming each
// message's sender starts when the receiver posts it (exact for the symmetric
// algorithms on a uniform network), and every member gets member 0's time.
std::vector<Time> evaluate_collective(const CollSpec& spec, std::span<const std::uint32_t> members,
                                      const CostModel& cost, bool data_completion,
                                      std::uint64_t exact_limit = 1u << 22);

}  // namespace cta::sim
