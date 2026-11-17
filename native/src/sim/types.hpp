// The vocabulary of the simulator: times, op kinds, collective types and the
// categories the critical path is cut into.
#pragma once

#include <cstdint>
#include <limits>

namespace cta::sim {

// Simulated time in integer nanoseconds.
using Time = std::int64_t;
inline constexpr Time kNever = std::numeric_limits<Time>::max();
inline constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

// What an op does. The codes are the ones Python passes in.
enum class OpKind : std::uint8_t {
  kCalc = 0,  // occupies one CPU of its rank for its duration
  kSend = 1,
  kRecv = 2,
  kColl = 3,  // a collective, costed as a whole (lowering replaces it with sends and recvs)
  kNop = 4,   // finishes as soon as it is ready and holds nothing
};

// How an edge constrains its destination: kRequires waits for the source to
// finish, kIrequires only for it to start (GOAL's "irequires").
enum class DepKind : std::uint8_t { kRequires = 0, kIrequires = 1 };

// Chakra's CollectiveCommType numbering.
enum class CollType : std::int8_t {
  kAllReduce = 0,
  kReduce = 1,
  kAllGather = 2,
  kGather = 3,
  kScatter = 4,
  kBroadcast = 5,
  kAllToAll = 6,
  kReduceScatter = 7,
  kReduceScatterBlock = 8,
  kBarrier = 9,
};
inline constexpr int kNumCollTypes = 10;

// Where an op of a lowered graph came from.
enum class Role : std::uint8_t {
  kOriginal = 0,  // copied from the input graph
  kEntry = 1,     // start of a lowered collective: takes its incoming edges
  kStep = 2,      // a send, recv or reduction of a lowered collective
  kExit = 3,      // end of a lowered collective: gives its outgoing edges
};

// What a stretch of the critical path was spent on.
enum class Segment : std::uint8_t {
  kCompute = 0,     // a calc's own duration
  kNoise = 1,       // OS noise that stretched a CPU activity
  kOverhead = 2,    // CPU time to send or receive a message (o, O)
  kLatency = 3,     // time in flight (L, link latencies)
  kTransfer = 4,    // serialisation of the bytes (G, g, bytes / bandwidth)
  kContention = 5,  // extra transfer time from sharing links with other flows
  kCollective = 6,  // an analytically costed collective
  kIdle = 7,        // a stretch nothing explains (should not happen)
};

}  // namespace cta::sim
