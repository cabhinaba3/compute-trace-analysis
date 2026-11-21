// Codes the simulator shares with Python.
#pragma once

#include <array>
#include <cstdint>
#include <limits>

namespace cta::sim {

using Time = std::int64_t;  // nanoseconds
inline constexpr Time kNever = std::numeric_limits<Time>::max();
inline constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

enum class OpKind : std::uint8_t { kCalc, kSend, kRecv, kColl, kNop };
// kRequires waits for the source to finish, kIrequires only for it to start.
enum class DepKind : std::uint8_t { kRequires, kIrequires };

// Chakra's CollectiveCommType numbering.
enum class CollType : std::int8_t {
  kAllReduce, kReduce, kAllGather, kGather, kScatter, kBroadcast, kAllToAll, kReduceScatter,
  kReduceScatterBlock, kBarrier,
};
inline constexpr int kNumCollTypes = 10;
inline constexpr std::array<const char*, kNumCollTypes> kCollTypeNames = {
    "ALL_REDUCE", "REDUCE", "ALL_GATHER", "GATHER", "SCATTER", "BROADCAST", "ALL_TO_ALL", "REDUCE_SCATTER",
    "REDUCE_SCATTER_BLOCK", "BARRIER"};

// Where an op of a lowered graph came from.
enum class Role : std::uint8_t { kOriginal, kEntry, kStep, kExit };

// What a stretch of the critical path was spent on.
enum class Segment : std::uint8_t {
  kCompute, kNoise, kOverhead, kLatency, kTransfer, kContention, kCollective, kIdle,
};

}  // namespace cta::sim
