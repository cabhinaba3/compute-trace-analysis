// What the chunk parser produces from one chunk of events.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "util/interner.hpp"

namespace cta::readers::kineto {

using std::int32_t;
using std::int64_t;

// Bytes after a chunk's text that the JSON parser may read.
constexpr std::size_t kChunkPadding = 64;

// A pid or tid: absent/null, an integer, or text (v = id in the tag interner).
struct Tag {
  std::int8_t kind = 0;
  int64_t v = 0;
  bool operator==(const Tag& o) const { return kind == o.kind && v == o.v; }
};
struct ThreadKey {
  Tag pid, tid;
  bool operator==(const ThreadKey& o) const { return pid == o.pid && tid == o.tid; }
};
struct ThreadKeyHash {
  std::size_t operator()(const ThreadKey& k) const {
    std::uint64_t h = static_cast<std::uint64_t>(k.pid.v) * 0x9e3779b97f4a7c15ULL;
    h ^= static_cast<std::uint64_t>(k.tid.v) + 0x7f4a7c15 + (h << 6) + (h >> 2);
    h ^= static_cast<std::uint64_t>(k.pid.kind) * 31 + static_cast<std::uint64_t>(k.tid.kind);
    return static_cast<std::size_t>(h);
  }
};

// ---------------------------------------------------------------------------
// What one parsed chunk holds. Strings and thread keys are local to the chunk
// and renumbered when the chunk is merged.
struct ChunkOut {
  Interner names;
  std::vector<std::uint8_t> nccl;  // per local name: contains "nccl", any case
  Interner tags;                   // text pids and tids
  std::vector<ThreadKey> threads;
  std::unordered_map<ThreadKey, int32_t, ThreadKeyHash> thread_index;

  std::vector<int32_t> g_name;
  std::vector<std::int8_t> g_cat;
  std::vector<int64_t> g_start, g_end, g_bytes, g_stream, g_device, g_corr;

  // Collective metadata of the GPU rows that have it: row index within this chunk's g_*.
  std::vector<int64_t> c_row, c_in, c_out, c_size;
  std::vector<int32_t> c_op, c_dtype, c_group;

  std::vector<int64_t> o_start, o_end;
  std::vector<int32_t> o_name, o_thread;

  std::vector<int64_t> s_id, s_start, s_end;

  std::vector<int32_t> r_name;  // runtime names in first-seen order
  std::vector<int64_t> r_dur, r_calls;
  std::vector<int32_t> r_slot;  // local name code -> index into r_*, or -1

  std::vector<int64_t> l_corr, l_start;
  std::vector<int32_t> l_thread;

  // Every runtime call: name code, time, thread code and correlation id (-1 if none).
  std::vector<int32_t> rc_name, rc_thread;
  std::vector<int64_t> rc_start, rc_end, rc_corr;

  int32_t name_code(std::string_view s);
  int32_t thread_code(const Tag& pid, const Tag& tid);
};

}  // namespace cta::readers::kineto
