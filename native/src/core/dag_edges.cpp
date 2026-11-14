#include "core/dag_edges.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <thread>

#include "common/cpus.hpp"

namespace cta {
namespace {

std::uint64_t mix(std::uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Finds the vertices of a (rank, node id). In a table where each rank is one
// run of rows and node ids count 0, 1, 2, ... along the run (what the GOAL and
// Chakra readers produce), the vertex is the run's first row plus the node id.
// Otherwise a hash table finds them, chaining the rows that share a key.
class VertexIndex {
 public:
  VertexIndex(std::size_t n, const std::uint64_t* rank, const std::uint64_t* node_id) {
    dense_ = try_dense(n, rank, node_id);
    if (!dense_) build_hash(n, rank, node_id);
  }

  // Each thread keeps its own cursor: edges of one rank come together, so the
  // run found for the last edge is usually the one the next edge needs.
  struct Cursor {
    const void* run = nullptr;
  };

  // Calls f(vertex) for each row with this rank and node id.
  template <typename F>
  void for_each(std::uint64_t rank, std::uint64_t id, Cursor& cursor, F&& f) const {
    if (dense_) {
      const Run* run = static_cast<const Run*>(cursor.run);
      if (!run || run->rank != rank) {
        run = find_run(rank);
        cursor.run = run;
      }
      if (run && id < run->length) f(static_cast<std::uint32_t>(run->first + id));
      return;
    }
    if (slots_.empty()) return;
    const std::size_t mask = slots_.size() - 1;
    for (std::size_t i = mix(rank * 0x9e3779b97f4a7c15ULL ^ id) & mask; slots_[i].head != kNone; i = (i + 1) & mask) {
      if (slots_[i].rank == rank && slots_[i].id == id) {
        for (std::uint32_t v = slots_[i].head; v != kNone; v = next_[v]) f(v);
        return;
      }
    }
  }

 private:
  static constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();
  struct Run {
    std::uint64_t rank = 0;
    std::uint64_t first = 0, length = 0;
  };
  struct Slot {
    std::uint64_t rank = 0, id = 0;
    std::uint32_t head = kNone;
  };

  bool try_dense(std::size_t n, const std::uint64_t* rank, const std::uint64_t* node_id) {
    runs_.clear();
    for (std::size_t i = 0; i < n; ++i) {
      if (runs_.empty() || runs_.back().rank != rank[i]) {
        if (node_id[i] != 0) return false;
        runs_.push_back({rank[i], i, 1});
      } else {
        if (node_id[i] != runs_.back().length) return false;
        ++runs_.back().length;
      }
    }
    // A rank in two runs would need both; the hash path handles that.
    std::vector<std::uint64_t> seen;
    seen.reserve(runs_.size());
    for (const Run& r : runs_) seen.push_back(r.rank);
    std::sort(seen.begin(), seen.end());
    if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) return false;
    sorted_runs_ = runs_;
    std::sort(sorted_runs_.begin(), sorted_runs_.end(), [](const Run& a, const Run& b) { return a.rank < b.rank; });
    return true;
  }

  const Run* find_run(std::uint64_t rank) const {
    auto it = std::lower_bound(sorted_runs_.begin(), sorted_runs_.end(), rank,
                               [](const Run& r, std::uint64_t k) { return r.rank < k; });
    if (it == sorted_runs_.end() || it->rank != rank) return nullptr;
    return &*it;
  }

  void build_hash(std::size_t n, const std::uint64_t* rank, const std::uint64_t* node_id) {
    std::size_t capacity = 16;
    while (capacity < 2 * n) capacity *= 2;
    slots_.assign(capacity, Slot{});
    next_.assign(n, kNone);
    const std::size_t mask = capacity - 1;
    for (std::size_t v = 0; v < n; ++v) {
      std::size_t i = mix(rank[v] * 0x9e3779b97f4a7c15ULL ^ node_id[v]) & mask;
      while (slots_[i].head != kNone && !(slots_[i].rank == rank[v] && slots_[i].id == node_id[v])) i = (i + 1) & mask;
      if (slots_[i].head == kNone) {
        slots_[i].rank = rank[v];
        slots_[i].id = node_id[v];
      }
      next_[v] = slots_[i].head;  // newest first; the order of matches does not matter
      slots_[i].head = static_cast<std::uint32_t>(v);
    }
  }

  bool dense_ = false;
  std::vector<Run> runs_, sorted_runs_;
  std::vector<Slot> slots_;
  std::vector<std::uint32_t> next_;
};

struct Channel {
  std::int64_t src, dst, tag;
  bool operator==(const Channel& o) const { return src == o.src && dst == o.dst && tag == o.tag; }
};
std::size_t hash_of(const Channel& c) {
  return static_cast<std::size_t>(mix(static_cast<std::uint64_t>(c.src) * 0x9e3779b97f4a7c15ULL ^
                                      mix(static_cast<std::uint64_t>(c.dst) * 0xc2b2ae3d27d4eb4fULL ^
                                          static_cast<std::uint64_t>(c.tag))));
}

// Links the k-th send of each channel to its k-th receive, in row order.
void link_point_to_point(std::size_t n, const bool* is_send, const bool* is_recv, const std::int64_t* src_rank,
                         const std::int64_t* dst_rank, const std::int64_t* tag, EdgeList& out) {
  constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();
  std::size_t p2p = 0;
  for (std::size_t v = 0; v < n; ++v) p2p += (is_send[v] || is_recv[v]) ? 1 : 0;
  // An open-addressing table of channel numbers, at most half full. Its slots
  // hold only the number; the channel's key is kept in `keys`.
  std::size_t capacity = 16;
  while (capacity < 2 * p2p) capacity *= 2;
  const std::size_t mask = capacity - 1;
  std::vector<std::uint32_t> table(capacity, kNone);
  std::vector<Channel> keys;
  std::vector<std::uint32_t> channel(n, kNone);
  std::vector<std::uint32_t> sends, recvs;  // per channel: how many
  for (std::size_t v = 0; v < n; ++v) {
    if (!is_send[v] && !is_recv[v]) continue;
    const Channel key{src_rank[v], dst_rank[v], tag[v]};
    std::size_t i = hash_of(key) & mask;
    while (table[i] != kNone && !(keys[table[i]] == key)) i = (i + 1) & mask;
    if (table[i] == kNone) {
      table[i] = static_cast<std::uint32_t>(keys.size());
      keys.push_back(key);
      sends.push_back(0);
      recvs.push_back(0);
    }
    channel[v] = table[i];
    if (is_send[v]) ++sends[table[i]];
    if (is_recv[v]) ++recvs[table[i]];
  }
  // Where each channel's sends and receives start in the flat lists.
  const std::size_t channels = sends.size();
  std::vector<std::uint64_t> send_at(channels + 1, 0), recv_at(channels + 1, 0);
  for (std::size_t c = 0; c < channels; ++c) {
    send_at[c + 1] = send_at[c] + sends[c];
    recv_at[c + 1] = recv_at[c] + recvs[c];
  }
  std::vector<std::uint32_t> send_list(send_at[channels]), recv_list(recv_at[channels]);
  std::vector<std::uint64_t> send_fill(send_at.begin(), send_at.end() - 1), recv_fill(recv_at.begin(), recv_at.end() - 1);
  for (std::size_t v = 0; v < n; ++v) {
    if (channel[v] == kNone) continue;
    if (is_send[v]) send_list[send_fill[channel[v]]++] = static_cast<std::uint32_t>(v);
    if (is_recv[v]) recv_list[recv_fill[channel[v]]++] = static_cast<std::uint32_t>(v);
  }
  for (std::size_t c = 0; c < channels; ++c) {
    const std::uint32_t pairs = std::min(sends[c], recvs[c]);
    for (std::uint32_t k = 0; k < pairs; ++k) {
      out.src.push_back(send_list[send_at[c] + k]);
      out.dst.push_back(recv_list[recv_at[c] + k]);
    }
  }
}

}  // namespace

EdgeList build_dag_edges(std::size_t n, const std::uint64_t* rank, const std::uint64_t* node_id, const bool* is_send,
                         const bool* is_recv, const std::int64_t* src_rank, const std::int64_t* dst_rank,
                         const std::int64_t* tag, std::size_t m, const std::uint64_t* e_rank,
                         const std::uint64_t* e_src, const std::uint64_t* e_dst, int threads) {
  if (n >= std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("too many vertices");
  if (threads < 0) throw std::invalid_argument("threads must be >= 0");
  if (threads == 0) threads = usable_cpus();

  using Clock = std::chrono::steady_clock;
  auto seconds = [](Clock::duration d) { return std::chrono::duration<double>(d).count(); };
  const auto t0 = Clock::now();
  const VertexIndex index(n, rank, node_id);
  const auto t1 = Clock::now();

  // Dependencies, looked up in blocks on the threads; the blocks are joined in
  // order, so the result does not depend on how many there are.
  const std::size_t blocks = std::max<std::size_t>(1, std::min<std::size_t>(static_cast<std::size_t>(threads), m / 65536 + 1));
  std::vector<EdgeList> parts(blocks);
  auto run_block = [&](std::size_t b) {
    const std::size_t begin = m * b / blocks, end = m * (b + 1) / blocks;
    EdgeList& part = parts[b];
    part.src.reserve(end - begin);
    part.dst.reserve(end - begin);
    std::vector<std::uint32_t> from, to;
    VertexIndex::Cursor cursor;
    for (std::size_t i = begin; i < end; ++i) {
      from.clear();
      to.clear();
      index.for_each(e_rank[i], e_src[i], cursor, [&](std::uint32_t v) { from.push_back(v); });
      if (from.empty()) continue;
      index.for_each(e_rank[i], e_dst[i], cursor, [&](std::uint32_t v) { to.push_back(v); });
      for (std::uint32_t u : from)
        for (std::uint32_t v : to) {
          part.src.push_back(u);
          part.dst.push_back(v);
        }
    }
  };
  if (blocks == 1) {
    run_block(0);
  } else {
    std::vector<std::thread> pool;
    for (std::size_t b = 1; b < blocks; ++b) pool.emplace_back(run_block, b);
    run_block(0);
    for (auto& t : pool) t.join();
  }

  const auto t2 = Clock::now();
  EdgeList out;
  std::size_t total = 0;
  for (const EdgeList& p : parts) total += p.src.size();
  out.src.reserve(total + n / 4);
  out.dst.reserve(total + n / 4);
  for (EdgeList& p : parts) {
    out.src.insert(out.src.end(), p.src.begin(), p.src.end());
    out.dst.insert(out.dst.end(), p.dst.begin(), p.dst.end());
    p = EdgeList();
  }
  link_point_to_point(n, is_send, is_recv, src_rank, dst_rank, tag, out);
  const auto t3 = Clock::now();
  out.index_seconds = seconds(t1 - t0);
  out.dependency_seconds = seconds(t2 - t1);
  out.link_seconds = seconds(t3 - t2);
  return out;
}

}  // namespace cta
