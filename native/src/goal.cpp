#include "goal.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common.hpp"

namespace cta {
namespace {

enum class Op : std::int8_t { kSend = 1, kRecv = 2, kCalc = 3 };

// Walks the schedule of one rank and reports it to an emitter. The order of the
// calls is the line order of the text file, so both writers share one source of
// truth. Labels are 1-based ("l1", "l2", ...) in node-creation order.
template <typename Emitter>
void generate_rank(int rank, int world_size, std::int64_t compute_ns,
                   std::int64_t allreduce_bytes, int steps, bool ring, Emitter& e) {
  int label_count = 0;
  int previous = 0;  // 0 = none
  auto next_label = [&] { return ++label_count; };
  for (int step = 0; step < steps; ++step) {
    const int calc = next_label();
    e.calc(calc, compute_ns);
    if (previous) e.requires_(calc, previous);
    previous = calc;
    if (world_size <= 1) continue;
    if (ring) {
      const std::int64_t chunk = allreduce_bytes / world_size;
      const int next_rank = (rank + 1) % world_size;
      const int prev_rank = (rank - 1 + world_size) % world_size;
      for (int phase = 0; phase < 2 * (world_size - 1); ++phase) {
        const int tag = step * 100 + phase;
        const int send = next_label();
        const int recv = next_label();
        e.send(send, chunk, next_rank, tag);
        e.recv(recv, chunk, prev_rank, tag);
        e.requires_(send, previous);
        e.requires_(recv, previous);
        previous = recv;
      }
    } else {
      const int barrier = next_label();
      e.calc(barrier, 0);
      for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) continue;
        const int tag = step * 100 + 1;
        const int send = next_label();
        const int recv = next_label();
        e.send(send, allreduce_bytes, peer, tag);
        e.recv(recv, allreduce_bytes, peer, tag);
        e.requires_(send, previous);
        e.requires_(recv, previous);
        e.requires_(barrier, send);
        e.requires_(barrier, recv);
      }
      previous = barrier;
    }
  }
}

struct TextEmitter {
  Sink& sink;
  std::string line;
  void label(int n) { line.push_back('l'); append_int(line, n); }
  void flush() { line.push_back('\n'); sink.put(line); line.clear(); }
  void calc(int n, std::int64_t ns) {
    label(n); line.append(": calc "); append_int(line, ns); line.append(" cpu 0"); flush();
  }
  void send(int n, std::int64_t b, int peer, int tag) {
    label(n); line.append(": send "); append_int(line, b); line.append("b to ");
    append_int(line, peer); line.append(" tag "); append_int(line, tag);
    line.append(" cpu 0 nic 0"); flush();
  }
  void recv(int n, std::int64_t b, int peer, int tag) {
    label(n); line.append(": recv "); append_int(line, b); line.append("b from ");
    append_int(line, peer); line.append(" tag "); append_int(line, tag);
    line.append(" cpu 0 nic 0"); flush();
  }
  void requires_(int dst, int src) {
    label(dst); line.append(" requires "); label(src); flush();
  }
};

struct NodeRec {
  Op op;
  std::int64_t size;
  std::uint32_t peer;
  std::uint32_t tag;
};

struct BinEmitter {
  std::vector<NodeRec> nodes;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;  // src -> dst
  void calc(int, std::int64_t ns) { nodes.push_back({Op::kCalc, ns, 0, 0}); }
  void send(int, std::int64_t b, int peer, int tag) {
    nodes.push_back({Op::kSend, b, static_cast<std::uint32_t>(peer), static_cast<std::uint32_t>(tag)});
  }
  void recv(int, std::int64_t b, int peer, int tag) {
    nodes.push_back({Op::kRecv, b, static_cast<std::uint32_t>(peer), static_cast<std::uint32_t>(tag)});
  }
  // "dst requires src": edge src -> dst, labels are node id + 1.
  void requires_(int dst, int src) {
    edges.emplace_back(static_cast<std::uint32_t>(src - 1), static_cast<std::uint32_t>(dst - 1));
  }
};

constexpr std::size_t kRecordBytes = 39;  // packed node struct of LogGOPSim's Parser.hpp
constexpr std::uint32_t kNoRequiresAppendix = 999;
constexpr std::uint32_t kNoIrequiresAppendix = 0xFFFFFFFFu;

template <typename T>
void put_le(std::string& out, T v) {
  char b[sizeof(T)];
  std::memcpy(b, &v, sizeof(T));  // host is little-endian (x86_64 / aarch64 Linux)
  out.append(b, sizeof(T));
}

std::string rank_blob(const BinEmitter& r) {
  const std::uint32_t n = static_cast<std::uint32_t>(r.nodes.size());
  std::string blob;
  if (n == 0) {
    put_le<std::uint32_t>(blob, 0);
    put_le<std::uint32_t>(blob, 0);
    return blob;
  }
  std::vector<std::uint32_t> in_degree(n, 0);
  std::vector<std::vector<std::uint32_t>> succ(n);
  for (const auto& [s, d] : r.edges) {
    ++in_degree[d];
    succ[s].push_back(d);
  }
  std::vector<std::uint32_t> roots;
  for (std::uint32_t i = 0; i < n; ++i)
    if (in_degree[i] == 0) roots.push_back(i);

  std::string records, appendix;
  std::uint32_t appendix_len = 0;
  records.reserve(static_cast<std::size_t>(n) * kRecordBytes);
  for (std::uint32_t i = 0; i < n; ++i) {
    const NodeRec& nr = r.nodes[i];
    put_le<std::uint32_t>(records, in_degree[i]);
    put_le<std::int8_t>(records, static_cast<std::int8_t>(nr.op));
    put_le<std::uint32_t>(records, nr.peer);
    put_le<std::uint64_t>(records, static_cast<std::uint64_t>(nr.size));
    put_le<std::uint32_t>(records, nr.tag);
    put_le<std::uint8_t>(records, 0);  // proc
    put_le<std::uint8_t>(records, 0);  // nic
    put_le<std::uint32_t>(records, static_cast<std::uint32_t>(succ[i].size()));
    put_le<std::uint32_t>(records, succ[i].empty() ? kNoRequiresAppendix : appendix_len);
    for (std::uint32_t d : succ[i]) put_le<std::uint32_t>(appendix, d);
    appendix_len += static_cast<std::uint32_t>(succ[i].size());
    put_le<std::uint32_t>(records, 0);  // num_startdeps (no irequires)
    put_le<std::uint32_t>(records, kNoIrequiresAppendix);
  }
  put_le<std::uint32_t>(blob, n);
  put_le<std::uint32_t>(blob, static_cast<std::uint32_t>(roots.size()));
  for (std::uint32_t id : roots) put_le<std::uint32_t>(blob, id);
  blob += records;
  blob += appendix;
  return blob;
}

}  // namespace

void write_goal_allreduce(const std::string& path, int world_size, std::int64_t compute_ns,
                          std::int64_t allreduce_bytes, int steps, const std::string& algorithm) {
  Sink sink(path);
  TextEmitter e{sink, {}};
  // Same bytes as "\n".join(["num_ranks N", "", "rank 0 {", ..., "}", "", ...]).
  sink.put("num_ranks " + std::to_string(world_size) + "\n");
  for (int rank = 0; rank < world_size; ++rank) {
    sink.put("\nrank " + std::to_string(rank) + " {\n");
    generate_rank(rank, world_size, compute_ns, allreduce_bytes, steps, algorithm == "ring", e);
    sink.put("}\n");
  }
  sink.close();
}

std::uint64_t write_goal_allreduce_bin(const std::string& path, int world_size,
                                       std::int64_t compute_ns, std::int64_t allreduce_bytes,
                                       int steps, const std::string& algorithm) {
  // The Python compiler derives the rank count from the nodes it parsed, so a
  // schedule without nodes is one empty rank.
  const int num_ranks = steps > 0 ? world_size : 1;
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot open " + path);
  auto check = [&](bool ok) { if (!ok) { std::fclose(f); throw std::runtime_error("write failed"); } };

  std::string head;
  put_le<std::uint64_t>(head, 4223);  // GOAL_MAGIC_COOKIE
  put_le<std::uint32_t>(head, static_cast<std::uint32_t>(num_ranks));
  put_le<std::uint8_t>(head, 0);
  put_le<std::uint8_t>(head, 0);
  check(std::fwrite(head.data(), 1, head.size(), f) == head.size());
  std::string table(static_cast<std::size_t>(num_ranks) * 16, '\0');
  check(std::fwrite(table.data(), 1, table.size(), f) == table.size());

  std::uint64_t offset = 6 + 16ull * static_cast<std::uint64_t>(num_ranks);
  std::uint64_t total_nodes = 0;
  std::string offsets;
  for (int rank = 0; rank < num_ranks; ++rank) {
    BinEmitter r;
    if (steps > 0)
      generate_rank(rank, world_size, compute_ns, allreduce_bytes, steps, algorithm == "ring", r);
    total_nodes += r.nodes.size();
    const std::string blob = rank_blob(r);
    check(std::fwrite(blob.data(), 1, blob.size(), f) == blob.size());
    put_le<std::uint64_t>(offsets, offset);
    put_le<std::uint64_t>(offsets, offset + blob.size());
    offset += blob.size();
  }
  check(std::fseek(f, static_cast<long>(head.size()), SEEK_SET) == 0);
  check(std::fwrite(offsets.data(), 1, offsets.size(), f) == offsets.size());
  check(std::fclose(f) == 0);
  return total_nodes;
}

}  // namespace cta
