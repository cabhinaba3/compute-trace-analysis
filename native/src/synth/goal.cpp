#include "synth/goal.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/format.hpp"
#include "common/sink.hpp"
#include "synth/goal_common.hpp"

namespace cta {
namespace {

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

struct BinEmitter : RankGraph {
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
  const bool ring = algorithm == "ring";
  return goal_bin::write_file(path, num_ranks, [&](int rank, RankGraph& g) {
    if (steps <= 0) return;
    BinEmitter e;
    generate_rank(rank, world_size, compute_ns, allreduce_bytes, steps, ring, e);
    g = std::move(e);
  });
}

}  // namespace cta
