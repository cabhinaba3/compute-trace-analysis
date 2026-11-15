// The SPCL LogGOPSim `.bin` schedule layout, shared by every GOAL generator.
#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cta {

enum class Op : std::int8_t { kSend = 1, kRecv = 2, kCalc = 3 };

struct NodeRec {
  Op op;
  std::int64_t size;
  std::uint32_t peer;
  std::uint32_t tag;
};

// One rank's nodes and "requires" edges (edge: source -> dependent).
struct RankGraph {
  std::vector<NodeRec> nodes;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;
};

namespace goal_bin {

constexpr std::size_t kRecordBytes = 39;  // packed node struct of LogGOPSim's Parser.hpp
constexpr std::uint32_t kNoRequiresAppendix = 999;
constexpr std::uint32_t kNoIrequiresAppendix = 0xFFFFFFFFu;
constexpr std::uint64_t kMagicCookie = 4223;

template <typename T>
void put_le(std::string& out, T v) {
  char b[sizeof(T)];
  std::memcpy(b, &v, sizeof(T));  // host is little-endian (x86_64 Linux)
  out.append(b, sizeof(T));
}

// One rank's blob: node count, root count, root ids, 39-byte records, appendix.
inline std::string rank_blob(const RankGraph& r) {
  const std::uint32_t n = static_cast<std::uint32_t>(r.nodes.size());
  std::string blob;
  if (n == 0) {
    put_le<std::uint32_t>(blob, 0);
    put_le<std::uint32_t>(blob, 0);
    return blob;
  }
  std::vector<std::uint32_t> in_degree(n, 0);
  // CSR successor lists in edge order (a counting sort), not one vector per node.
  std::vector<std::uint32_t> row(static_cast<std::size_t>(n) + 1, 0);
  for (const auto& [s, d] : r.edges) {
    ++in_degree[d];
    ++row[s + 1];
  }
  for (std::uint32_t i = 0; i < n; ++i) row[i + 1] += row[i];
  std::vector<std::uint32_t> succ(r.edges.size());
  {
    std::vector<std::uint32_t> fill(row.begin(), row.end() - 1);
    for (const auto& [s, d] : r.edges) succ[fill[s]++] = d;
  }
  std::vector<std::uint32_t> roots;
  for (std::uint32_t i = 0; i < n; ++i)
    if (in_degree[i] == 0) roots.push_back(i);

  std::string records, appendix;
  records.reserve(static_cast<std::size_t>(n) * kRecordBytes);
  appendix.reserve(r.edges.size() * 4);
  std::uint32_t appendix_len = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    const NodeRec& nr = r.nodes[i];
    const std::uint32_t count = row[i + 1] - row[i];
    put_le<std::uint32_t>(records, in_degree[i]);
    put_le<std::int8_t>(records, static_cast<std::int8_t>(nr.op));
    put_le<std::uint32_t>(records, nr.peer);
    put_le<std::uint64_t>(records, static_cast<std::uint64_t>(nr.size));
    put_le<std::uint32_t>(records, nr.tag);
    put_le<std::uint8_t>(records, 0);  // proc
    put_le<std::uint8_t>(records, 0);  // nic
    put_le<std::uint32_t>(records, count);
    put_le<std::uint32_t>(records, count == 0 ? kNoRequiresAppendix : appendix_len);
    for (std::uint32_t k = row[i]; k < row[i + 1]; ++k) put_le<std::uint32_t>(appendix, succ[k]);
    appendix_len += count;
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

// Writes a whole `.bin` file. fill_rank(rank, RankGraph&) builds each rank in turn;
// only one rank is in memory at a time. Returns the total node count.
template <typename FillRank>
std::uint64_t write_file(const std::string& path, int num_ranks, FillRank&& fill_rank) {
  std::unique_ptr<std::FILE, int (*)(std::FILE*)> file(std::fopen(path.c_str(), "wb"), &std::fclose);
  if (!file) throw std::runtime_error("cannot open " + path);
  std::FILE* f = file.get();
  auto check = [&](bool ok) {
    if (!ok) throw std::runtime_error("write failed: " + path);
  };
  std::string head;
  put_le<std::uint64_t>(head, kMagicCookie);
  put_le<std::uint32_t>(head, static_cast<std::uint32_t>(num_ranks));
  put_le<std::uint8_t>(head, 0);
  put_le<std::uint8_t>(head, 0);
  check(std::fwrite(head.data(), 1, head.size(), f) == head.size());
  const std::string table(static_cast<std::size_t>(num_ranks) * 16, '\0');
  check(std::fwrite(table.data(), 1, table.size(), f) == table.size());

  // Jump-table offsets count from the end of the 8-byte cookie.
  std::uint64_t offset = 6 + 16ull * static_cast<std::uint64_t>(num_ranks);
  std::uint64_t total_nodes = 0;
  std::string offsets;
  for (int rank = 0; rank < num_ranks; ++rank) {
    RankGraph g;
    fill_rank(rank, g);
    total_nodes += g.nodes.size();
    const std::string blob = rank_blob(g);
    check(std::fwrite(blob.data(), 1, blob.size(), f) == blob.size());
    put_le<std::uint64_t>(offsets, offset);
    put_le<std::uint64_t>(offsets, offset + blob.size());
    offset += blob.size();
  }
  check(std::fseek(f, static_cast<long>(head.size()), SEEK_SET) == 0);
  check(std::fwrite(offsets.data(), 1, offsets.size(), f) == offsets.size());
  check(std::fclose(file.release()) == 0);
  return total_nodes;
}

}  // namespace goal_bin
}  // namespace cta
