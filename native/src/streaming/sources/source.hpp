// Unified streaming source interface and per-format factory functions.
#pragma once

#include <sys/resource.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "streaming/batch.hpp"

namespace cta::streaming {

// Returns a safe upper bound on simultaneously open file descriptors so multi-
// rank streaming over thousands of .et/.json files never hits EMFILE.
inline std::size_t safe_fd_limit(std::size_t requested = 0) {
  std::size_t cap = 256;
  struct rlimit rl {};
  if (::getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) {
    const std::size_t soft = static_cast<std::size_t>(rl.rlim_cur);
    cap = soft > 64 ? std::min<std::size_t>(soft / 2, 1024) : std::max<std::size_t>(8, soft / 2);
  }
  if (requested > 0) cap = std::min(cap, requested);
  return std::max<std::size_t>(4, cap);
}

// Returns whether a path refers to a regular seekable file (not stdin/FIFO/pipe).
inline bool is_seekable_path(const std::string& path) {
  if (path.empty() || path == "-") return false;
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) return false;
  return S_ISREG(st.st_mode);
}

inline std::uint64_t regular_file_size(const std::string& path) {
  if (path.empty() || path == "-") return 0;
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return 0;
  return static_cast<std::uint64_t>(st.st_size);
}

// Abstract streaming source yielding bounded GraphChunks.
class GraphSource {
 public:
  virtual ~GraphSource() = default;

  // Fills `out` with up to ~`max_nodes` nodes (and their edges). Returns true
  // if a non-empty chunk was produced, false at EOF.
  virtual bool next_chunk(GraphChunk& out, std::size_t max_nodes) = 0;

  virtual bool is_seekable() const = 0;
  virtual void reset() = 0;
  virtual std::string format_name() const = 0;
  virtual std::uint64_t estimated_bytes() const = 0;
};

// Decoded view of one packed 39-byte SPCL LogGOPSim GOAL binary node record.
struct GoalBinNode {
  std::uint32_t deps_cnt = 0;
  std::int8_t op_type = 0;  // 1=SEND, 2=RECV, 3=CALC
  std::uint32_t peer = 0;
  std::uint64_t size = 0;
  std::uint32_t tag = 0;
  std::uint8_t proc = 0;
  std::uint8_t nic = 0;
  std::uint32_t num_deps = 0;
  std::uint32_t deps_apdx = 0;
  std::uint32_t num_startdeps = 0;
  std::uint32_t startdeps_apdx = 0;
};

// Random-access zero-copy view over one rank in an mmapped GOAL .bin file.
struct GoalBinRankView {
  std::uint32_t rank = 0;
  std::uint32_t num_nodes = 0;
  std::uint32_t num_roots = 0;
  const std::uint8_t* roots_bytes = nullptr;     // num_roots * 4 little-endian u32
  const std::uint8_t* records_bytes = nullptr;   // num_nodes * 39 packed bytes
  const std::uint8_t* appendix_bytes = nullptr;  // appendix_len * 4 little-endian u32
  std::size_t appendix_len = 0;

  GoalBinNode node(std::uint32_t idx) const;
  std::uint32_t root(std::uint32_t idx) const;
  std::uint32_t appendix(std::size_t idx) const;
};

// Memory-mapped GOAL .bin container exposing O(1) random access to every rank.
class GoalBinFile {
 public:
  explicit GoalBinFile(const std::string& path);
  ~GoalBinFile();
  GoalBinFile(const GoalBinFile&) = delete;
  GoalBinFile& operator=(const GoalBinFile&) = delete;

  std::uint32_t num_ranks() const { return static_cast<std::uint32_t>(ranks_.size()); }
  std::uint64_t file_size() const { return size_; }
  const GoalBinRankView& rank_view(std::uint32_t rank) const { return ranks_[rank]; }
  std::uint64_t total_nodes() const;

 private:
  const std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
  std::vector<GoalBinRankView> ranks_;
};

// Factory functions for the four supported execution-graph formats.
std::unique_ptr<GraphSource> open_goal_bin_stream(const std::string& path);

std::unique_ptr<GraphSource> open_goal_text_stream(const std::string& path,
                                                   GoalSemantics semantics = GoalSemantics::kCtaLegacy);

std::unique_ptr<GraphSource> open_chakra_et_stream(const std::vector<std::string>& paths,
                                                   const std::vector<std::uint32_t>& ranks);

std::unique_ptr<GraphSource> open_chakra_host_stream(const std::vector<std::string>& paths,
                                                     const std::vector<std::uint32_t>& ranks);

}  // namespace cta::streaming
