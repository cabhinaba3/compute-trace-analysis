// Engine B: 1-pass multi-rank frontier and windowed DAG solver.
//
// Two specialized paths:
//   1. solve_goal_bin_frontier(path, opts): exact O(frontier) multi-rank Kahn
//      over an mmapped GOAL .bin using its jump table and pre-compiled CSR,
//      with per-rank file-order P2P channel numbering, raw-edge dedup,
//      unmatched-recv release, and cycle witness extraction.
//   2. FrontierEngine::solve_dag(source, opts) & solve_chakra_et_frontier(...):
//      1-pass bounded-window DAG solver with Certified Eviction-Epoch Intervals
//      ([makespan_lb, makespan_ub] + Bloom-guarded sparse 64-bit ID fallback),
//      bounded lookahead for dangling/forward deps, optional 2-node virtual
//      collective barriers, and strict window_local vs global_exact slack
//      cut certification.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "streaming/engines/engine.hpp"

namespace cta::streaming::engines {

std::unique_ptr<GraphEngine> make_frontier_engine();

// Exact O(frontier) solver directly over an mmapped GOAL .bin file.
DagStreamResult solve_goal_bin_frontier(const std::string& bin_path, const DagSolveOptions& opts = {});

// Multi-rank lockstep 1-pass frontier solver over Chakra .et files (one per
// rank), reading ranks round-robin with FD multiplexing to keep P2P and
// collective barriers tightly synchronized in bounded memory.
DagStreamResult solve_chakra_et_frontier(const std::vector<std::string>& paths,
                                         const std::vector<std::uint32_t>& ranks,
                                         const DagSolveOptions& opts = {});

}  // namespace cta::streaming::engines
