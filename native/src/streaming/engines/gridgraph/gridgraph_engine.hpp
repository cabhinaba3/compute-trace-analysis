// Engine C: GridGraph 2D out-of-core exact DAG solver (Zhu et al., USENIX ATC '15)
// operating over `.cta_graph` topological blocks with exact Pass 1 forward DP,
// Pass 2 reverse block slack (including strict GOAL irequires), CP op sequence
// backtracking, cycle witness reporting, and optional Scalasca delay attribution.
#pragma once

#include <memory>
#include <string>

#include "streaming/cache/graph_cache.hpp"
#include "streaming/engines/engine.hpp"

namespace cta::streaming::engines {

std::unique_ptr<GraphEngine> make_gridgraph_engine();

// Solves directly from an already-built `.cta_graph` cache file (zero rebuild!).
DagStreamResult solve_from_graph_cache(const cache::GraphCacheReader& reader, const DagSolveOptions& opts);

}  // namespace cta::streaming::engines
