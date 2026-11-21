// Registry for pluggable streaming and out-of-core GraphEngine implementations.
#include "streaming/engines/engine.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "streaming/engines/frontier/frontier_engine.hpp"
#include "streaming/engines/graphchi/graphchi_engine.hpp"
#include "streaming/engines/gridgraph/gridgraph_engine.hpp"
#include "streaming/engines/xstream/xstream_engine.hpp"

namespace cta::streaming::engines {

std::unique_ptr<GraphEngine> make_graph_engine(const std::string& name) {
  if (name == "gridgraph") return make_gridgraph_engine();
  if (name == "frontier") return make_frontier_engine();
  if (name == "graphchi") return make_graphchi_engine();
  if (name == "xstream") return make_xstream_engine();
  throw std::invalid_argument("unknown graph engine '" + name + "'; available: gridgraph, frontier, graphchi, xstream");
}

std::vector<std::string> available_graph_engines() { return {"gridgraph", "frontier", "graphchi", "xstream"}; }

}  // namespace cta::streaming::engines
