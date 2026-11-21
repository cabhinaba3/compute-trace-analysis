// Registry for pluggable streaming and out-of-core GraphEngine implementations.
#include "streaming/engines/engine.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "streaming/engines/frontier/frontier_engine.hpp"

namespace cta::streaming::engines {

std::unique_ptr<GraphEngine> make_graph_engine(const std::string& name) {
  if (name == "frontier") return make_frontier_engine();
  throw std::invalid_argument("unknown graph engine '" + name + "'; available: frontier");
}

std::vector<std::string> available_graph_engines() { return {"frontier"}; }

}  // namespace cta::streaming::engines
