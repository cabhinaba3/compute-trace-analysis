// Pluggable GraphEngine implementing X-Stream Edge-Centric Scatter-Gather
// (Amitabha Roy, Ivo Mihailovic, Willy Zwaenepoel — ACM SOSP 2013).
#pragma once

#include <memory>

#include "streaming/engines/engine.hpp"

namespace cta::streaming::engines {

std::unique_ptr<GraphEngine> make_xstream_engine();

}  // namespace cta::streaming::engines
