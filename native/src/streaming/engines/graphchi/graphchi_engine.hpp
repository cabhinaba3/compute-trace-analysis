// Pluggable GraphEngine implementing GraphChi Parallel Sliding Windows (PSW)
// (Aapo Kyrola, Guy Blelloch, Carlos Guestrin — USENIX OSDI 2012).
#pragma once

#include <memory>

#include "streaming/engines/engine.hpp"

namespace cta::streaming::engines {

std::unique_ptr<GraphEngine> make_graphchi_engine();

}  // namespace cta::streaming::engines
