// Streaming parallel reader for JAX/XProf Chrome-trace JSON (.trace.json.gz,
// perfetto_trace.json.gz, .json, .json.gz).
#pragma once

#include <cstdint>
#include <string>

#include "readers/jax/reader.hpp"

namespace cta::readers::jax {

JaxColumns read_jax_json(const std::string& path, int threads, std::uint64_t chunk_bytes,
                         std::uint64_t block_bytes);

}  // namespace cta::readers::jax
