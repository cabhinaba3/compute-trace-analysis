// Native protobuf reader for OpenXLA/XProf XPlane traces (.xplane.pb and
// .xplane.pb.gz).
#pragma once

#include <string>

#include "readers/jax/reader.hpp"

namespace cta::readers::jax {

JaxColumns read_jax_xplane(const std::string& path, int threads);

}  // namespace cta::readers::jax
