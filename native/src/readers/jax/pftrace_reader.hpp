// Native protobuf reader for binary Perfetto traces (.pftrace, .pftrace.gz,
// .perfetto-trace), supporting TrackEvent + TrackDescriptor + InternedData as
// well as legacy ChromeEventBundle packets.
#pragma once

#include <string>

#include "readers/jax/reader.hpp"

namespace cta::readers::jax {

JaxColumns read_jax_pftrace(const std::string& path);

}  // namespace cta::readers::jax
