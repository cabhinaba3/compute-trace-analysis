// Synthetic trtexec timing, layer-profile and layer-info exports.
#pragma once

#include <string>

namespace cta::synth {

void write_trtexec_timing(const std::string& path, int queries, double h2d_ms, double compute_ms,
                          double d2h_ms, double overlap_ms, double idle_gap_ms);

void write_trtexec_layers(const std::string& profile_path, const std::string& engine_path,
                          bool fused, const std::string& precision);

}  // namespace cta::synth
