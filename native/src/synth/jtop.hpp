// Synthetic Jetson jtop telemetry traces.
#pragma once

#include <optional>
#include <string>

namespace cta::synth {

void write_jtop_trace(const std::string& path, int duration_s, int num_cores, double power_mw,
                      double gpu_util, double gpu_mem_mb, std::optional<double> temp_c);

}  // namespace cta::synth
