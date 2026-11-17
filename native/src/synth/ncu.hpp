// Synthetic Nsight Compute CSV exports.
#pragma once

#include <string>

namespace cta::synth {

void write_ncu_csv(const std::string& path, double gemm_dur_ns, double sm_pct, double dram_pct,
                   double tensor_pct, double flops, double dram_bytes);

}  // namespace cta::synth
