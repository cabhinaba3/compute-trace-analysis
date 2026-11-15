#pragma once

#include <optional>
#include <string>

namespace cta {

void write_ncu_csv(const std::string& path, double gemm_dur_ns, double sm_pct, double dram_pct,
                   double tensor_pct, double flops, double dram_bytes);

void write_jtop_trace(const std::string& path, int duration_s, int num_cores, double power_mw,
                      double gpu_util, double gpu_mem_mb, std::optional<double> temp_c);

void write_trtexec_timing(const std::string& path, int queries, double h2d_ms, double compute_ms,
                          double d2h_ms, double overlap_ms, double idle_gap_ms);

void write_trtexec_layers(const std::string& profile_path, const std::string& engine_path,
                          bool fused, const std::string& precision);

}  // namespace cta
