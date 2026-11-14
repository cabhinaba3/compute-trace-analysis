#include "small.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "common.hpp"
#include "json_value.hpp"

namespace cta {
namespace {

std::string num(double v) {
  std::string s;
  append_double(s, v);
  return s;
}

Json layer_profile(const char* name, double time_ms, double avg, double median, double pct) {
  return Json(JsonObject{{"name", name}, {"timeMs", time_ms}, {"averageMs", avg},
                         {"medianMs", median}, {"percentage", pct}});
}

Json layer_info(const char* name, const char* type, const std::string& precision, const char* tactic) {
  return Json(JsonObject{{"Name", name}, {"LayerType", type}, {"Precision", precision},
                         {"TacticName", tactic}});
}

}  // namespace

void write_ncu_csv(const std::string& path, double gemm_dur_ns, double sm_pct, double dram_pct,
                   double tensor_pct, double flops, double dram_bytes) {
  const std::string gemm_row = "\"0\",\"ampere_fp16_s16816gemm\",\"" + num(gemm_dur_ns) + "\",\"" +
                               num(sm_pct) + "\",\"" + num(dram_pct) + "\",\"" + num(tensor_pct) +
                               "\",\"62.5\",\"" + num(flops) + "\",\"" + num(dram_bytes) + "\",\"" +
                               num(dram_pct * 0.5) + "\",\"" + num(sm_pct * 0.4) + "\"";
  const std::vector<std::string> lines = {
      "==PROF== Connected to process 12345",
      "\"ID\",\"Kernel Name\",\"gpu__time_duration.sum\","
      "\"sm__throughput.avg.pct_of_peak_sustained_elapsed\","
      "\"gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed\","
      "\"sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active\","
      "\"sm__warps_active.avg.pct_of_peak_sustained_active\","
      "\"FLOPs\",\"dram__bytes.sum\","
      "\"smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct\","
      "\"smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct\"",
      "\"\",\"\",\"nsecond\",\"%\",\"%\",\"%\",\"%\",\"\",\"byte\",\"%\",\"%\"",
      gemm_row,
      "\"1\",\"vectorized_elementwise_relu\",\"100000.0\",\"20.0\",\"88.0\",\"0.0\","
      "\"55.0\",\"1e8\",\"2e8\",\"48.0\",\"5.0\""};
  Sink sink(path);
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (i) sink.put('\n');
    sink.put(lines[i]);
  }
  sink.close();
}

void write_jtop_trace(const std::string& path, int duration_s, int num_cores, double power_mw,
                      double gpu_util, double gpu_mem_mb, std::optional<double> temp_c) {
  const std::int64_t start_ns = static_cast<std::int64_t>(1700000000000000000LL);
  const std::int64_t ns_per_s = 1000000000;
  JsonArray samples;
  for (int second = 0; second <= duration_s; ++second) {
    JsonObject s{{"time", start_ns + static_cast<std::int64_t>(second) * ns_per_s},
                 {"GPU", gpu_util},
                 {"gpu_memory_usage", gpu_mem_mb},
                 {"RAM", 4096.0},
                 {"SWAP", 128.0},
                 {"Power TOT", power_mw},
                 {"VDD_GPU_SOC", power_mw * 0.65}};
    for (int core = 1; core <= num_cores; ++core) s.emplace_back("CPU" + std::to_string(core), 35.0);
    if (temp_c) s.emplace_back("Temp gpu", *temp_c);
    samples.emplace_back(std::move(s));
  }
  write_json_file(path, Json(std::move(samples)), 2);
}

void write_trtexec_timing(const std::string& path, int queries, double h2d_ms, double compute_ms,
                          double d2h_ms, double overlap_ms, double idle_gap_ms) {
  JsonArray timings;
  double query_start = 10.0;
  for (int q = 0; q < queries; ++q) {
    const double start_h2d = query_start;
    const double end_h2d = start_h2d + h2d_ms;
    const double start_compute = std::max(start_h2d, end_h2d - overlap_ms);
    const double end_compute = start_compute + compute_ms;
    const double start_d2h = std::max(start_compute, end_compute - overlap_ms * 0.5);
    const double end_d2h = start_d2h + d2h_ms;
    timings.emplace_back(JsonObject{{"startH2dMs", start_h2d},   {"endH2dMs", end_h2d},
                                    {"startComputeMs", start_compute}, {"endComputeMs", end_compute},
                                    {"startD2hMs", start_d2h},   {"endD2hMs", end_d2h},
                                    {"h2dMs", h2d_ms},           {"computeMs", compute_ms},
                                    {"d2hMs", d2h_ms},           {"latencyMs", end_d2h - start_h2d}});
    query_start = end_d2h + idle_gap_ms;
  }
  write_json_file(path, Json(std::move(timings)), 2);
}

void write_trtexec_layers(const std::string& profile_path, const std::string& engine_path,
                          bool fused, const std::string& precision) {
  JsonArray profile{Json(JsonObject{{"count", 100}})};
  JsonArray layers;
  if (fused) {
    profile.push_back(layer_profile("Conv_0 + Bias_1 + Relu_2", 80.0, 0.8, 0.79, 66.7));
    profile.push_back(layer_profile("Linear_3", 40.0, 0.4, 0.4, 33.3));
    layers.push_back(layer_info("Conv_0 + Bias_1 + Relu_2", "CaskConvolution", precision, "sm87_xmma_fprop"));
    layers.push_back(layer_info("Linear_3", "CaskGemm", precision, "sm87_gemm"));
  } else {
    profile.push_back(layer_profile("Conv_0", 110.0, 1.1, 1.1, 55.0));
    profile.push_back(layer_profile("Bias_1", 20.0, 0.2, 0.2, 10.0));
    profile.push_back(layer_profile("Relu_2", 15.0, 0.15, 0.15, 7.5));
    profile.push_back(layer_profile("Linear_3", 55.0, 0.55, 0.55, 27.5));
    layers.push_back(layer_info("Conv_0", "Convolution", precision, "cudnn_conv"));
    layers.push_back(layer_info("Bias_1", "ElementWise", precision, "ew_add"));
    layers.push_back(layer_info("Relu_2", "Activation", precision, "relu"));
    layers.push_back(layer_info("Linear_3", "MatrixMultiply", precision, "cublas_gemm"));
  }
  write_json_file(profile_path, Json(std::move(profile)), 2);
  write_json_file(engine_path, Json(JsonObject{{"Layers", std::move(layers)}}), 2);
}

}  // namespace cta
