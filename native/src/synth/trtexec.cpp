#include "synth/trtexec.hpp"

#include <algorithm>
#include <string>

#include "util/json_value.hpp"

namespace cta::synth {
namespace {

Json layer_profile(const char* name, double time_ms, double avg, double median, double pct) {
  return Json(JsonObject{{"name", name}, {"timeMs", time_ms}, {"averageMs", avg},
                         {"medianMs", median}, {"percentage", pct}});
}

Json layer_info(const char* name, const char* type, const std::string& precision, const char* tactic) {
  return Json(JsonObject{{"Name", name}, {"LayerType", type}, {"Precision", precision},
                         {"TacticName", tactic}});
}

}  // namespace

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

}  // namespace cta::synth
