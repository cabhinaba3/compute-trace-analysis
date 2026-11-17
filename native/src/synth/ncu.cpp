#include "synth/ncu.hpp"

#include <string>
#include <vector>

#include "util/format.hpp"
#include "util/sink.hpp"

namespace cta::synth {
namespace {

std::string num(double v) {
  std::string s;
  append_double(s, v);
  return s;
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

}  // namespace cta::synth
