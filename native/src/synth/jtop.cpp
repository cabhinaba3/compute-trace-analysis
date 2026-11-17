#include "synth/jtop.hpp"

#include <optional>
#include <string>

#include "util/json_value.hpp"

namespace cta::synth {
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

}  // namespace cta::synth
