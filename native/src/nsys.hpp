#pragma once

#include <string>

namespace cta {

void write_nsys_sqlite(const std::string& path, int steps, int gemm_us, int comm_us,
                       int comm_overlap_us, int warmup_penalty_us, int lock_wait_us,
                       double tensor_active_pct, bool legacy_gpu_metrics);

}  // namespace cta
