#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cta {

// One `trace.rank<N>.et` per rank: a GlobalMetadata message, then per step a
// compute node and an all-reduce node, each length-delimited. Returns the paths.
std::vector<std::string> write_chakra_et_run(const std::string& out_dir, int world_size,
                                             std::int64_t compute_us, std::int64_t allreduce_bytes,
                                             int steps, int straggler_rank,
                                             double straggler_factor);

// Host execution trace in its JSON form.
void write_chakra_host_et(const std::string& path, int steps, double compute_us,
                          std::int64_t comm_bytes);

}  // namespace cta
