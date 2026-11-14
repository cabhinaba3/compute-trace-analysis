#pragma once

#include <cstdint>
#include <string>

namespace cta {

// `.goal` text schedule: per step a calc, then a ring (or direct) AllReduce.
void write_goal_allreduce(const std::string& path, int world_size, std::int64_t compute_ns,
                          std::int64_t allreduce_bytes, int steps,
                          const std::string& algorithm);

// The same schedule written straight to the SPCL LogGOPSim `.bin` layout, without
// going through the text form. Returns the node count.
std::uint64_t write_goal_allreduce_bin(const std::string& path, int world_size,
                                       std::int64_t compute_ns, std::int64_t allreduce_bytes,
                                       int steps, const std::string& algorithm);

}  // namespace cta
