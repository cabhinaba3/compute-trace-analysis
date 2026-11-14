#pragma once

#include <algorithm>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

namespace cta {

// CPUs this process may run on. Containers often report the host's cores, so
// the affinity mask is read rather than std::thread::hardware_concurrency().
inline int usable_cpus() {
#if defined(__linux__)
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof set, &set) == 0) return std::max(1, CPU_COUNT(&set));
#endif
  return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

}  // namespace cta
