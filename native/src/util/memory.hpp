#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace cta {

// Bytes of memory this process can still use: on Linux, the container limit
// (cgroup v2 or v1) minus what the container uses, or MemAvailable when there
// is no limit; on macOS, free + inactive + speculative pages (or physical RAM
// if Mach stats fail). Returns 0 if unavailable.
inline std::uint64_t available_memory_bytes() {
#if defined(__linux__)
  auto read_number = [](const char* path) -> std::uint64_t {
    std::ifstream in(path);
    std::string text;
    if (!(in >> text) || text == "max") return 0;
    try {
      return std::stoull(text);
    } catch (...) {
      return 0;
    }
  };
  std::uint64_t host = 0;
  {
    std::ifstream in("/proc/meminfo");
    std::string key;
    std::uint64_t kb = 0;
    std::string unit;
    while (in >> key >> kb >> unit)
      if (key == "MemAvailable:") {
        host = kb * 1024;
        break;
      }
  }
  std::uint64_t limit = read_number("/sys/fs/cgroup/memory.max");
  std::uint64_t used = read_number("/sys/fs/cgroup/memory.current");
  if (limit == 0) {
    limit = read_number("/sys/fs/cgroup/memory/memory.limit_in_bytes");
    used = read_number("/sys/fs/cgroup/memory/memory.usage_in_bytes");
    if (limit >= (std::uint64_t{1} << 60)) limit = 0;  // v1 reports "no limit" as a huge number
  }
  const std::uint64_t cgroup = limit > used ? limit - used : 0;
  if (cgroup != 0 && host != 0) return cgroup < host ? cgroup : host;
  return cgroup != 0 ? cgroup : host;
#elif defined(__APPLE__)
  vm_size_t page_size = 0;
  if (host_page_size(mach_host_self(), &page_size) == KERN_SUCCESS) {
    vm_statistics64_data_t stats{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count) ==
        KERN_SUCCESS) {
      const std::uint64_t reclaimable = static_cast<std::uint64_t>(stats.free_count) +
                                        static_cast<std::uint64_t>(stats.inactive_count) +
                                        static_cast<std::uint64_t>(stats.speculative_count);
      if (reclaimable > 0) return reclaimable * static_cast<std::uint64_t>(page_size);
    }
  }
  std::uint64_t total = 0;
  std::size_t len = sizeof(total);
  if (::sysctlbyname("hw.memsize", &total, &len, nullptr, 0) == 0 && total > 0) {
    return total / 2;
  }
  return 0;
#else
  return 0;
#endif
}

// Effective memory budget for streaming engines: explicit request if > 0, else
// 60% of available memory (or 4 GiB fallback if OS memory cannot be queried).
inline std::uint64_t resolve_memory_budget(std::uint64_t requested_bytes) {
  if (requested_bytes > 0) return requested_bytes;
  const std::uint64_t avail = available_memory_bytes();
  if (avail == 0) return std::uint64_t{4} << 30;
  return (avail / 10) * 6;
}

}  // namespace cta
