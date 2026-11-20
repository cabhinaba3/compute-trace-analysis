#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace cta {

// Bytes of memory this process can still use: the container limit (cgroup v2
// or v1) minus what the container uses, or MemAvailable when there is no limit.
// 0 if neither can be read.
inline std::uint64_t available_memory_bytes() {
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
}

}  // namespace cta
