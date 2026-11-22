#include "readers/jax/reader.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>

#include "readers/jax/common.hpp"
#include "readers/jax/json_reader.hpp"
#include "readers/jax/pftrace_reader.hpp"
#include "readers/jax/xplane_reader.hpp"
#include "util/cpus.hpp"

namespace cta::readers {
namespace {

bool ends_with_ci(std::string_view s, std::string_view suffix) {
  if (s.size() < suffix.size()) return false;
  const std::size_t off = s.size() - suffix.size();
  for (std::size_t i = 0; i < suffix.size(); ++i) {
    if (jax::ascii_lower(s[off + i]) != jax::ascii_lower(suffix[i])) return false;
  }
  return true;
}

std::string detect_format_by_content(const std::string& path) {
  auto source = jax::open_source(path);
  char buf[512];
  const std::size_t n = source->read(buf, sizeof buf);
  std::size_t i = 0;
  while (i < n && (buf[i] == ' ' || buf[i] == '\t' || buf[i] == '\r' || buf[i] == '\n')) ++i;
  if (i < n && (buf[i] == '{' || buf[i] == '[')) return "json";
  if (n >= 2 && static_cast<unsigned char>(buf[0]) == 0x0au) {
    // Both XSpace.planes (field 1) and Perfetto Trace.packet (field 1) start with tag 0x0a.
    // Peek the first submessage's first field tag to distinguish XPlane vs TracePacket.
    std::size_t p = 1;
    int shift = 0;
    while (p < n && shift < 64) {
      const unsigned char b = static_cast<unsigned char>(buf[p++]);
      if ((b & 0x80u) == 0) break;
      shift += 7;
    }
    if (p < n) {
      const unsigned char inner = static_cast<unsigned char>(buf[p]);
      // XPlane fields 1..6: id (0x08), name (0x12), lines (0x1a), event_meta (0x22), stat_meta (0x2a), stats (0x32).
      // Perfetto TracePacket fields: chrome_events (0x2a), timestamp (0x40), seq_id (0x50),
      // track_event (0x5a), interned_data (0x62), sequence_flags (0x68), defaults (0xda 0x03), track_descriptor (0xe2 0x03).
      if (inner == 0x08u || inner == 0x12u || inner == 0x1au || inner == 0x22u || inner == 0x32u) {
        return "xplane";
      }
      return "pftrace";
    }
  }
  if (n == 0) return "xplane";
  throw std::invalid_argument("cannot determine JAX trace format for " + path);
}

std::string resolve_format(const std::string& path, const std::string& requested) {
  if (requested == "json" || requested == "xplane" || requested == "pftrace") return requested;
  if (requested != "auto") {
    throw std::invalid_argument("unknown JAX format '" + requested + "'; use auto, json, xplane or pftrace");
  }
  if (ends_with_ci(path, ".xplane.pb") || ends_with_ci(path, ".xplane.pb.gz")) return "xplane";
  if (ends_with_ci(path, ".pftrace") || ends_with_ci(path, ".pftrace.gz") ||
      ends_with_ci(path, ".perfetto-trace") || ends_with_ci(path, ".perfetto-trace.gz")) {
    return "pftrace";
  }
  if (ends_with_ci(path, ".json") || ends_with_ci(path, ".json.gz")) return "json";
  return detect_format_by_content(path);
}

}  // namespace

JaxColumns read_jax(const std::string& path, const std::string& format, int threads, std::uint64_t chunk_bytes,
                    std::uint64_t block_bytes) {
  if (threads < 0) throw std::invalid_argument("threads must be >= 0");
  if (block_bytes < 1) throw std::invalid_argument("block_bytes must be at least 1");
  if (chunk_bytes < 64) throw std::invalid_argument("chunk_bytes must be at least 64");
  if (threads == 0) threads = usable_cpus();

  const std::string resolved = resolve_format(path, format);
  if (resolved == "json") return jax::read_jax_json(path, threads, chunk_bytes, block_bytes);
  if (resolved == "xplane") return jax::read_jax_xplane(path, threads);
  return jax::read_jax_pftrace(path);
}

}  // namespace cta::readers
