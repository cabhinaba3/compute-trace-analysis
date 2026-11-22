#include "readers/jax/xplane_reader.hpp"

#include <google/protobuf/io/coded_stream.h>

#include <chrono>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "readers/jax/common.hpp"
#include "util/ordered_pool.hpp"
#include "util/util.hpp"
#include "xplane.pb.h"

namespace cta::readers::jax {
namespace {

enum class StatKey : std::uint8_t {
  kNone = 0,
  kStepNum,
  kCorrelationId,
  kHloOp,
  kHloModule,
  kModule,
  kName,
  kTfOp,
  kMemcpyDetails,
  kMemsetDetails,
  kDeviceOrdinal,
};

StatKey classify_stat_name(std::string_view name) {
  if (name.size() < 4 || name.size() > 14) return StatKey::kNone;
  if (name == "correlation_id") return StatKey::kCorrelationId;
  if (name == "name") return StatKey::kName;
  if (name == "tf_op") return StatKey::kTfOp;
  if (name == "hlo_op") return StatKey::kHloOp;
  if (name == "hlo_module") return StatKey::kHloModule;
  if (name == "module") return StatKey::kModule;
  if (name == "step_num") return StatKey::kStepNum;
  if (name == "memcpy_details") return StatKey::kMemcpyDetails;
  if (name == "Memset_details") return StatKey::kMemsetDetails;
  if (name == "device_ordinal") return StatKey::kDeviceOrdinal;
  return StatKey::kNone;
}

struct EventStats {
  bool has_step_num = false;
  bool has_hlo_op = false;
  int64_t step_num = 0;
  int64_t correlation_id = -1;
  int64_t device_ordinal = 0;
  std::string arg_name;
  std::string tf_op;
  std::string hlo_op;
  std::string hlo_module;
  std::string module;
  std::string memcpy_details;
  std::string memset_details;

  void reset() {
    has_step_num = false;
    has_hlo_op = false;
    step_num = 0;
    correlation_id = -1;
    device_ordinal = 0;
    arg_name.clear();
    tf_op.clear();
    hlo_op.clear();
    hlo_module.clear();
    module.clear();
    memcpy_details.clear();
    memset_details.clear();
  }
};

using StatMetaMap = google::protobuf::Map<std::int64_t, cta::xplane::XStatMetadata>;

std::string stat_as_string(const cta::xplane::XStat& stat, const StatMetaMap& stat_meta) {
  switch (stat.value_case()) {
    case cta::xplane::XStat::kStrValue:
      return stat.str_value();
    case cta::xplane::XStat::kBytesValue:
      return stat.bytes_value();
    case cta::xplane::XStat::kRefValue: {
      const auto it = stat_meta.find(static_cast<int64_t>(stat.ref_value()));
      return it != stat_meta.end() ? it->second.name() : std::string();
    }
    case cta::xplane::XStat::kInt64Value: {
      const int64_t v = stat.int64_value();
      return v == 0 ? std::string() : std::to_string(v);
    }
    case cta::xplane::XStat::kUint64Value: {
      const uint64_t v = stat.uint64_value();
      return v == 0 ? std::string() : std::to_string(v);
    }
    case cta::xplane::XStat::kDoubleValue: {
      const double d = stat.double_value();
      if (d == 0.0) return {};
      char tmp[40];
      std::snprintf(tmp, sizeof tmp, "%.17g", d);
      return tmp;
    }
    case cta::xplane::XStat::VALUE_NOT_SET:
      return {};
  }
  return {};
}

bool stat_as_nonneg(const cta::xplane::XStat& stat, const StatMetaMap& stat_meta, int64_t& out) {
  switch (stat.value_case()) {
    case cta::xplane::XStat::kInt64Value:
      if (stat.int64_value() >= 0) {
        out = stat.int64_value();
        return true;
      }
      return false;
    case cta::xplane::XStat::kUint64Value:
      if (stat.uint64_value() <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        out = static_cast<int64_t>(stat.uint64_value());
        return true;
      }
      return false;
    case cta::xplane::XStat::kStrValue:
      return parse_nonneg_digits(stat.str_value(), out);
    case cta::xplane::XStat::kBytesValue:
      return parse_nonneg_digits(stat.bytes_value(), out);
    case cta::xplane::XStat::kRefValue: {
      const auto it = stat_meta.find(static_cast<int64_t>(stat.ref_value()));
      return it != stat_meta.end() && parse_nonneg_digits(it->second.name(), out);
    }
    default:
      return false;
  }
}

bool stat_as_signed(const cta::xplane::XStat& stat, const StatMetaMap& stat_meta, int64_t& out) {
  switch (stat.value_case()) {
    case cta::xplane::XStat::kInt64Value:
      out = stat.int64_value();
      return true;
    case cta::xplane::XStat::kUint64Value:
      if (stat.uint64_value() <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        out = static_cast<int64_t>(stat.uint64_value());
        return true;
      }
      return false;
    case cta::xplane::XStat::kStrValue:
      return parse_signed_digits(stat.str_value(), out);
    case cta::xplane::XStat::kBytesValue:
      return parse_signed_digits(stat.bytes_value(), out);
    case cta::xplane::XStat::kRefValue: {
      const auto it = stat_meta.find(static_cast<int64_t>(stat.ref_value()));
      return it != stat_meta.end() && parse_signed_digits(it->second.name(), out);
    }
    default:
      return false;
  }
}

void apply_stat(const cta::xplane::XStat& stat, const std::unordered_map<int64_t, StatKey>& key_map,
                const StatMetaMap& stat_meta, EventStats& out) {
  const auto it = key_map.find(stat.metadata_id());
  if (it == key_map.end()) return;
  switch (it->second) {
    case StatKey::kNone:
      return;
    case StatKey::kStepNum: {
      int64_t sn = 0;
      if (stat_as_signed(stat, stat_meta, sn)) {
        out.has_step_num = true;
        out.step_num = sn;
      }
      return;
    }
    case StatKey::kCorrelationId: {
      int64_t cid = -1;
      if (stat_as_nonneg(stat, stat_meta, cid)) out.correlation_id = cid;
      return;
    }
    case StatKey::kDeviceOrdinal: {
      int64_t ord = 0;
      if (stat_as_nonneg(stat, stat_meta, ord)) out.device_ordinal = ord;
      return;
    }
    case StatKey::kHloOp:
      out.has_hlo_op = true;
      out.hlo_op = stat_as_string(stat, stat_meta);
      return;
    case StatKey::kHloModule:
      out.hlo_module = stat_as_string(stat, stat_meta);
      return;
    case StatKey::kModule:
      out.module = stat_as_string(stat, stat_meta);
      return;
    case StatKey::kName:
      out.arg_name = stat_as_string(stat, stat_meta);
      return;
    case StatKey::kTfOp:
      out.tf_op = stat_as_string(stat, stat_meta);
      return;
    case StatKey::kMemcpyDetails:
      out.memcpy_details = stat_as_string(stat, stat_meta);
      return;
    case StatKey::kMemsetDetails:
      out.memset_details = stat_as_string(stat, stat_meta);
      return;
  }
}

struct PlaneOp {
  int32_t name, arg_name, tf_op, hlo_op, module, details;
  int64_t start_ns, end_ns, stream, device, correlation_id;
};

struct PlaneStep {
  int64_t step_num, start_ns, end_ns;
};

struct PlaneExec {
  int32_t module;
  int64_t start_ns, end_ns;
};

struct PlaneApiCall {
  int32_t name;
  int64_t start_ns, end_ns, line_id, correlation_id;
};

struct PlaneOut {
  Interner strings;
  std::vector<PlaneOp> ops;
  std::vector<PlaneStep> steps;
  std::vector<PlaneExec> execs;
  std::vector<PlaneApiCall> calls;
};

std::string_view pstr(const Interner& interner, int32_t code) {
  if (code < 0) return {};
  return interner.items()[static_cast<std::size_t>(code)];
}

int32_t opt_intern(Interner& interner, std::string_view s) {
  return s.empty() ? -1 : interner.intern(s);
}

std::unique_ptr<PlaneOut> parse_plane_bytes(const std::string& bytes) {
  cta::xplane::XPlane plane;
  {
    google::protobuf::io::CodedInputStream cis(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                                               static_cast<int>(bytes.size()));
    cis.SetTotalBytesLimit(std::numeric_limits<int>::max());
    if (!plane.ParseFromCodedStream(&cis) || !cis.ConsumedEntireMessage()) {
      throw std::invalid_argument("invalid XPlane protobuf");
    }
  }

  auto out = std::make_unique<PlaneOut>();
  if (plane.lines_size() == 0) return out;

  int64_t device_ordinal = 0;
  const bool is_device = parse_device_ordinal(plane.name(), device_ordinal);

  std::string_view line_prefix;
  if (is_device) {
    std::vector<std::string_view> line_names;
    line_names.reserve(static_cast<std::size_t>(plane.lines_size()));
    for (const auto& line : plane.lines()) {
      line_names.emplace_back(line.display_name().empty() ? line.name() : line.display_name());
    }
    line_prefix = select_device_line_prefix(line_names);
  }

  const auto& stat_meta = plane.stat_metadata();
  std::unordered_map<int64_t, StatKey> key_map;
  key_map.reserve(static_cast<std::size_t>(stat_meta.size()));
  for (const auto& [id, meta] : stat_meta) {
    const StatKey k = classify_stat_name(meta.name());
    if (k != StatKey::kNone) key_map.emplace(id, k);
  }

  const auto& event_meta = plane.event_metadata();
  EventStats st;

  for (const auto& line : plane.lines()) {
    if (is_device && !line_prefix.empty()) {
      const std::string_view lname = line.display_name().empty() ? line.name() : line.display_name();
      if (!lname.starts_with(line_prefix)) continue;
    }
    const int64_t line_ts_ns = line.timestamp_ns();
    const int64_t line_id = line.id();
    const int64_t stream = euclidean_mod_stream(line_id);

    for (const auto& ev : line.events()) {
      if (ev.data_case() == cta::xplane::XEvent::kNumOccurrences) continue;
      const int64_t duration_ns = ps_to_ns(ev.duration_ps());
      if (duration_ns <= 0) continue;
      const int64_t start_ns = line_ts_ns + ps_to_ns(ev.offset_ps());
      const int64_t end_ns = start_ns + duration_ns;

      const cta::xplane::XEventMetadata* meta = nullptr;
      if (const auto mit = event_meta.find(ev.metadata_id()); mit != event_meta.end()) {
        meta = &mit->second;
      }

      st.reset();
      if (meta && meta->stats_size() > 0) {
        for (const auto& s : meta->stats()) apply_stat(s, key_map, stat_meta, st);
      }
      for (const auto& s : ev.stats()) apply_stat(s, key_map, stat_meta, st);

      if (st.has_step_num && !is_device) {
        out->steps.push_back({st.step_num, start_ns, end_ns});
        continue;
      }

      const std::string_view ename =
          meta ? (meta->display_name().empty() ? std::string_view(meta->name())
                                               : std::string_view(meta->display_name()))
               : std::string_view();

      if (!is_device && ename.starts_with(kExecutePrefix) && !st.arg_name.empty()) {
        out->execs.push_back({out->strings.intern(st.arg_name), start_ns, end_ns});
      }

      if (!is_device && !st.has_hlo_op) {
        if (st.correlation_id >= 0) {
          out->calls.push_back({out->strings.intern(ename), start_ns, end_ns, line_id, st.correlation_id});
        }
        continue;
      }

      const std::string_view mod = st.hlo_module.empty() ? std::string_view(st.module)
                                                         : std::string_view(st.hlo_module);
      const std::string_view details = st.memcpy_details.empty() ? std::string_view(st.memset_details)
                                                                 : std::string_view(st.memcpy_details);
      const int64_t dev = is_device ? device_ordinal : st.device_ordinal;
      out->ops.push_back({
          out->strings.intern(ename),
          opt_intern(out->strings, st.arg_name),
          opt_intern(out->strings, st.tf_op),
          opt_intern(out->strings, st.hlo_op),
          opt_intern(out->strings, mod),
          opt_intern(out->strings, details),
          start_ns,
          end_ns,
          stream,
          dev,
          st.correlation_id,
      });
    }
  }
  return out;
}

}  // namespace

JaxColumns read_jax_xplane(const std::string& path, int threads) {
  using Clock = std::chrono::steady_clock;
  ProtoFrameReader reader(open_source(path));
  double split_s = 0.0, wait_s = 0.0;

  JaxEventBuilder builder("xplane");
  std::unordered_map<int64_t, int64_t> line_thread_map;
  int64_t next_thread_id = 0;

  auto merge_plane = [&](std::unique_ptr<PlaneOut>&& pout) {
    const PlaneOut& p = *pout;
    line_thread_map.clear();
    for (const PlaneStep& s : p.steps) builder.add_step(s.step_num, s.start_ns, s.end_ns);
    for (const PlaneExec& ex : p.execs) {
      builder.add_execution(pstr(p.strings, ex.module), ex.start_ns, ex.end_ns);
    }
    for (const PlaneApiCall& c : p.calls) {
      auto [it, inserted] = line_thread_map.emplace(c.line_id, next_thread_id);
      if (inserted) ++next_thread_id;
      builder.add_api_call(pstr(p.strings, c.name), c.start_ns, c.end_ns, it->second, c.correlation_id);
    }
    for (const PlaneOp& op : p.ops) {
      builder.add_op(pstr(p.strings, op.name), pstr(p.strings, op.arg_name), pstr(p.strings, op.tf_op),
                     pstr(p.strings, op.hlo_op), pstr(p.strings, op.module), pstr(p.strings, op.details),
                     op.start_ns, op.end_ns, op.stream, op.device, op.correlation_id);
    }
  };

  using Pool = OrderedPool<std::string, std::unique_ptr<PlaneOut>>;
  Pool pool(
      threads,
      []() -> Pool::Work {
        return [](std::string& payload) { return parse_plane_bytes(payload); };
      },
      merge_plane, static_cast<std::size_t>(threads) + 4);

  std::string payload;
  while (true) {
    const auto t0 = Clock::now();
    if (!reader.next_message(1u, payload)) break;
    const auto t1 = Clock::now();
    split_s += duration_seconds(t1 - t0);
    pool.submit(std::move(payload));
    wait_s += duration_seconds(Clock::now() - t1);
    payload.clear();
  }

  const auto t_finish = Clock::now();
  pool.finish();
  JaxColumns columns = builder.finish();
  columns.timing.finish = duration_seconds(Clock::now() - t_finish);
  columns.timing.read = reader.read_seconds();
  columns.timing.submit_wait = wait_s;
  columns.timing.split = std::max(0.0, split_s - reader.read_seconds());
  return columns;
}

}  // namespace cta::readers::jax
