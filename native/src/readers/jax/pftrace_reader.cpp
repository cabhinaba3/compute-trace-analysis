#include "readers/jax/pftrace_reader.hpp"

#include <google/protobuf/io/coded_stream.h>

#include <chrono>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "perfetto_trace.pb.h"
#include "readers/jax/common.hpp"
#include "util/util.hpp"

namespace cta::readers::jax {
namespace {

constexpr std::uint32_t kSeqIncrementalStateCleared = 1u;

struct SliceArgs {
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
};

void merge_args(SliceArgs& dst, const SliceArgs& src) {
  if (!dst.has_step_num && src.has_step_num) {
    dst.has_step_num = true;
    dst.step_num = src.step_num;
  }
  if (!dst.has_hlo_op && src.has_hlo_op) {
    dst.has_hlo_op = true;
    dst.hlo_op = src.hlo_op;
  }
  if (dst.correlation_id < 0 && src.correlation_id >= 0) dst.correlation_id = src.correlation_id;
  if (dst.device_ordinal == 0 && src.device_ordinal != 0) dst.device_ordinal = src.device_ordinal;
  if (dst.arg_name.empty()) dst.arg_name = src.arg_name;
  if (dst.tf_op.empty()) dst.tf_op = src.tf_op;
  if (dst.hlo_module.empty()) dst.hlo_module = src.hlo_module;
  if (dst.module.empty()) dst.module = src.module;
  if (dst.memcpy_details.empty()) dst.memcpy_details = src.memcpy_details;
  if (dst.memset_details.empty()) dst.memset_details = src.memset_details;
}

struct SequenceState {
  std::unordered_map<uint64_t, std::string> event_names;
  std::unordered_map<uint64_t, std::string> annotation_names;
  std::unordered_map<uint64_t, std::string> annotation_strings;
  uint64_t default_track_uuid = 0;
  int64_t last_timestamp_ns = 0;

  void clear_incremental() {
    event_names.clear();
    annotation_names.clear();
    annotation_strings.clear();
    default_track_uuid = 0;
    last_timestamp_ns = 0;
  }
};

struct TrackInfo {
  uint64_t parent_uuid = 0;
  bool has_parent = false;
  std::string name;
  bool has_process = false;
  std::string process_name;
  int32_t pid = 0;
  bool has_pid = false;
  bool has_thread = false;
  std::string thread_name;
  int32_t tid = 0;
  bool has_tid = false;
};

struct OpenSlice {
  std::string name;
  int64_t start_ns = 0;
  SliceArgs args;
};

struct BufferedEvent {
  bool is_chrome = false;
  uint64_t track_uuid = 0;
  int32_t chrome_pid = 0;
  int32_t chrome_tid = 0;
  std::string name;
  int64_t start_ns = 0;
  int64_t end_ns = 0;
  SliceArgs args;
};

struct ChromeMeta {
  bool is_thread = false;
  int32_t pid = 0;
  int32_t tid = 0;
  std::string name;
};

void apply_kv_string(std::string_view key, std::string_view val, SliceArgs& out) {
  if (key == "name") {
    out.arg_name.assign(val);
  } else if (key == "tf_op") {
    out.tf_op.assign(val);
  } else if (key == "hlo_op") {
    out.has_hlo_op = true;
    out.hlo_op.assign(val);
  } else if (key == "hlo_module") {
    out.hlo_module.assign(val);
  } else if (key == "module") {
    out.module.assign(val);
  } else if (key == "memcpy_details") {
    out.memcpy_details.assign(val);
  } else if (key == "Memset_details") {
    out.memset_details.assign(val);
  } else if (key == "correlation_id") {
    int64_t cid = -1;
    if (parse_nonneg_digits(val, cid)) out.correlation_id = cid;
  } else if (key == "step_num") {
    int64_t sn = 0;
    if (parse_signed_digits(val, sn)) {
      out.has_step_num = true;
      out.step_num = sn;
    }
  } else if (key == "device_ordinal") {
    int64_t ord = 0;
    if (parse_nonneg_digits(val, ord)) out.device_ordinal = ord;
  }
}

void apply_kv_int(std::string_view key, int64_t val, SliceArgs& out) {
  if (key == "correlation_id") {
    if (val >= 0) out.correlation_id = val;
  } else if (key == "step_num") {
    out.has_step_num = true;
    out.step_num = val;
  } else if (key == "device_ordinal") {
    if (val >= 0) out.device_ordinal = val;
  } else if (key == "hlo_op") {
    out.has_hlo_op = true;
    out.hlo_op = (val == 0) ? std::string() : std::to_string(val);
  } else if (val != 0) {
    apply_kv_string(key, std::to_string(val), out);
  }
}

void apply_kv_uint(std::string_view key, uint64_t val, SliceArgs& out) {
  if (val <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    apply_kv_int(key, static_cast<int64_t>(val), out);
  }
}

void read_debug_annotation(const cta::perfetto::DebugAnnotation& ann, const SequenceState& seq, SliceArgs& out,
                           int depth = 0) {
  if (depth > 8) return;
  if (ann.dict_entries_size() > 0) {
    for (const auto& entry : ann.dict_entries()) {
      read_debug_annotation(entry, seq, out, depth + 1);
    }
    return;
  }
  std::string_view key;
  if (ann.has_name()) {
    key = ann.name();
  } else if (ann.has_name_iid()) {
    const auto it = seq.annotation_names.find(ann.name_iid());
    if (it != seq.annotation_names.end()) key = it->second;
  }
  if (key.empty() || key.size() < 4 || key.size() > 14) return;

  switch (ann.value_case()) {
    case cta::perfetto::DebugAnnotation::kStringValue:
      apply_kv_string(key, ann.string_value(), out);
      return;
    case cta::perfetto::DebugAnnotation::kStringValueIid: {
      const auto it = seq.annotation_strings.find(ann.string_value_iid());
      if (it != seq.annotation_strings.end()) apply_kv_string(key, it->second, out);
      return;
    }
    case cta::perfetto::DebugAnnotation::kIntValue:
      apply_kv_int(key, ann.int_value(), out);
      return;
    case cta::perfetto::DebugAnnotation::kUintValue:
      apply_kv_uint(key, ann.uint_value(), out);
      return;
    case cta::perfetto::DebugAnnotation::kLegacyJsonValue:
      apply_kv_string(key, ann.legacy_json_value(), out);
      return;
    case cta::perfetto::DebugAnnotation::kDoubleValue: {
      const double d = ann.double_value();
      if (d != 0.0) {
        char tmp[40];
        std::snprintf(tmp, sizeof tmp, "%.17g", d);
        apply_kv_string(key, tmp, out);
      }
      return;
    }
    case cta::perfetto::DebugAnnotation::kBoolValue:
      if (ann.bool_value()) apply_kv_string(key, "True", out);
      return;
    default:
      if (key == "hlo_op") out.has_hlo_op = true;
      return;
  }
}

std::string_view chrome_string(const cta::perfetto::ChromeEventBundle& bundle, std::uint32_t idx) {
  // ChromeEvent string_table indices are 1-based (0 = unset).
  if (idx == 0 || idx > static_cast<std::uint32_t>(bundle.string_table_size())) return {};
  return bundle.string_table(static_cast<int>(idx - 1));
}

void read_chrome_arg(const cta::perfetto::ChromeEvent::Arg& arg, const cta::perfetto::ChromeEventBundle& bundle,
                     SliceArgs& out) {
  std::string_view key;
  if (arg.has_name()) {
    key = arg.name();
  } else if (arg.has_name_index()) {
    key = chrome_string(bundle, arg.name_index());
  }
  if (key.empty() || key.size() < 4 || key.size() > 14) return;
  switch (arg.value_case()) {
    case cta::perfetto::ChromeEvent::Arg::kStringValue:
      apply_kv_string(key, arg.string_value(), out);
      return;
    case cta::perfetto::ChromeEvent::Arg::kJsonValue:
      apply_kv_string(key, arg.json_value(), out);
      return;
    case cta::perfetto::ChromeEvent::Arg::kIntValue:
      apply_kv_int(key, arg.int_value(), out);
      return;
    case cta::perfetto::ChromeEvent::Arg::kUintValue:
      apply_kv_uint(key, arg.uint_value(), out);
      return;
    case cta::perfetto::ChromeEvent::Arg::kDoubleValue: {
      const double d = arg.double_value();
      if (d != 0.0) {
        char tmp[40];
        std::snprintf(tmp, sizeof tmp, "%.17g", d);
        apply_kv_string(key, tmp, out);
      }
      return;
    }
    case cta::perfetto::ChromeEvent::Arg::kBoolValue:
      if (arg.bool_value()) apply_kv_string(key, "True", out);
      return;
    default:
      if (key == "hlo_op") out.has_hlo_op = true;
      return;
  }
}

struct ResolvedTrack {
  bool is_device = false;
  int64_t device_ordinal = 0;
  uint64_t proc_key = 0;
  std::string line_name;
  int64_t stream = 0;
};

}  // namespace

JaxColumns read_jax_pftrace(const std::string& path) {
  using Clock = std::chrono::steady_clock;
  ProtoFrameReader reader(open_source(path));

  std::unordered_map<uint64_t, TrackInfo> descriptors;
  std::unordered_map<std::uint32_t, SequenceState> sequences;
  std::unordered_map<uint64_t, std::vector<OpenSlice>> open_slices;
  std::vector<BufferedEvent> events;
  std::vector<ChromeMeta> chrome_metas;

  double parse_s = 0.0;
  cta::perfetto::TracePacket packet;
  std::string payload;

  while (true) {
    if (!reader.next_message(1u, payload)) break;
    const auto t0 = Clock::now();
    packet.Clear();
    {
      google::protobuf::io::CodedInputStream cis(reinterpret_cast<const std::uint8_t*>(payload.data()),
                                                 static_cast<int>(payload.size()));
      cis.SetTotalBytesLimit(std::numeric_limits<int>::max());
      if (!packet.ParseFromCodedStream(&cis) || !cis.ConsumedEntireMessage()) {
        throw std::invalid_argument("invalid Perfetto protobuf");
      }
    }

    const std::uint32_t seq_id =
        packet.has_trusted_packet_sequence_id() ? packet.trusted_packet_sequence_id() : 0u;
    SequenceState& seq = sequences[seq_id];
    if (packet.has_sequence_flags() && (packet.sequence_flags() & kSeqIncrementalStateCleared) != 0u) {
      seq.clear_incremental();
    }
    if (packet.has_trace_packet_defaults() && packet.trace_packet_defaults().has_track_event_defaults() &&
        packet.trace_packet_defaults().track_event_defaults().has_track_uuid()) {
      seq.default_track_uuid = packet.trace_packet_defaults().track_event_defaults().track_uuid();
    }
    if (packet.has_interned_data()) {
      const auto& idata = packet.interned_data();
      for (const auto& en : idata.event_names()) {
        if (en.has_iid() && en.has_name()) seq.event_names[en.iid()] = en.name();
      }
      for (const auto& dn : idata.debug_annotation_names()) {
        if (dn.has_iid() && dn.has_name()) seq.annotation_names[dn.iid()] = dn.name();
      }
      for (const auto& ds : idata.debug_annotation_string_values()) {
        if (ds.has_iid() && ds.has_str()) seq.annotation_strings[ds.iid()] = ds.str();
      }
    }

    switch (packet.data_case()) {
      case cta::perfetto::TracePacket::kTrackDescriptor: {
        const auto& td = packet.track_descriptor();
        if (td.has_uuid()) {
          TrackInfo& info = descriptors[td.uuid()];
          if (td.has_parent_uuid()) {
            info.parent_uuid = td.parent_uuid();
            info.has_parent = true;
          }
          if (td.has_name()) {
            info.name = td.name();
          } else if (td.has_static_name()) {
            info.name = td.static_name();
          }
          if (td.has_process()) {
            info.has_process = true;
            if (td.process().has_pid()) {
              info.pid = td.process().pid();
              info.has_pid = true;
            }
            if (td.process().has_process_name()) info.process_name = td.process().process_name();
          }
          if (td.has_thread()) {
            info.has_thread = true;
            if (td.thread().has_pid()) {
              info.pid = td.thread().pid();
              info.has_pid = true;
            }
            if (td.thread().has_tid()) {
              info.tid = td.thread().tid();
              info.has_tid = true;
            }
            if (td.thread().has_thread_name()) info.thread_name = td.thread().thread_name();
          }
        }
        break;
      }

      case cta::perfetto::TracePacket::kTrackEvent: {
        const auto& te = packet.track_event();
        int64_t ts_ns = seq.last_timestamp_ns;
        if (packet.has_timestamp()) {
          ts_ns = static_cast<int64_t>(packet.timestamp());
        } else if (te.has_timestamp_absolute_us()) {
          ts_ns = te.timestamp_absolute_us() * 1000;
        } else if (te.has_timestamp_delta_us()) {
          ts_ns = seq.last_timestamp_ns + te.timestamp_delta_us() * 1000;
        }
        seq.last_timestamp_ns = ts_ns;

        const uint64_t track_uuid = te.has_track_uuid() ? te.track_uuid() : seq.default_track_uuid;
        std::string ename;
        if (te.has_name()) {
          ename = te.name();
        } else if (te.has_name_iid()) {
          if (const auto it = seq.event_names.find(te.name_iid()); it != seq.event_names.end()) {
            ename = it->second;
          }
        } else if (te.has_legacy_event() && te.legacy_event().has_name_iid()) {
          if (const auto it = seq.event_names.find(te.legacy_event().name_iid()); it != seq.event_names.end()) {
            ename = it->second;
          }
        }

        SliceArgs args;
        for (const auto& ann : te.debug_annotations()) read_debug_annotation(ann, seq, args);

        const auto type = te.has_type() ? te.type() : cta::perfetto::TrackEvent::TYPE_UNSPECIFIED;
        if (type == cta::perfetto::TrackEvent::TYPE_SLICE_BEGIN) {
          open_slices[track_uuid].push_back({std::move(ename), ts_ns, std::move(args)});
        } else if (type == cta::perfetto::TrackEvent::TYPE_SLICE_END) {
          auto it = open_slices.find(track_uuid);
          if (it != open_slices.end() && !it->second.empty()) {
            OpenSlice open = std::move(it->second.back());
            it->second.pop_back();
            merge_args(open.args, args);
            if (ts_ns > open.start_ns) {
              events.push_back(
                  {false, track_uuid, 0, 0, std::move(open.name), open.start_ns, ts_ns, std::move(open.args)});
            }
          }
        } else if (type == cta::perfetto::TrackEvent::TYPE_UNSPECIFIED && te.has_legacy_event()) {
          const auto& le = te.legacy_event();
          if (le.has_phase() && le.phase() == 'X' && le.has_duration_us() && le.duration_us() > 0) {
            const int64_t end_ns = ts_ns + le.duration_us() * 1000;
            events.push_back({false, track_uuid, 0, 0, std::move(ename), ts_ns, end_ns, std::move(args)});
          }
        }
        break;
      }

      case cta::perfetto::TracePacket::kChromeEvents: {
        const auto& bundle = packet.chrome_events();
        for (const auto& ce : bundle.chrome_events()) {
          const int32_t phase = ce.has_phase() ? ce.phase() : 0;
          std::string_view ename;
          if (ce.has_name()) {
            ename = ce.name();
          } else if (ce.has_name_index()) {
            ename = chrome_string(bundle, ce.name_index());
          }
          SliceArgs args;
          for (const auto& a : ce.args()) read_chrome_arg(a, bundle, args);
          const int32_t pid = ce.has_process_id() ? ce.process_id() : 0;
          const int32_t tid = ce.has_thread_id() ? ce.thread_id() : 0;
          if (phase == 'M') {
            if (ename == "process_name") {
              chrome_metas.push_back({false, pid, 0, std::move(args.arg_name)});
            } else if (ename == "thread_name") {
              chrome_metas.push_back({true, pid, tid, std::move(args.arg_name)});
            }
          } else if (phase == 'X') {
            const int64_t dur_ns = ce.has_duration() ? ce.duration() * 1000 : 0;
            if (dur_ns > 0) {
              const int64_t start_ns = ce.has_timestamp() ? ce.timestamp() * 1000 : 0;
              events.push_back(
                  {true, 0, pid, tid, std::string(ename), start_ns, start_ns + dur_ns, std::move(args)});
            }
          }
        }
        break;
      }

      default:
        break;
    }
    parse_s += duration_seconds(Clock::now() - t0);
  }

  const auto t_finish = Clock::now();

  // Resolve TrackDescriptor hierarchy: find each track's owning process,
  // device ordinal, line name, stream id, and per-device line prefix.
  std::unordered_map<uint64_t, ResolvedTrack> resolved_tracks;
  std::unordered_map<uint64_t, std::vector<std::string_view>> device_proc_lines;

  for (const auto& [uuid, info] : descriptors) {
    ResolvedTrack rt;
    rt.line_name = !info.thread_name.empty() ? info.thread_name : info.name;
    rt.stream =
        info.has_tid ? euclidean_mod_stream(info.tid) : euclidean_mod_stream(static_cast<int64_t>(uuid));

    uint64_t cur = uuid;
    uint64_t proc_uuid = uuid;
    for (int hop = 0; hop < 32; ++hop) {
      const auto it = descriptors.find(cur);
      if (it == descriptors.end()) break;
      const TrackInfo& node = it->second;
      int64_t ord = 0;
      const std::string_view candidate = !node.process_name.empty() ? std::string_view(node.process_name)
                                                                    : std::string_view(node.name);
      if (parse_device_ordinal(candidate, ord)) {
        rt.is_device = true;
        rt.device_ordinal = ord;
        proc_uuid = cur;
        break;
      }
      if (node.has_process) {
        proc_uuid = cur;
        break;
      }
      if (!node.has_parent || node.parent_uuid == cur) {
        proc_uuid = cur;
        break;
      }
      cur = node.parent_uuid;
    }
    rt.proc_key = proc_uuid;
    resolved_tracks.emplace(uuid, std::move(rt));
  }

  for (const auto& [uuid, rt] : resolved_tracks) {
    if (rt.is_device && uuid != rt.proc_key && !rt.line_name.empty()) {
      device_proc_lines[rt.proc_key].emplace_back(rt.line_name);
    }
  }
  std::unordered_map<uint64_t, std::string_view> track_prefix_by_proc;
  for (const auto& [proc_uuid, lnames] : device_proc_lines) {
    track_prefix_by_proc[proc_uuid] = select_device_line_prefix(lnames);
  }

  // Resolve legacy ChromeEventBundle process_name / thread_name metadata.
  std::unordered_map<int32_t, std::string> chrome_proc_names;
  std::unordered_map<uint64_t, std::string> chrome_thread_names;
  std::unordered_map<int32_t, std::vector<uint64_t>> chrome_pid_threads;
  auto chrome_pt_key = [](int32_t pid, int32_t tid) -> uint64_t {
    return (static_cast<uint64_t>(static_cast<std::uint32_t>(pid)) << 32) |
           static_cast<uint64_t>(static_cast<std::uint32_t>(tid));
  };
  for (const ChromeMeta& m : chrome_metas) {
    if (!m.is_thread) {
      chrome_proc_names[m.pid] = m.name;
    } else {
      const uint64_t key = chrome_pt_key(m.pid, m.tid);
      auto [it, inserted] = chrome_thread_names.emplace(key, m.name);
      if (!inserted) {
        it->second = m.name;
      } else {
        chrome_pid_threads[m.pid].push_back(key);
      }
    }
  }
  std::unordered_map<int32_t, int64_t> chrome_devices;
  std::unordered_map<int32_t, std::string_view> chrome_prefixes;
  for (const auto& [pid, pname] : chrome_proc_names) {
    int64_t ord = 0;
    if (parse_device_ordinal(pname, ord)) {
      chrome_devices[pid] = ord;
      std::vector<std::string_view> lnames;
      if (const auto tit = chrome_pid_threads.find(pid); tit != chrome_pid_threads.end()) {
        for (uint64_t k : tit->second) lnames.emplace_back(chrome_thread_names[k]);
      }
      chrome_prefixes[pid] = select_device_line_prefix(lnames);
    }
  }

  // Emit all buffered events in trace order through JaxEventBuilder.
  JaxEventBuilder builder("pftrace");
  std::unordered_map<uint64_t, int64_t> host_thread_ids;

  for (const BufferedEvent& ev : events) {
    bool is_on_device = false;
    int64_t device_ord = ev.args.device_ordinal;
    int64_t stream = 0;
    uint64_t host_thread_key = 0;
    bool line_allowed = true;

    if (!ev.is_chrome) {
      host_thread_key = ev.track_uuid;
      stream = euclidean_mod_stream(static_cast<int64_t>(ev.track_uuid));
      if (const auto it = resolved_tracks.find(ev.track_uuid); it != resolved_tracks.end()) {
        const ResolvedTrack& rt = it->second;
        is_on_device = rt.is_device;
        stream = rt.stream;
        if (is_on_device) {
          device_ord = rt.device_ordinal;
          if (const auto pit = track_prefix_by_proc.find(rt.proc_key);
              pit != track_prefix_by_proc.end() && !pit->second.empty()) {
            line_allowed = std::string_view(rt.line_name).starts_with(pit->second);
          }
        }
      }
    } else {
      host_thread_key = chrome_pt_key(ev.chrome_pid, ev.chrome_tid) ^ 0x9e3779b97f4a7c15ULL;
      stream = euclidean_mod_stream(ev.chrome_tid);
      if (const auto dit = chrome_devices.find(ev.chrome_pid); dit != chrome_devices.end()) {
        is_on_device = true;
        device_ord = dit->second;
        const std::string_view prefix = chrome_prefixes[ev.chrome_pid];
        if (!prefix.empty()) {
          const auto tnit = chrome_thread_names.find(chrome_pt_key(ev.chrome_pid, ev.chrome_tid));
          const std::string_view tname =
              tnit != chrome_thread_names.end() ? std::string_view(tnit->second) : "";
          line_allowed = tname.starts_with(prefix);
        }
      }
    }

    if (ev.args.has_step_num && !is_on_device) {
      builder.add_step(ev.args.step_num, ev.start_ns, ev.end_ns);
      continue;
    }

    if (!is_on_device && std::string_view(ev.name).starts_with(kExecutePrefix) && !ev.args.arg_name.empty()) {
      builder.add_execution(ev.args.arg_name, ev.start_ns, ev.end_ns);
    }

    if (!is_on_device && !ev.args.has_hlo_op) {
      if (ev.args.correlation_id >= 0) {
        auto [it, inserted] = host_thread_ids.emplace(host_thread_key, static_cast<int64_t>(host_thread_ids.size()));
        builder.add_api_call(ev.name, ev.start_ns, ev.end_ns, it->second, ev.args.correlation_id);
      }
      continue;
    }

    if (is_on_device && !line_allowed) continue;

    const std::string_view mod =
        ev.args.hlo_module.empty() ? std::string_view(ev.args.module) : std::string_view(ev.args.hlo_module);
    const std::string_view details = ev.args.memcpy_details.empty() ? std::string_view(ev.args.memset_details)
                                                                    : std::string_view(ev.args.memcpy_details);
    builder.add_op(ev.name, ev.args.arg_name, ev.args.tf_op, ev.args.hlo_op, mod, details, ev.start_ns,
                   ev.end_ns, stream, device_ord, ev.args.correlation_id);
  }

  JaxColumns columns = builder.finish();
  columns.timing.finish = duration_seconds(Clock::now() - t_finish);
  columns.timing.read = reader.read_seconds();
  columns.timing.split = parse_s;
  return columns;
}

}  // namespace cta::readers::jax
