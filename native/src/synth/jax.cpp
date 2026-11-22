#include "synth/jax.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "perfetto_trace.pb.h"
#include "util/format.hpp"
#include "util/rng.hpp"
#include "util/sink.hpp"
#include "xplane.pb.h"

namespace cta::synth {
namespace {

using std::int64_t;
using std::uint64_t;

struct SynthArg {
  std::string key;
  std::string str_val;
  int64_t int_val = 0;
  bool is_int = false;
};

struct SynthEvent {
  int pid = 0;
  int tid = 0;
  std::string name;
  int64_t start_us = 0;
  int64_t dur_us = 0;
  std::vector<SynthArg> args;
};

struct SynthWorkload {
  int devices = 2;
  std::vector<SynthEvent> events;
};

int64_t round_us(double us) { return std::max<int64_t>(1, static_cast<int64_t>(std::llround(us))); }

// Builds one canonical workload shared identically by Chrome-trace JSON,
// XPlane protobuf, and Perfetto protobuf writers.
SynthWorkload build_workload(int steps, int devices, double gemm_us, double comm_us, bool annotated) {
  if (steps < 0) throw std::invalid_argument("steps must be >= 0");
  if (devices < 1) throw std::invalid_argument("devices must be >= 1");
  if (gemm_us <= 0.0 || comm_us <= 0.0) throw std::invalid_argument("gemm_us and comm_us must be > 0");

  SynthWorkload w;
  w.devices = devices;
  const int64_t gemm = round_us(gemm_us);
  const int64_t comm = round_us(comm_us);
  const int64_t step_period = std::max<int64_t>(1000, gemm + comm + 300);

  auto str_arg = [](const char* k, std::string v) { return SynthArg{k, std::move(v), 0, false}; };
  auto int_arg = [](const char* k, int64_t v) { return SynthArg{k, std::to_string(v), v, true}; };

  int64_t corr = 1;

  // Out-of-step warmup / autotune work before step 0.
  {
    const int64_t c = corr++;
    w.events.push_back({1, 2, "cuLaunchKernel", 5, 3, {int_arg("correlation_id", c)}});
    w.events.push_back({10,
                        7,
                        "autotune.0",
                        10,
                        12,
                        {str_arg("hlo_op", "autotune.0"), str_arg("hlo_module", "jit_warmup"),
                         int_arg("correlation_id", c)}});
  }

  for (int step = 0; step < steps; ++step) {
    const int64_t t0 = 1000 + static_cast<int64_t>(step) * step_period;
    const int64_t step_dur = gemm + comm + 200;

    if (annotated) {
      w.events.push_back({1, 1, "train", t0, step_dur, {int_arg("step_num", step)}});
    }
    w.events.push_back({1,
                        1,
                        "CommonPjRtLoadedExecutable::Execute (jit_train_step)",
                        t0 + 5,
                        25,
                        {str_arg("name", "jit_train_step"), int_arg("run_id", step)}});

    for (int d = 0; d < devices; ++d) {
      const int dev_pid = 10 + d;
      const int64_t d_off = static_cast<int64_t>(d) * 2;

      // 1. Compute fusion kernel.
      {
        const int64_t c = corr++;
        w.events.push_back({1, 2, "cuLaunchKernel", t0 + 10 + d_off, 4, {int_arg("correlation_id", c)}});
        w.events.push_back({dev_pid,
                            7,
                            "gemm_fusion_dot.1",
                            t0 + 20,
                            gemm,
                            {str_arg("hlo_op", "dot.1"), str_arg("hlo_module", "jit_train_step"),
                             str_arg("name", "jit(train_step)/jvp()/dot_general"),
                             int_arg("correlation_id", c)}});
        // Derived XLA Ops duplicate line (must be ignored because Stream #7 is present).
        w.events.push_back({dev_pid,
                            8,
                            "dot.1",
                            t0 + 20,
                            gemm,
                            {str_arg("hlo_op", "dot.1"), str_arg("hlo_module", "jit_train_step"),
                             str_arg("tf_op", "jit(train_step)/jvp()/dot_general"),
                             int_arg("correlation_id", c)}});
      }

      // 2. NCCL AllReduce collective.
      {
        const int64_t c = corr++;
        w.events.push_back({1, 2, "cuLaunchKernel", t0 + 16 + d_off, 4, {int_arg("correlation_id", c)}});
        w.events.push_back({dev_pid,
                            7,
                            "ncclDevKernel_AllReduce_bf16_RING(ncclDevKernelArgsStorage<4096ul>)",
                            t0 + 25 + gemm,
                            comm,
                            {str_arg("hlo_op", "all-reduce.2"), str_arg("hlo_module", "jit_train_step"),
                             str_arg("name", "jit(train_step)/psum"), int_arg("correlation_id", c)}});
      }

      // 3. Peer-to-peer copy.
      {
        const int64_t c = corr++;
        w.events.push_back({1, 2, "MemcpyP2P", t0 + 22 + d_off, 3, {int_arg("correlation_id", c)}});
        w.events.push_back({dev_pid,
                            7,
                            "MemcpyP2P",
                            t0 + 30 + gemm + comm,
                            15,
                            {str_arg("hlo_module", "jit_train_step"),
                             str_arg("memcpy_details",
                                     "kind_src:device kind_dst:device size:1048576 dest:1 async:1"),
                             int_arg("correlation_id", c)}});
      }

      // 4. Device memset.
      {
        const int64_t c = corr++;
        w.events.push_back({1, 2, "Memset 3", t0 + 28 + d_off, 2, {int_arg("correlation_id", c)}});
        w.events.push_back({dev_pid,
                            7,
                            "Memset 3",
                            t0 + 50 + gemm + comm,
                            4,
                            {str_arg("hlo_module", "jit_train_step"),
                             str_arg("Memset_details", "kind:device num_bytes:65536 async:1"),
                             int_arg("correlation_id", c)}});
      }
    }
  }
  return w;
}

void append_meta_json(std::string& out, int pid, int tid, bool is_thread, std::string_view name) {
  out.append("{\"ph\": \"M\", \"pid\": ");
  append_int(out, pid);
  if (is_thread) {
    out.append(", \"tid\": ");
    append_int(out, tid);
    out.append(", \"name\": \"thread_name\"");
  } else {
    out.append(", \"name\": \"process_name\"");
  }
  out.append(", \"args\": {\"name\": ");
  append_json_string(out, name);
  out.append("}}");
}

void append_event_json(std::string& out, const SynthEvent& e) {
  out.append("{\"ph\": \"X\", \"pid\": ");
  append_int(out, e.pid);
  out.append(", \"tid\": ");
  append_int(out, e.tid);
  out.append(", \"name\": ");
  append_json_string(out, e.name);
  out.append(", \"ts\": ");
  append_int(out, e.start_us);
  out.append(", \"dur\": ");
  append_int(out, e.dur_us);
  if (!e.args.empty()) {
    out.append(", \"args\": {");
    for (std::size_t i = 0; i < e.args.size(); ++i) {
      if (i) out.append(", ");
      append_json_string(out, e.args[i].key);
      out.append(": ");
      append_json_string(out, e.args[i].str_val);
    }
    out.push_back('}');
  }
  out.push_back('}');
}

}  // namespace

void write_jax_trace(const std::string& path, int steps, int devices, double gemm_us, double comm_us,
                     bool annotated) {
  const SynthWorkload w = build_workload(steps, devices, gemm_us, comm_us, annotated);
  Sink sink(path);
  sink.put("{\"displayTimeUnit\": \"ns\", \"traceEvents\": [");
  bool first = true;
  std::string buf;
  auto emit = [&] {
    if (!first) sink.put(", ");
    first = false;
    sink.put(buf);
    buf.clear();
  };

  append_meta_json(buf, 1, 0, false, "/host:CPU");
  emit();
  append_meta_json(buf, 1, 1, true, "python");
  emit();
  append_meta_json(buf, 1, 2, true, "pjrt_async_work_runner");
  emit();
  for (int d = 0; d < w.devices; ++d) {
    append_meta_json(buf, 10 + d, 0, false, "/device:GPU:" + std::to_string(d));
    emit();
    append_meta_json(buf, 10 + d, 7, true, "Stream #7(Compute,MemcpyP2P,Memset)");
    emit();
    append_meta_json(buf, 10 + d, 8, true, "XLA Ops");
    emit();
  }
  for (const SynthEvent& e : w.events) {
    append_event_json(buf, e);
    emit();
  }
  sink.put("]}");
  sink.close();
}

void write_jax_xplane(const std::string& path, int steps, int devices, double gemm_us, double comm_us,
                      bool annotated) {
  const SynthWorkload w = build_workload(steps, devices, gemm_us, comm_us, annotated);
  cta::xplane::XSpace space;
  space.add_hostnames("synth-jax-host");

  // Helper to manage per-plane metadata tables & lines.
  struct PlaneBuilder {
    cta::xplane::XPlane* plane = nullptr;
    std::unordered_map<std::string, int64_t> event_ids;
    std::unordered_map<std::string, int64_t> stat_ids;
    std::unordered_map<int, cta::xplane::XLine*> lines;

    int64_t event_id(const std::string& name) {
      auto [it, ins] = event_ids.emplace(name, static_cast<int64_t>(event_ids.size()) + 1);
      if (ins) {
        auto& meta = (*plane->mutable_event_metadata())[it->second];
        meta.set_id(it->second);
        meta.set_name(name);
      }
      return it->second;
    }

    int64_t stat_id(const std::string& name) {
      auto [it, ins] = stat_ids.emplace(name, static_cast<int64_t>(stat_ids.size()) + 1);
      if (ins) {
        auto& meta = (*plane->mutable_stat_metadata())[it->second];
        meta.set_id(it->second);
        meta.set_name(name);
      }
      return it->second;
    }

    cta::xplane::XLine* line(int tid, const std::string& name) {
      auto it = lines.find(tid);
      if (it != lines.end()) return it->second;
      auto* l = plane->add_lines();
      l->set_id(tid);
      l->set_name(name);
      l->set_display_name(name);
      l->set_timestamp_ns(0);
      lines.emplace(tid, l);
      return l;
    }
  };

  std::unordered_map<int, PlaneBuilder> builders;
  {
    auto* hp = space.add_planes();
    hp->set_id(1);
    hp->set_name("/host:CPU");
    auto& hb = builders[1];
    hb.plane = hp;
    hb.line(1, "python");
    hb.line(2, "pjrt_async_work_runner");
  }
  for (int d = 0; d < w.devices; ++d) {
    auto* dp = space.add_planes();
    dp->set_id(d);
    dp->set_name("/device:GPU:" + std::to_string(d));
    auto& db = builders[10 + d];
    db.plane = dp;
    db.line(7, "Stream #7(Compute,MemcpyP2P,Memset)");
    db.line(8, "XLA Ops");
  }

  for (const SynthEvent& e : w.events) {
    PlaneBuilder& pb = builders.at(e.pid);
    cta::xplane::XLine* ln = pb.lines.at(e.tid);
    cta::xplane::XEvent* xev = ln->add_events();
    xev->set_metadata_id(pb.event_id(e.name));
    xev->set_offset_ps(e.start_us * 1000000LL);
    xev->set_duration_ps(e.dur_us * 1000000LL);
    for (const SynthArg& a : e.args) {
      cta::xplane::XStat* st = xev->add_stats();
      st->set_metadata_id(pb.stat_id(a.key));
      if (a.is_int) {
        st->set_int64_value(a.int_val);
      } else {
        // Mirror real XProf: intern string stats via ref_value into stat_metadata.
        st->set_ref_value(static_cast<uint64_t>(pb.stat_id(a.str_val)));
      }
    }
  }

  std::string serialized;
  if (!space.SerializeToString(&serialized)) throw std::runtime_error("failed to serialize XSpace protobuf");
  Sink sink(path);
  sink.put(serialized);
  sink.close();
}

void write_jax_pftrace(const std::string& path, int steps, int devices, double gemm_us, double comm_us,
                       bool annotated, bool legacy_chrome) {
  const SynthWorkload w = build_workload(steps, devices, gemm_us, comm_us, annotated);
  cta::perfetto::Trace trace;

  if (legacy_chrome) {
    auto* pkt = trace.add_packet();
    auto* bundle = pkt->mutable_chrome_events();
    auto add_meta = [&](int pid, int tid, bool is_thread, const std::string& val) {
      auto* ce = bundle->add_chrome_events();
      ce->set_phase('M');
      ce->set_process_id(pid);
      if (is_thread) ce->set_thread_id(tid);
      ce->set_name(is_thread ? "thread_name" : "process_name");
      auto* a = ce->add_args();
      a->set_name("name");
      a->set_string_value(val);
    };
    add_meta(1, 0, false, "/host:CPU");
    add_meta(1, 1, true, "python");
    add_meta(1, 2, true, "pjrt_async_work_runner");
    for (int d = 0; d < w.devices; ++d) {
      add_meta(10 + d, 0, false, "/device:GPU:" + std::to_string(d));
      add_meta(10 + d, 7, true, "Stream #7(Compute,MemcpyP2P,Memset)");
      add_meta(10 + d, 8, true, "XLA Ops");
    }
    for (const SynthEvent& e : w.events) {
      auto* ce = bundle->add_chrome_events();
      ce->set_phase('X');
      ce->set_process_id(e.pid);
      ce->set_thread_id(e.tid);
      ce->set_name(e.name);
      ce->set_timestamp(e.start_us);
      ce->set_duration(e.dur_us);
      for (const SynthArg& a : e.args) {
        auto* ca = ce->add_args();
        ca->set_name(a.key);
        if (a.is_int) {
          ca->set_int_value(a.int_val);
        } else {
          ca->set_string_value(a.str_val);
        }
      }
    }
  } else {
    constexpr std::uint32_t kSeqId = 1u;
    auto proc_uuid = [](int pid) -> uint64_t { return static_cast<uint64_t>(pid) * 1000ull; };
    auto track_uuid = [&](int pid, int tid) -> uint64_t {
      return proc_uuid(pid) + static_cast<uint64_t>(tid);
    };

    auto emit_proc = [&](int pid, const std::string& name) {
      auto* pkt = trace.add_packet();
      pkt->set_trusted_packet_sequence_id(kSeqId);
      auto* td = pkt->mutable_track_descriptor();
      td->set_uuid(proc_uuid(pid));
      td->set_name(name);
      auto* pd = td->mutable_process();
      pd->set_pid(pid);
      pd->set_process_name(name);
    };
    auto emit_thread = [&](int pid, int tid, const std::string& name) {
      auto* pkt = trace.add_packet();
      pkt->set_trusted_packet_sequence_id(kSeqId);
      auto* td = pkt->mutable_track_descriptor();
      td->set_uuid(track_uuid(pid, tid));
      td->set_parent_uuid(proc_uuid(pid));
      td->set_name(name);
      auto* th = td->mutable_thread();
      th->set_pid(pid);
      th->set_tid(tid);
      th->set_thread_name(name);
    };

    emit_proc(1, "/host:CPU");
    emit_thread(1, 1, "python");
    emit_thread(1, 2, "pjrt_async_work_runner");
    for (int d = 0; d < w.devices; ++d) {
      emit_proc(10 + d, "/device:GPU:" + std::to_string(d));
      emit_thread(10 + d, 7, "Stream #7(Compute,MemcpyP2P,Memset)");
      emit_thread(10 + d, 8, "XLA Ops");
    }

    // Intern event names and annotation keys on the first event packet.
    std::unordered_map<std::string, uint64_t> ev_iids;
    std::unordered_map<std::string, uint64_t> ann_iids;
    std::unordered_map<std::string, uint64_t> str_iids;
    bool first_seq_pkt = true;

    for (const SynthEvent& e : w.events) {
      const uint64_t tuuid = track_uuid(e.pid, e.tid);
      auto* begin_pkt = trace.add_packet();
      begin_pkt->set_trusted_packet_sequence_id(kSeqId);
      if (first_seq_pkt) {
        begin_pkt->set_sequence_flags(1u);  // SEQ_INCREMENTAL_STATE_CLEARED
        first_seq_pkt = false;
      }
      begin_pkt->set_timestamp(static_cast<uint64_t>(e.start_us) * 1000ull);
      auto* idata = begin_pkt->mutable_interned_data();
      auto* te = begin_pkt->mutable_track_event();
      te->set_type(cta::perfetto::TrackEvent::TYPE_SLICE_BEGIN);
      te->set_track_uuid(tuuid);

      auto [eit, eins] = ev_iids.emplace(e.name, static_cast<uint64_t>(ev_iids.size()) + 1ull);
      if (eins) {
        auto* en = idata->add_event_names();
        en->set_iid(eit->second);
        en->set_name(e.name);
      }
      te->set_name_iid(eit->second);

      for (const SynthArg& a : e.args) {
        auto* da = te->add_debug_annotations();
        auto [kit, kins] = ann_iids.emplace(a.key, static_cast<uint64_t>(ann_iids.size()) + 1ull);
        if (kins) {
          auto* dn = idata->add_debug_annotation_names();
          dn->set_iid(kit->second);
          dn->set_name(a.key);
        }
        da->set_name_iid(kit->second);
        if (a.is_int) {
          da->set_int_value(a.int_val);
        } else {
          auto [sit, sins] = str_iids.emplace(a.str_val, static_cast<uint64_t>(str_iids.size()) + 1ull);
          if (sins) {
            auto* ds = idata->add_debug_annotation_string_values();
            ds->set_iid(sit->second);
            ds->set_str(a.str_val);
          }
          da->set_string_value_iid(sit->second);
        }
      }
      if (idata->event_names_size() == 0 && idata->debug_annotation_names_size() == 0 &&
          idata->debug_annotation_string_values_size() == 0) {
        begin_pkt->clear_interned_data();
      }

      auto* end_pkt = trace.add_packet();
      end_pkt->set_trusted_packet_sequence_id(kSeqId);
      end_pkt->set_timestamp(static_cast<uint64_t>(e.start_us + e.dur_us) * 1000ull);
      auto* ete = end_pkt->mutable_track_event();
      ete->set_type(cta::perfetto::TrackEvent::TYPE_SLICE_END);
      ete->set_track_uuid(tuuid);
    }
  }

  std::string serialized;
  if (!trace.SerializeToString(&serialized)) throw std::runtime_error("failed to serialize Perfetto protobuf");
  Sink sink(path);
  sink.put(serialized);
  sink.close();
}

namespace {

const std::array<std::string_view, 8> kJaxKernelNames = {
    "nvjet_tst_128x64_64x8_1x2_h_bz_NNT",
    "loop_multiply_fusion_1",
    "input_concatenate_fusion",
    "gemm_fusion_dot_7",
    "loop_convert_multiply_fusion",
    "wrapped_convert_1",
    "void jax_kernel<{lambda(int)#1}>(char const*, \"quoted\")",
    "void jax_fusion<\"a\\\\b\", {brace}>(float*)"};

const std::array<std::string_view, 6> kJaxScopes = {
    "jit(train_step)/jvp()/dot_general",
    "jit(train_step)/jvp()/convert_element_type",
    "jit(train_step)/transpose(jvp())/dot_general",
    "jit(train_step)/transpose(jvp())/transpose",
    "jit(train_step)/jvp()/layer_norm",
    "jit(train_step)/jvp()/softmax"};

const std::array<std::string_view, 4> kPyFrames = {
    "$jax/_src/pjit.py:1686 _cpp_pjit",
    "$jax/_src/core.py:1007 bind_with_trace",
    "$jax/_src/lax/lax.py:6345 dot_general",
    "$jax/_src/tree_util.py:199 tree_flatten"};

}  // namespace

std::uint64_t write_jax_trace_large(const std::string& path, const JaxLargeSpec& spec) {
  if (spec.devices < 1) throw std::invalid_argument("devices must be >= 1");
  if (spec.layers_per_step < 1) throw std::invalid_argument("layers_per_step must be >= 1");

  Rng rng(spec.seed * 1000003ull + 17ull);
  Sink sink(path);
  std::uint64_t events = 0;
  bool first = true;
  std::string ev;
  auto emit = [&] {
    if (!first) sink.put(", ");
    first = false;
    sink.put(ev);
    ev.clear();
    ++events;
  };

  sink.put("{\"displayTimeUnit\": \"ns\", \"traceEvents\": [");
  append_meta_json(ev, 1, 0, false, "/host:CPU");
  emit();
  append_meta_json(ev, 1, 1, true, "python");
  emit();
  append_meta_json(ev, 1, 2, true, "pjrt_async_work_runner");
  emit();
  for (int d = 0; d < spec.devices; ++d) {
    append_meta_json(ev, 10 + d, 0, false, "/device:GPU:" + std::to_string(d));
    emit();
    append_meta_json(ev, 10 + d, 24, true, "Stream #24(Compute,MemcpyD2D,MemcpyP2P,Memset)");
    emit();
    append_meta_json(ev, 10 + d, 25, true, "XLA Ops");
    emit();
  }

  double cpu_ts = 1000.0;
  std::vector<double> gpu_free(static_cast<std::size_t>(spec.devices), 1000.0);
  int64_t corr = 100;
  int step = 0;

  while (sink.bytes() < spec.target_bytes) {
    const double step_start = cpu_ts;
    double step_end = cpu_ts;

    ev = "{\"ph\": \"X\", \"pid\": 1, \"tid\": 1, \"name\": \"CommonPjRtLoadedExecutable::Execute "
         "(jit_train_step)\", \"ts\": ";
    append_double(ev, cpu_ts);
    ev.append(", \"dur\": 35.0, \"args\": {\"name\": \"jit_train_step\", \"run_id\": \"");
    append_int(ev, step);
    ev.append("\"}}");
    emit();

    for (int layer = 0; layer < spec.layers_per_step; ++layer) {
      for (int p = 0; p < spec.python_events_per_layer; ++p) {
        const auto frame = kPyFrames[static_cast<std::size_t>(rng.range(0, kPyFrames.size() - 1))];
        ev = "{\"ph\": \"X\", \"pid\": 1, \"tid\": 1, \"name\": ";
        append_json_string(ev, frame);
        ev.append(", \"ts\": ");
        append_double(ev, cpu_ts + static_cast<double>(p) * 1.5);
        ev.append(", \"dur\": 1.2}");
        emit();
      }

      for (int d = 0; d < spec.devices; ++d) {
        const int64_t c = corr++;
        const double launch_ts = cpu_ts + 4.0 + static_cast<double>(d) * 1.5;
        ev = "{\"ph\": \"X\", \"pid\": 1, \"tid\": 2, \"name\": \"cuLaunchKernel\", \"ts\": ";
        append_double(ev, launch_ts);
        ev.append(", \"dur\": 3.5, \"args\": {\"correlation_id\": \"");
        append_int(ev, c);
        ev.append("\", \"device_id\": \"");
        append_int(ev, d);
        ev.append("\"}}");
        emit();

        const double kstart = std::max(gpu_free[static_cast<std::size_t>(d)], launch_ts + 6.0);
        const double kdur = 12.0 + rng.unit() * 180.0;
        const auto kname =
            kJaxKernelNames[static_cast<std::size_t>(rng.range(0, kJaxKernelNames.size() - 1))];
        const auto scope = kJaxScopes[static_cast<std::size_t>(rng.range(0, kJaxScopes.size() - 1))];
        ev = "{\"ph\": \"X\", \"pid\": ";
        append_int(ev, 10 + d);
        ev.append(", \"tid\": 24, \"name\": ");
        append_json_string(ev, kname);
        ev.append(", \"ts\": ");
        append_double(ev, kstart);
        ev.append(", \"dur\": ");
        append_double(ev, kdur);
        ev.append(", \"args\": {\"correlation_id\": \"");
        append_int(ev, c);
        ev.append("\", \"hlo_op\": \"fusion.");
        append_int(ev, layer);
        ev.append("\", \"hlo_module\": \"jit_train_step\", \"name\": ");
        append_json_string(ev, scope);
        ev.append(", \"tf_op\": \"XlaModule:\"}}");
        emit();

        // Occasional derived XLA Ops duplicate event.
        if (layer % 4 == 0) {
          ev = "{\"ph\": \"X\", \"pid\": ";
          append_int(ev, 10 + d);
          ev.append(", \"tid\": 25, \"name\": \"fusion.");
          append_int(ev, layer);
          ev.append("\", \"ts\": ");
          append_double(ev, kstart);
          ev.append(", \"dur\": ");
          append_double(ev, kdur);
          ev.append(", \"args\": {\"hlo_op\": \"fusion.");
          append_int(ev, layer);
          ev.append("\", \"hlo_module\": \"jit_train_step\"}}");
          emit();
        }
        gpu_free[static_cast<std::size_t>(d)] = kstart + kdur;

        if (layer == spec.layers_per_step - 1) {
          const int64_t cc = corr++;
          ev = "{\"ph\": \"X\", \"pid\": 1, \"tid\": 2, \"name\": \"cuLaunchKernel\", \"ts\": ";
          append_double(ev, launch_ts + 4.0);
          ev.append(", \"dur\": 4.0, \"args\": {\"correlation_id\": \"");
          append_int(ev, cc);
          ev.append("\"}}");
          emit();

          const double cstart = gpu_free[static_cast<std::size_t>(d)] + 2.0;
          const double cdur = 45.0 + rng.unit() * 60.0;
          ev = "{\"ph\": \"X\", \"pid\": ";
          append_int(ev, 10 + d);
          ev.append(", \"tid\": 24, \"name\": "
                    "\"ncclDevKernel_AllReduce_bf16_RING(ncclDevKernelArgsStorage<4096ul>)\", \"ts\": ");
          append_double(ev, cstart);
          ev.append(", \"dur\": ");
          append_double(ev, cdur);
          ev.append(", \"args\": {\"correlation_id\": \"");
          append_int(ev, cc);
          ev.append("\", \"hlo_op\": \"all-reduce.1\", \"hlo_module\": \"jit_train_step\", \"name\": "
                    "\"jit(train_step)/psum\"}}");
          emit();
          gpu_free[static_cast<std::size_t>(d)] = cstart + cdur;
        }
        step_end = std::max(step_end, gpu_free[static_cast<std::size_t>(d)]);
      }
      cpu_ts += 20.0;
      step_end = std::max(step_end, cpu_ts);
    }

    if (spec.annotated) {
      ev = "{\"ph\": \"X\", \"pid\": 1, \"tid\": 1, \"name\": \"train\", \"ts\": ";
      append_double(ev, step_start);
      ev.append(", \"dur\": ");
      append_double(ev, step_end - step_start);
      ev.append(", \"args\": {\"step_num\": \"");
      append_int(ev, step);
      ev.append("\"}}");
      emit();
    }
    cpu_ts = step_end + 25.0;
    for (double& f : gpu_free) f = std::max(f, cpu_ts);
    ++step;
  }

  sink.put("]}");
  sink.close();
  return events;
}

}  // namespace cta::synth
