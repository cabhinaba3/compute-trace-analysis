#include "kineto.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

#include "common.hpp"

namespace cta {
namespace {

// One complete-event ("ph": "X") object in json.dumps formatting.
void append_x_event(std::string& out, std::string_view cat, std::string_view name,
                    double ts, double dur, int pid, int tid,
                    std::string_view args_json = {}) {
  out.append("{\"ph\": \"X\", \"cat\": ");
  append_json_string(out, cat);
  out.append(", \"name\": ");
  append_json_string(out, name);
  out.append(", \"ts\": ");
  append_double(out, ts);
  out.append(", \"dur\": ");
  append_double(out, dur);
  out.append(", \"pid\": ");
  append_int(out, pid);
  out.append(", \"tid\": ");
  append_int(out, tid);
  if (!args_json.empty()) {
    out.append(", \"args\": ");
    out.append(args_json);
  }
  out.push_back('}');
}

std::string corr_args(std::int64_t corr, bool with_stream) {
  std::string a = "{\"correlation\": ";
  append_int(a, corr);
  if (with_stream) a.append(", \"stream\": 7");
  a.push_back('}');
  return a;
}

}  // namespace

void write_kineto_trace(const std::string& path, int steps, double attn_us,
                        double gemm_us, const std::string& kernel_variant) {
  const bool fp16 = kernel_variant == "fp16";
  const char* gemm_name = fp16 ? "ampere_fp16_s16816gemm" : "sm90_wgmma_fp8_gemm_kernel";
  const char* attn_name = fp16 ? "fmha_cutlassF_f16" : "flash_attn_fwd_fp8_kernel";

  Sink sink(path);
  sink.put("{\"schemaVersion\": 1, \"traceEvents\": [");
  bool first = true;
  std::string ev;
  auto emit = [&] {
    if (!first) sink.put(", ");
    first = false;
    sink.put(ev);
    ev.clear();
  };

  double step_ts = 1000.0;
  std::int64_t corr = 100;
  for (int step = 0; step < steps; ++step) {
    const double step_dur = attn_us + gemm_us + 50.0;
    append_x_event(ev, "user_annotation", "ProfilerStep#" + std::to_string(step),
                   step_ts, step_dur, 1, 1);
    emit();
    append_x_event(ev, "cpu_op", "aten::scaled_dot_product_attention", step_ts + 2.0, 15.0, 1, 1);
    emit();
    append_x_event(ev, "cuda_runtime", "cudaLaunchKernel", step_ts + 5.0, 5.0, 1, 1,
                   corr_args(corr, false));
    emit();
    append_x_event(ev, "kernel", attn_name, step_ts + 10.0, attn_us, 0, 7, corr_args(corr, true));
    emit();
    ++corr;

    const double gemm_ts = step_ts + 10.0 + attn_us;
    append_x_event(ev, "cpu_op", "aten::linear", step_ts + 20.0, 15.0, 1, 1);
    emit();
    append_x_event(ev, "cuda_runtime", "cudaLaunchKernel", step_ts + 25.0, 5.0, 1, 1,
                   corr_args(corr, false));
    emit();
    append_x_event(ev, "kernel", gemm_name, gemm_ts, gemm_us, 0, 7, corr_args(corr, true));
    emit();
    ++corr;

    step_ts += step_dur + 20.0;
  }
  sink.put("]}");
  sink.close();
}

namespace {

const std::array<std::string_view, 12> kAtenOps = {
    "aten::linear", "aten::addmm", "aten::layer_norm", "aten::gelu",
    "aten::softmax", "aten::scaled_dot_product_attention", "aten::add",
    "aten::mul", "aten::to", "aten::view", "aten::transpose", "aten::embedding"};

// Long demangled kernel names, some with braces, quotes and backslashes: the
// cases a chunk-boundary scanner must not trip over.
const std::array<std::string_view, 8> kKernelNames = {
    "void at::native::vectorized_elementwise_kernel<4, at::native::AbsFunctor<float>, at::detail::Array<char*, 2> >(int, at::native::AbsFunctor<float>, at::detail::Array<char*, 2>)",
    "void cutlass::Kernel<cutlass_80_tensorop_s16816gemm_f16_128x128_32x3_nn_align8>(cutlass_80_tensorop_s16816gemm_f16_128x128_32x3_nn_align8::Params)",
    "ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nn",
    "void at::native::(anonymous namespace)::softmax_warp_forward<float, float, float, 7, false, false>(float*, float const*, int, int, int)",
    "void foo<{lambda(int)#1}>(char const*, \"quoted\")",
    "void bar<{lambda(float*)#2}, \"a\\\\b\">(int)",
    "fmha_cutlassF_f16_aligned_64x64_rf_sm80",
    "void (anonymous namespace)::layer_norm_kernel<__half>(__half*, __half const*, int)"};

void append_dims(std::string& out, Rng& rng, int tensors) {
  out.push_back('[');
  for (int i = 0; i < tensors; ++i) {
    if (i) out.append(", ");
    out.push_back('[');
    const int rank = static_cast<int>(rng.range(1, 4));
    for (int d = 0; d < rank; ++d) {
      if (d) out.append(", ");
      append_int(out, 1ll << rng.range(0, 12));
    }
    out.push_back(']');
  }
  out.push_back(']');
}

}  // namespace

std::uint64_t write_kineto_trace_large(const std::string& path, const KinetoLargeSpec& spec) {
  Rng rng(spec.seed * 1000003ull + static_cast<std::uint64_t>(spec.rank));
  Sink sink(path);
  std::uint64_t events = 0;
  bool first = true;
  std::string ev, a;
  auto emit = [&] {
    if (!first) sink.put(", ");
    first = false;
    sink.put(ev);
    ev.clear();
    ++events;
  };

  sink.put("{\"schemaVersion\": 1, \"deviceProperties\": [{\"id\": 0, \"name\": \"NVIDIA H200\"}], \"traceEvents\": [");
  // Metadata first, like real traces.
  for (const char* m : {"{\"ph\": \"M\", \"name\": \"process_name\", \"pid\": 0, \"tid\": 0, \"args\": {\"name\": \"python\"}}",
                        "{\"ph\": \"M\", \"name\": \"process_name\", \"pid\": 1, \"tid\": 0, \"args\": {\"name\": \"python\"}}",
                        "{\"ph\": \"M\", \"name\": \"thread_name\", \"pid\": 0, \"tid\": 7, \"args\": {\"name\": \"stream 7\"}}"}) {
    ev.assign(m);
    emit();
  }

  double cpu_ts = 1000.0;
  double gpu_free = 1000.0;
  std::int64_t corr = 100, ext_id = 1, py_id = 1;
  int step = 0;
  while (sink.bytes() < spec.target_bytes) {
    const double step_start = cpu_ts;
    double last_end = cpu_ts;
    for (int layer = 0; layer < spec.layers_per_step; ++layer) {
      const auto op = kAtenOps[static_cast<std::size_t>(rng.range(0, kAtenOps.size() - 1))];
      // At least 36 us, so the children and the launches below fit inside the op.
      const double op_dur = 36.0 + rng.unit() * 60.0;

      a = "{\"External id\": ";
      append_int(a, ext_id);
      a.append(", \"Sequence number\": ");
      append_int(a, ext_id * 3);
      a.append(", \"Fwd thread id\": 0, \"Record function id\": 0, \"Input Dims\": ");
      append_dims(a, rng, spec.tensors_per_op);
      a.append(", \"Input type\": [");
      for (int i = 0; i < spec.tensors_per_op; ++i) a.append(i ? ", \"c10::Half\"" : "\"c10::Half\"");
      a.append("], \"Input strides\": ");
      append_dims(a, rng, spec.tensors_per_op);
      a.append(", \"Concrete Inputs\": [");
      for (int i = 0; i < spec.tensors_per_op; ++i) a.append(i ? ", \"\"" : "\"\"");
      a.append("]}");
      append_x_event(ev, "cpu_op", op, cpu_ts, op_dur, 1, 1, a);
      emit();

      // Children share the op's thread, so they must nest in it and not overlap each
      // other: equal slots inside [2, op_dur - 18].
      const double slot = (op_dur - 20.0) / std::max(1, spec.child_cpu_ops_per_layer);
      for (int c = 0; c < spec.child_cpu_ops_per_layer; ++c) {
        a = "{\"External id\": ";
        append_int(a, ext_id);
        a.append(", \"Ev Idx\": ");
        append_int(a, c);
        a.push_back('}');
        append_x_event(ev, "cpu_op", kAtenOps[static_cast<std::size_t>(rng.range(0, kAtenOps.size() - 1))],
                       cpu_ts + 2.0 + slot * c, slot * 0.9, 1, 1, a);
        emit();
      }
      for (int p = 0; p < spec.python_events_per_layer; ++p) {
        a = "{\"Python id\": ";
        append_int(a, py_id);
        a.append(", \"Python parent id\": ");
        append_int(a, py_id - 1);
        a.push_back('}');
        append_x_event(ev, "python_function", "torch/nn/modules/module.py(1501): _call_impl",
                       cpu_ts, op_dur + 5.0, 1, 1, a);
        emit();
        ++py_id;
      }

      // Launches sit at the end of the op, after the children, and end inside it.
      const double launch_ts = cpu_ts + op_dur - 16.0;
      a = "{\"cbid\": 211, \"correlation\": ";
      append_int(a, corr);
      a.append(", \"External id\": ");
      append_int(a, ext_id);
      a.push_back('}');
      append_x_event(ev, "cuda_runtime", "cudaLaunchKernel", launch_ts, 5.0, 1, 1, a);
      emit();

      const bool memcpy_layer = layer % 8 == 7;
      const double kstart = std::max(gpu_free, launch_ts + 8.0);
      const double kdur = 5.0 + rng.unit() * 400.0;
      a = "{\"queued\": 0, \"device\": 0, \"context\": 1, \"stream\": 7, \"correlation\": ";
      append_int(a, corr);
      a.append(", \"registers per thread\": 128, \"shared memory\": 49152, \"grid\": [");
      append_int(a, rng.range(1, 4096));
      a.append(", 1, 1], \"block\": [256, 1, 1], \"est. achieved occupancy %\": 25}");
      append_x_event(ev, "kernel", kKernelNames[static_cast<std::size_t>(rng.range(0, kKernelNames.size() - 1))],
                     kstart, kdur, 0, 7, a);
      emit();
      // Flow arrow from the launch to the kernel.
      ev = "{\"ph\": \"s\", \"id\": ";
      append_int(ev, corr);
      ev.append(", \"pid\": 1, \"tid\": 1, \"ts\": ");
      append_double(ev, launch_ts);
      ev.append(", \"cat\": \"ac2g\", \"name\": \"ac2g\"}");
      emit();
      ev = "{\"ph\": \"f\", \"id\": ";
      append_int(ev, corr);
      ev.append(", \"pid\": 0, \"tid\": 7, \"ts\": ");
      append_double(ev, kstart);
      ev.append(", \"cat\": \"ac2g\", \"name\": \"ac2g\", \"bp\": \"e\"}");
      emit();
      gpu_free = kstart + kdur;
      ++corr;

      if (memcpy_layer) {
        const std::int64_t bytes = rng.range(1 << 10, 1 << 24);
        a = "{\"correlation\": ";
        append_int(a, corr);
        append_x_event(ev, "cuda_runtime", "cudaMemcpyAsync", cpu_ts + op_dur - 10.0, 3.0, 1, 1, a + "}");
        emit();
        a.append(", \"stream\": 7, \"bytes\": ");
        append_int(a, bytes);
        a.push_back('}');
        const double mstart = std::max(gpu_free, launch_ts + 12.0);
        const double mdur = 2.0 + static_cast<double>(bytes) / 2.0e4;
        append_x_event(ev, "gpu_memcpy", "Memcpy HtoD (Pinned -> Device)", mstart, mdur, 0, 7, a);
        emit();
        gpu_free = mstart + mdur;
        ++corr;
        a = "{\"correlation\": ";
        append_int(a, corr);
        a.append(", \"stream\": 7, \"bytes\": 4096}");
        std::string launch_args = "{\"correlation\": ";
        append_int(launch_args, corr);
        launch_args.push_back('}');
        append_x_event(ev, "cuda_runtime", "cudaMemsetAsync", cpu_ts + op_dur - 6.0, 3.0, 1, 1, launch_args);
        emit();
        append_x_event(ev, "gpu_memset", "Memset (Device)", gpu_free, 1.5, 0, 7, a);
        emit();
        gpu_free += 1.5;
        ++corr;
      }
      ++ext_id;
      cpu_ts += op_dur + 6.0;
      last_end = std::max(cpu_ts, gpu_free);
    }
    // Step-level annotations close over their children.
    append_x_event(ev, "user_annotation", "ProfilerStep#" + std::to_string(step), step_start,
                   last_end - step_start, 1, 1);
    emit();
    append_x_event(ev, "gpu_user_annotation", "ProfilerStep#" + std::to_string(step), step_start,
                   last_end - step_start, 0, 7);
    emit();
    cpu_ts = last_end + 20.0;
    ++step;
  }
  sink.put("]");
  if (spec.distributed_info_after) {
    std::string d = ", \"distributedInfo\": {\"backend\": \"nccl\", \"rank\": ";
    append_int(d, spec.rank);
    d.append(", \"world_size\": ");
    append_int(d, spec.world_size);
    d.append(", \"pg_count\": 1}");
    sink.put(d);
  }
  sink.put("}");
  sink.close();
  return events;
}

}  // namespace cta
