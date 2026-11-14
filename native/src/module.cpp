#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "chakra.hpp"
#include "goal.hpp"
#include "intervals.hpp"
#include "kineto.hpp"
#include "kineto_read.hpp"
#include "nsys.hpp"
#include "small.hpp"
#include "training.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T>
const char* dtype_name();
template <> const char* dtype_name<std::int8_t>() { return "int8"; }
template <> const char* dtype_name<std::int32_t>() { return "int32"; }
template <> const char* dtype_name<std::int64_t>() { return "int64"; }
template <> const char* dtype_name<double>() { return "float64"; }

// Hands a vector to NumPy without copying; NumPy frees it when the array dies.
template <typename T>
nb::object to_numpy(std::vector<T>&& values) {
  if (values.empty()) return nb::module_::import_("numpy").attr("empty")(0, dtype_name<T>());
  auto* owned = new std::vector<T>(std::move(values));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  nb::ndarray<nb::numpy, T, nb::ndim<1>> array(owned->data(), {owned->size()}, owner);
  return nb::cast(array);
}

nb::dict kineto_columns_to_dict(cta::KinetoColumns&& c) {
  nb::dict d;
  if (c.rank) d["rank"] = *c.rank; else d["rank"] = nb::none();
  d["names"] = nb::cast(c.names);
  d["gpu_name"] = to_numpy(std::move(c.gpu_name));
  d["gpu_category"] = to_numpy(std::move(c.gpu_category));
  d["gpu_start"] = to_numpy(std::move(c.gpu_start));
  d["gpu_end"] = to_numpy(std::move(c.gpu_end));
  d["gpu_bytes"] = to_numpy(std::move(c.gpu_bytes));
  d["gpu_stream"] = to_numpy(std::move(c.gpu_stream));
  d["gpu_device"] = to_numpy(std::move(c.gpu_device));
  d["gpu_launch_ns"] = to_numpy(std::move(c.gpu_launch_ns));
  d["gpu_launch_thread"] = to_numpy(std::move(c.gpu_launch_thread));
  d["op_start"] = to_numpy(std::move(c.op_start));
  d["op_end"] = to_numpy(std::move(c.op_end));
  d["op_name"] = to_numpy(std::move(c.op_name));
  d["op_thread"] = to_numpy(std::move(c.op_thread));
  d["step_id"] = to_numpy(std::move(c.step_id));
  d["step_start"] = to_numpy(std::move(c.step_start));
  d["step_end"] = to_numpy(std::move(c.step_end));
  d["runtime_name"] = to_numpy(std::move(c.runtime_name));
  d["runtime_duration"] = to_numpy(std::move(c.runtime_duration));
  d["runtime_calls"] = to_numpy(std::move(c.runtime_calls));
  nb::dict timing;
  timing["read"] = c.read_seconds;
  timing["split"] = c.split_seconds;
  timing["submit_wait"] = c.submit_wait_seconds;
  timing["finish"] = c.finish_seconds;
  d["timing"] = timing;
  return d;
}

}  // namespace

NB_MODULE(_native, m) {
  m.doc() = "Native helpers for compute-trace-analysis (synthetic trace generators).";

  m.def(
      "write_kineto_trace",
      [](const std::string& path, int steps, double attn_us, double gemm_us,
         const std::string& kernel_variant) {
        cta::write_kineto_trace(path, steps, attn_us, gemm_us, kernel_variant);
      },
      "path"_a, "steps"_a = 3, "attn_us"_a = 400.0, "gemm_us"_a = 600.0,
      "kernel_variant"_a = "fp16", nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "write_kineto_trace_large",
      [](const std::string& path, std::uint64_t target_bytes, std::uint64_t seed,
         int rank, int world_size, int layers_per_step, int child_cpu_ops_per_layer,
         int python_events_per_layer, int tensors_per_op, bool distributed_info_after) {
        cta::KinetoLargeSpec s;
        s.target_bytes = target_bytes;
        s.seed = seed;
        s.rank = rank;
        s.world_size = world_size;
        s.layers_per_step = layers_per_step;
        s.child_cpu_ops_per_layer = child_cpu_ops_per_layer;
        s.python_events_per_layer = python_events_per_layer;
        s.tensors_per_op = tensors_per_op;
        s.distributed_info_after = distributed_info_after;
        return cta::write_kineto_trace_large(path, s);
      },
      "path"_a, "target_bytes"_a = 200ull << 20, "seed"_a = 1, "rank"_a = 0,
      "world_size"_a = 1, "layers_per_step"_a = 24, "child_cpu_ops_per_layer"_a = 3,
      "python_events_per_layer"_a = 2, "tensors_per_op"_a = 3,
      "distributed_info_after"_a = true, nb::call_guard<nb::gil_scoped_release>());

  using nb::call_guard;
  using Nogil = nb::gil_scoped_release;

  m.def("write_goal_allreduce", &cta::write_goal_allreduce, "path"_a, "world_size"_a = 4,
        "compute_ns"_a = 1000000, "allreduce_bytes"_a = 400000, "steps"_a = 2,
        "algorithm"_a = "ring", call_guard<Nogil>());
  m.def("write_goal_allreduce_bin", &cta::write_goal_allreduce_bin, "path"_a, "world_size"_a = 4,
        "compute_ns"_a = 1000000, "allreduce_bytes"_a = 400000, "steps"_a = 2,
        "algorithm"_a = "ring", call_guard<Nogil>());

  m.def("write_nsys_sqlite", &cta::write_nsys_sqlite, "path"_a, "steps"_a = 4, "gemm_us"_a = 800,
        "comm_us"_a = 400, "comm_overlap_us"_a = 100, "warmup_penalty_us"_a = 1200,
        "lock_wait_us"_a = 50, "tensor_active_pct"_a = 65.0, "legacy_gpu_metrics"_a = false,
        call_guard<Nogil>());

  m.def(
      "write_chakra_et_run",
      [](const std::string& out_dir, int world_size, std::int64_t compute_us,
         std::int64_t allreduce_bytes, int steps, std::optional<int> straggler_rank,
         double straggler_factor) {
        nb::gil_scoped_release release;
        return cta::write_chakra_et_run(out_dir, world_size, compute_us, allreduce_bytes, steps,
                                        straggler_rank.value_or(-1), straggler_factor);
      },
      "out_dir"_a, "world_size"_a = 4, "compute_us"_a = 1000, "allreduce_bytes"_a = 400000,
      "steps"_a = 2, "straggler_rank"_a = nb::none(), "straggler_factor"_a = 1.0);
  m.def("write_chakra_host_et", &cta::write_chakra_host_et, "path"_a, "steps"_a = 2,
        "compute_us"_a = 500.0, "comm_bytes"_a = 200000, call_guard<Nogil>());

  m.def("write_ncu_csv", &cta::write_ncu_csv, "path"_a, "gemm_dur_ns"_a = 500000.0,
        "sm_pct"_a = 40.0, "dram_pct"_a = 82.0, "tensor_pct"_a = 35.0, "flops"_a = 2.0e10,
        "dram_bytes"_a = 1.0e9, call_guard<Nogil>());
  m.def("write_jtop_trace", &cta::write_jtop_trace, "path"_a, "duration_s"_a = 5,
        "num_cores"_a = 8, "power_mw"_a = 15000.0, "gpu_util"_a = 85.0, "gpu_mem_mb"_a = 2048.0,
        "temp_c"_a = std::optional<double>(62.0), call_guard<Nogil>());
  m.def("write_trtexec_timing", &cta::write_trtexec_timing, "path"_a, "queries"_a = 20,
        "h2d_ms"_a = 0.4, "compute_ms"_a = 2.0, "d2h_ms"_a = 0.3, "overlap_ms"_a = 0.2,
        "idle_gap_ms"_a = 0.1, call_guard<Nogil>());
  m.def("write_trtexec_layers", &cta::write_trtexec_layers, "profile_path"_a, "engine_path"_a,
        "fused"_a = false, "precision"_a = "FP16", call_guard<Nogil>());

  nb::class_<cta::TrainingSpec>(m, "TrainingSpec")
      .def(nb::init<>())
      .def_rw("dp", &cta::TrainingSpec::dp)
      .def_rw("pp", &cta::TrainingSpec::pp)
      .def_rw("tp", &cta::TrainingSpec::tp)
      .def_rw("microbatches", &cta::TrainingSpec::microbatches)
      .def_rw("layers_per_stage", &cta::TrainingSpec::layers_per_stage)
      .def_rw("steps", &cta::TrainingSpec::steps)
      .def_rw("moe_every", &cta::TrainingSpec::moe_every)
      .def_rw("ep_size", &cta::TrainingSpec::ep_size)
      .def_rw("dp_buckets", &cta::TrainingSpec::dp_buckets)
      .def_rw("layer_fwd_ns", &cta::TrainingSpec::layer_fwd_ns)
      .def_rw("layer_bwd_ns", &cta::TrainingSpec::layer_bwd_ns)
      .def_rw("optimizer_ns", &cta::TrainingSpec::optimizer_ns)
      .def_rw("expert_ns", &cta::TrainingSpec::expert_ns)
      .def_rw("tp_bytes", &cta::TrainingSpec::tp_bytes)
      .def_rw("pp_bytes", &cta::TrainingSpec::pp_bytes)
      .def_rw("dp_bytes", &cta::TrainingSpec::dp_bytes)
      .def_rw("moe_bytes", &cta::TrainingSpec::moe_bytes)
      .def("ranks", &cta::TrainingSpec::ranks)
      .def("validate", &cta::TrainingSpec::validate);

  m.def("write_goal_training", &cta::write_goal_training, "path"_a, "spec"_a,
        call_guard<Nogil>());
  m.def("write_goal_training_bin", &cta::write_goal_training_bin, "path"_a, "spec"_a,
        call_guard<Nogil>());
  m.def("write_chakra_training", &cta::write_chakra_training, "out_dir"_a, "spec"_a,
        "threads"_a = 0, call_guard<Nogil>());

  m.def(
      "read_kineto_trace",
      [](const std::string& path, int threads, std::uint64_t chunk_bytes,
         std::uint64_t block_bytes) {
        cta::KinetoColumns columns;
        {
          nb::gil_scoped_release release;
          columns = cta::read_kineto_columns(path, threads, chunk_bytes, block_bytes);
        }
        return kineto_columns_to_dict(std::move(columns));
      },
      "path"_a, "threads"_a = 0, "chunk_bytes"_a = 32ull << 20, "block_bytes"_a = 8ull << 20,
      "Reads a Kineto trace (.json or .json.gz) into dictionary-coded columns.");

  m.def(
      "nesting_parents",
      [](nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> starts,
         nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> ends) {
        if (starts.shape(0) != ends.shape(0))
          throw std::invalid_argument("starts and ends must have the same length");
        return to_numpy(cta::nesting_parents(starts.data(), ends.data(), starts.shape(0)));
      },
      "starts"_a, "ends"_a,
      "Enclosing range of each range, for ranges sorted by start (-1 for none).");
}
