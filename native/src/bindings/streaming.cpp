#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bindings/bindings.hpp"
#include "bindings/convert.hpp"
#include "streaming/batch.hpp"
#include "streaming/engines/engine.hpp"
#include "streaming/engines/frontier/frontier_engine.hpp"
#include "streaming/engines/online_agg/online_agg.hpp"
#include "streaming/sketches/quantile.hpp"
#include "streaming/sources/source.hpp"
#include "util/memory.hpp"
#include "util/util.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace cta::bindings {
namespace {

streaming::GoalSemantics parse_goal_semantics(const std::string& text) {
  if (text == "cta_legacy" || text == "legacy") return streaming::GoalSemantics::kCtaLegacy;
  if (text == "strict") return streaming::GoalSemantics::kStrict;
  throw std::invalid_argument("goal_semantics must be 'cta_legacy' or 'strict', got '" + text + "'");
}

streaming::CollectiveSync parse_collective_sync(const std::string& text) {
  if (text == "none") return streaming::CollectiveSync::kNone;
  if (text == "barrier") return streaming::CollectiveSync::kBarrier;
  throw std::invalid_argument("collective_sync must be 'none' or 'barrier', got '" + text + "'");
}

streaming::engines::DagSolveOptions build_dag_options(
    const std::string& engine, std::uint64_t memory_budget_bytes, const std::string& goal_semantics,
    const std::string& collective_sync, std::uint32_t world_size_hint,
    const std::unordered_map<std::string, std::uint32_t>& coll_group_sizes, std::size_t window_size,
    std::size_t max_lookahead, bool fail_on_cycle, bool extract_critical_path_ops, bool delay_attribution,
    const std::string& scratch_dir, std::size_t chunk_nodes) {
  streaming::engines::DagSolveOptions opts;
  opts.engine = engine;
  opts.memory_budget_bytes = memory_budget_bytes;
  opts.goal_semantics = parse_goal_semantics(goal_semantics);
  opts.collective_sync = parse_collective_sync(collective_sync);
  opts.world_size_hint = world_size_hint;
  opts.coll_group_sizes = coll_group_sizes;
  opts.window_size = window_size > 0 ? window_size : 262144;
  opts.max_lookahead = max_lookahead > 0 ? max_lookahead : 65536;
  opts.fail_on_cycle = fail_on_cycle;
  opts.extract_critical_path_ops = extract_critical_path_ops;
  opts.delay_attribution = delay_attribution;
  opts.scratch_dir = scratch_dir;
  opts.chunk_nodes = chunk_nodes > 0 ? chunk_nodes : 524288;
  return opts;
}

class PyGraphStream {
 public:
  PyGraphStream(std::unique_ptr<streaming::GraphSource> source, std::size_t chunk_nodes)
      : source_(std::move(source)), chunk_nodes_(chunk_nodes > 0 ? chunk_nodes : 524288) {}

  nb::object next_chunk(std::size_t max_nodes) {
    if (!source_) return nb::none();
    const std::size_t limit = max_nodes > 0 ? max_nodes : chunk_nodes_;
    streaming::GraphChunk chunk;
    bool got = false;
    {
      nb::gil_scoped_release release;
      got = source_->next_chunk(chunk, limit);
    }
    if (!got) return nb::none();
    return to_dict(std::move(chunk));
  }

  nb::object iter_next() {
    nb::object chunk = next_chunk(chunk_nodes_);
    if (chunk.is_none()) throw nb::stop_iteration();
    return chunk;
  }

  bool is_seekable() const { return source_ && source_->is_seekable(); }
  std::string format_name() const { return source_ ? source_->format_name() : ""; }
  std::uint64_t estimated_bytes() const { return source_ ? source_->estimated_bytes() : 0; }
  void reset() {
    if (!source_) return;
    nb::gil_scoped_release release;
    source_->reset();
  }
  streaming::GraphSource& source() {
    if (!source_) throw std::runtime_error("GraphStream has no source");
    return *source_;
  }

 private:
  std::unique_ptr<streaming::GraphSource> source_;
  std::size_t chunk_nodes_;
};

}  // namespace

void bind_streaming(nb::module_& m) {
  m.def("available_memory_bytes", &cta::available_memory_bytes,
        "Returns available host/cgroup memory in bytes, or 0 if unavailable.");
  m.def("resolve_memory_budget", &cta::resolve_memory_budget, "requested_bytes"_a = 0,
        "Returns requested_bytes if > 0, else 60% of available memory (or 4 GiB fallback).");

  nb::class_<PyGraphStream>(m, "GraphStream", "Bounded-memory streaming reader over a GOAL or Chakra execution graph.")
      .def("next_chunk", &PyGraphStream::next_chunk, "max_nodes"_a = 0,
           "Returns the next chunk dict of zero-copy NumPy arrays, or None at EOF.")
      .def("__iter__", [](PyGraphStream& self) -> PyGraphStream& { return self; }, nb::rv_policy::reference)
      .def("__next__", &PyGraphStream::iter_next)
      .def_prop_ro("is_seekable", &PyGraphStream::is_seekable)
      .def_prop_ro("format_name", &PyGraphStream::format_name)
      .def_prop_ro("estimated_bytes", &PyGraphStream::estimated_bytes)
      .def("reset", &PyGraphStream::reset, "Resets a seekable stream back to the start.");

  m.def(
      "open_goal_bin_stream",
      [](const std::string& path, std::size_t chunk_nodes) {
        std::unique_ptr<streaming::GraphSource> src;
        {
          nb::gil_scoped_release release;
          src = streaming::open_goal_bin_stream(path);
        }
        return PyGraphStream(std::move(src), chunk_nodes);
      },
      "path"_a, "chunk_nodes"_a = 524288, "Opens a GOAL binary (.bin) file or pipe as a chunked GraphStream.");

  m.def(
      "open_goal_text_stream",
      [](const std::string& path, const std::string& goal_semantics, std::size_t chunk_nodes) {
        const auto sem = parse_goal_semantics(goal_semantics);
        std::unique_ptr<streaming::GraphSource> src;
        {
          nb::gil_scoped_release release;
          src = streaming::open_goal_text_stream(path, sem);
        }
        return PyGraphStream(std::move(src), chunk_nodes);
      },
      "path"_a, "goal_semantics"_a = "cta_legacy", "chunk_nodes"_a = 524288,
      "Opens a GOAL text (.goal) file or pipe as a chunked GraphStream.");

  m.def(
      "open_chakra_et_stream",
      [](const std::vector<std::string>& paths, const std::vector<std::uint32_t>& ranks, std::size_t chunk_nodes) {
        std::unique_ptr<streaming::GraphSource> src;
        {
          nb::gil_scoped_release release;
          src = streaming::open_chakra_et_stream(paths, ranks);
        }
        return PyGraphStream(std::move(src), chunk_nodes);
      },
      "paths"_a, "ranks"_a, "chunk_nodes"_a = 524288,
      "Opens Chakra .et protobuf files (one per rank) as a chunked GraphStream.");

  m.def(
      "open_chakra_host_stream",
      [](const std::vector<std::string>& paths, const std::vector<std::uint32_t>& ranks, std::size_t chunk_nodes) {
        std::unique_ptr<streaming::GraphSource> src;
        {
          nb::gil_scoped_release release;
          src = streaming::open_chakra_host_stream(paths, ranks);
        }
        return PyGraphStream(std::move(src), chunk_nodes);
      },
      "paths"_a, "ranks"_a, "chunk_nodes"_a = 524288,
      "Opens Chakra host execution trace JSON files as a chunked GraphStream.");

  m.def(
      "read_goal_bin",
      [](const std::string& path) {
        streaming::GraphChunk all;
        {
          nb::gil_scoped_release release;
          auto src = streaming::open_goal_bin_stream(path);
          streaming::GraphChunk part;
          while (src->next_chunk(part, 1u << 22)) {
            const std::uint64_t name_base = all.nodes.name_bytes.size();
            for (std::uint64_t end : part.nodes.name_end) all.nodes.name_end.push_back(name_base + end);
            append_vector(all.nodes.name_bytes, part.nodes.name_bytes);
            const std::uint64_t ct_base = all.nodes.comm_type_bytes.size();
            for (std::uint64_t end : part.nodes.comm_type_end) all.nodes.comm_type_end.push_back(ct_base + end);
            append_vector(all.nodes.comm_type_bytes, part.nodes.comm_type_bytes);
            const std::uint64_t cg_base = all.nodes.comm_group_bytes.size();
            for (std::uint64_t end : part.nodes.comm_group_end) all.nodes.comm_group_end.push_back(cg_base + end);
            append_vector(all.nodes.comm_group_bytes, part.nodes.comm_group_bytes);
            append_vector(all.nodes.rank, part.nodes.rank);
            append_vector(all.nodes.node_id, part.nodes.node_id);
            append_vector(all.nodes.op_class, part.nodes.op_class);
            append_vector(all.nodes.duration_ns, part.nodes.duration_ns);
            append_vector(all.nodes.size_bytes, part.nodes.size_bytes);
            append_vector(all.nodes.num_ops, part.nodes.num_ops);
            append_vector(all.nodes.src_rank, part.nodes.src_rank);
            append_vector(all.nodes.dst_rank, part.nodes.dst_rank);
            append_vector(all.nodes.message_tag, part.nodes.message_tag);
            append_vector(all.edges.rank, part.edges.rank);
            append_vector(all.edges.src_id, part.edges.src_id);
            append_vector(all.edges.dst_id, part.edges.dst_id);
            append_vector(all.edges.edge_kind, part.edges.edge_kind);
          }
        }
        return to_dict(std::move(all));
      },
      "path"_a, "Reads a GOAL binary (.bin) schedule into node and edge columns in C++.");

  m.def(
      "stream_aggregates",
      [](PyGraphStream& stream, bool include_collectives_in_traffic, std::uint32_t min_world_size,
         const std::string& quantile_sketch, std::size_t quantile_exact_cap, double quantile_accuracy,
         std::size_t chunk_nodes) {
        streaming::engines::OnlineAggOptions opts;
        opts.include_collectives_in_traffic = include_collectives_in_traffic;
        opts.min_world_size = min_world_size;
        opts.quantile_sketch = quantile_sketch;
        opts.quantile_exact_cap = quantile_exact_cap;
        opts.quantile_accuracy = quantile_accuracy;
        opts.chunk_nodes = chunk_nodes;
        streaming::engines::OnlineAggResult r;
        {
          nb::gil_scoped_release release;
          r = streaming::engines::run_online_aggregates(stream.source(), opts);
        }
        return to_dict(std::move(r));
      },
      "stream"_a, "include_collectives_in_traffic"_a = true, "min_world_size"_a = 1, "quantile_sketch"_a = "ddsketch",
      "quantile_exact_cap"_a = 65536, "quantile_accuracy"_a = 0.001, "chunk_nodes"_a = 524288,
      "Runs Engine A (1-pass O(P^2) aggregate analysis + pluggable quantile sketch) over a GraphStream.");

  nb::class_<streaming::sketches::HybridQuantile>(
      m, "HybridQuantile", "Exact-first quantile estimator that spills to DDSketch, KLL, or Greenwald-Khanna.")
      .def(nb::init<std::string, std::size_t, double>(), "sketch_kind"_a = "ddsketch", "exact_cap"_a = 65536,
           "accuracy"_a = 0.001)
      .def("add", &streaming::sketches::HybridQuantile::add, "value"_a, "weight"_a = 1)
      .def(
          "add_array",
          [](streaming::sketches::HybridQuantile& self,
             nb::ndarray<const std::uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> values) {
            nb::gil_scoped_release release;
            const std::uint64_t* data = values.data();
            const std::size_t n = values.shape(0);
            for (std::size_t i = 0; i < n; ++i) self.add(data[i]);
          },
          "values"_a)
      .def("quantile", &streaming::sketches::HybridQuantile::quantile, "q"_a)
      .def_prop_ro("count", &streaming::sketches::HybridQuantile::count)
      .def_prop_ro("is_exact", &streaming::sketches::HybridQuantile::is_exact)
      .def_prop_ro("sketch_name",
                   [](const streaming::sketches::HybridQuantile& self) { return std::string(self.sketch_name()); })
      .def_prop_ro("error_bound", &streaming::sketches::HybridQuantile::error_bound);

  m.def("available_graph_engines", &streaming::engines::available_graph_engines,
        "Returns the registered pluggable streaming/out-of-core DAG engine names.");

  m.def(
      "solve_dag_stream",
      [](PyGraphStream& stream, const std::string& engine, std::uint64_t memory_budget_bytes,
         const std::string& goal_semantics, const std::string& collective_sync, std::uint32_t world_size_hint,
         const std::unordered_map<std::string, std::uint32_t>& coll_group_sizes, std::size_t window_size,
         std::size_t max_lookahead, bool fail_on_cycle, bool extract_critical_path_ops, bool delay_attribution,
         const std::string& scratch_dir, std::size_t chunk_nodes) {
        const auto opts = build_dag_options(engine, memory_budget_bytes, goal_semantics, collective_sync,
                                            world_size_hint, coll_group_sizes, window_size, max_lookahead,
                                            fail_on_cycle, extract_critical_path_ops, delay_attribution, scratch_dir,
                                            chunk_nodes);
        streaming::engines::DagStreamResult r;
        {
          nb::gil_scoped_release release;
          auto eng = streaming::engines::make_graph_engine(opts.engine);
          r = eng->solve_dag(stream.source(), opts);
        }
        return to_dict(std::move(r));
      },
      "stream"_a, "engine"_a = "frontier", "memory_budget_bytes"_a = 0, "goal_semantics"_a = "cta_legacy",
      "collective_sync"_a = "none", "world_size_hint"_a = 0,
      "coll_group_sizes"_a = std::unordered_map<std::string, std::uint32_t>{}, "window_size"_a = 262144,
      "max_lookahead"_a = 65536, "fail_on_cycle"_a = false, "extract_critical_path_ops"_a = false,
      "delay_attribution"_a = false, "scratch_dir"_a = "", "chunk_nodes"_a = 524288,
      "Solves an execution graph using a registered pluggable streaming/out-of-core GraphEngine.");

  m.def(
      "solve_goal_bin_frontier",
      [](const std::string& path, const std::string& goal_semantics, bool fail_on_cycle,
         bool extract_critical_path_ops, const std::string& scratch_dir) {
        const auto opts = build_dag_options("frontier", 0, goal_semantics, "none", 0, {}, 262144, 65536, fail_on_cycle,
                                            extract_critical_path_ops, false, scratch_dir, 524288);
        streaming::engines::DagStreamResult r;
        {
          nb::gil_scoped_release release;
          r = streaming::engines::solve_goal_bin_frontier(path, opts);
        }
        return to_dict(std::move(r));
      },
      "path"_a, "goal_semantics"_a = "cta_legacy", "fail_on_cycle"_a = false, "extract_critical_path_ops"_a = false,
      "scratch_dir"_a = "", "Exact O(frontier) multi-rank Kahn DAG solver directly over an mmapped GOAL .bin file.");

  m.def(
      "solve_chakra_et_frontier",
      [](const std::vector<std::string>& paths, const std::vector<std::uint32_t>& ranks,
         const std::string& collective_sync, const std::unordered_map<std::string, std::uint32_t>& coll_group_sizes,
         std::size_t window_size, std::size_t max_lookahead, bool fail_on_cycle, bool extract_critical_path_ops,
         const std::string& scratch_dir) {
        const auto opts = build_dag_options("frontier", 0, "cta_legacy", collective_sync,
                                            static_cast<std::uint32_t>(paths.size()), coll_group_sizes, window_size,
                                            max_lookahead, fail_on_cycle, extract_critical_path_ops, false, scratch_dir,
                                            524288);
        streaming::engines::DagStreamResult r;
        {
          nb::gil_scoped_release release;
          r = streaming::engines::solve_chakra_et_frontier(paths, ranks, opts);
        }
        return to_dict(std::move(r));
      },
      "paths"_a, "ranks"_a, "collective_sync"_a = "none",
      "coll_group_sizes"_a = std::unordered_map<std::string, std::uint32_t>{}, "window_size"_a = 262144,
      "max_lookahead"_a = 65536, "fail_on_cycle"_a = false, "extract_critical_path_ops"_a = false,
      "scratch_dir"_a = "",
      "Multi-rank lockstep 1-pass frontier DAG solver over Chakra .et files with Certified Eviction-Epoch bounds.");
}

}  // namespace cta::bindings
