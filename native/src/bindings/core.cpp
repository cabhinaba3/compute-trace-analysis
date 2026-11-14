#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <stdexcept>

#include "bindings/bindings.hpp"
#include "bindings/numpy.hpp"
#include "core/dag.hpp"
#include "core/dag_edges.hpp"
#include "core/intervals.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace cta::bindings {
namespace {
template <typename T>
using Column = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
}  // namespace

void bind_core(nb::module_& m) {
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

  m.def(
      "solve_dag",
      [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> weights,
         nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu> is_comm,
         nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu> is_compute,
         nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> size_bytes,
         nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> edge_src,
         nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> edge_dst) {
        const std::size_t n = weights.shape(0);
        if (is_comm.shape(0) != n || is_compute.shape(0) != n || size_bytes.shape(0) != n)
          throw std::invalid_argument("per-vertex arrays must have the same length");
        if (edge_src.shape(0) != edge_dst.shape(0))
          throw std::invalid_argument("edge_src and edge_dst must have the same length");
        cta::DagResult r;
        {
          nb::gil_scoped_release release;
          r = cta::solve_dag(weights.data(), is_comm.data(), is_compute.data(), size_bytes.data(), n,
                             edge_src.data(), edge_dst.data(), edge_src.shape(0));
        }
        nb::dict d;
        d["num_edges"] = r.num_edges;
        d["unordered"] = r.unordered;
        d["depth"] = r.depth;
        d["makespan"] = r.makespan;
        d["critical_compute"] = r.critical_compute;
        d["critical_comm"] = r.critical_comm;
        d["critical_comm_bytes"] = r.critical_comm_bytes;
        d["mean_slack"] = r.mean_slack;
        d["zero_slack_ratio"] = r.zero_slack_ratio;
        return d;
      },
      "weights"_a, "is_comm"_a, "is_compute"_a, "size_bytes"_a, "edge_src"_a, "edge_dst"_a,
      "Critical path and slack of a graph; see core/dag.hpp for what each field means.");

  m.def(
      "build_dag_edges",
      [](Column<std::uint64_t> rank, Column<std::uint64_t> node_id, Column<bool> is_send, Column<bool> is_recv,
         Column<std::int64_t> src_rank, Column<std::int64_t> dst_rank, Column<std::int64_t> tag,
         Column<std::uint64_t> e_rank, Column<std::uint64_t> e_src, Column<std::uint64_t> e_dst, int threads) {
        const std::size_t n = rank.shape(0);
        if (node_id.shape(0) != n || is_send.shape(0) != n || is_recv.shape(0) != n || src_rank.shape(0) != n ||
            dst_rank.shape(0) != n || tag.shape(0) != n)
          throw std::invalid_argument("per-vertex arrays must have the same length");
        const std::size_t m = e_rank.shape(0);
        if (e_src.shape(0) != m || e_dst.shape(0) != m)
          throw std::invalid_argument("per-edge arrays must have the same length");
        cta::EdgeList edges;
        {
          nb::gil_scoped_release release;
          edges = cta::build_dag_edges(n, rank.data(), node_id.data(), is_send.data(), is_recv.data(), src_rank.data(),
                                       dst_rank.data(), tag.data(), m, e_rank.data(), e_src.data(), e_dst.data(),
                                       threads);
        }
        nb::dict timing;
        timing["index"] = edges.index_seconds;
        timing["dependencies"] = edges.dependency_seconds;
        timing["link"] = edges.link_seconds;
        return nb::make_tuple(to_numpy(std::move(edges.src)), to_numpy(std::move(edges.dst)), timing);
      },
      "rank"_a, "node_id"_a, "is_send"_a, "is_recv"_a, "src_rank"_a, "dst_rank"_a, "tag"_a, "e_rank"_a, "e_src"_a,
      "e_dst"_a, "threads"_a = 0,
      "Vertex-id edges of an execution graph from its per-rank dependencies and its sends and receives.");
}

}  // namespace cta::bindings
