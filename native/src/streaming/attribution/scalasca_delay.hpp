// Scalasca Backward Wait-State & Delay Root-Cause Attribution
// (David Böhme, Felix Wolf, Bronis R. de Supinski, Martin Schulz, Markus Geimer —
// IEEE IPDPS 2012 & SC 2016).
//
// Piggybacks directly on Engine C's reverse topological block sweep:
//   - At each synchronizing op (P2P comm_recv or collective barrier), waiting
//     time W(v) = max(0, ES(v) - max_local_pred_EF(v)) is attributed backward
//     along the latest-arriving synchronizing predecessor (propagating direct
//     wait states and indirect wait cascades to the root-cause culprit op/rank).
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "streaming/engines/engine.hpp"

namespace cta::streaming::attribution {

struct DelayBucket {
  double direct_wait_ns = 0.0;
  double propagated_delay_ns = 0.0;
  std::uint64_t op_count = 0;
};

// Accumulates per-(rank, op_name) direct wait states and propagated root-cause
// delays during Pass 2's reverse topological sweep, and emits the top culprits
// sorted descending by propagated_delay_ns.
class ScalascaDelayAccumulator {
 public:
  void record_direct_wait(std::uint32_t rank, std::string_view op_name, double wait_ns) {
    if (wait_ns <= 0.0) return;
    auto& b = buckets_[{rank, std::string(op_name)}];
    b.direct_wait_ns += wait_ns;
    ++b.op_count;
  }

  void record_root_cause(std::uint32_t rank, std::string_view op_name, double propagated_ns) {
    if (propagated_ns <= 0.0) return;
    auto& b = buckets_[{rank, std::string(op_name)}];
    b.propagated_delay_ns += propagated_ns;
    if (b.direct_wait_ns == 0.0) ++b.op_count;
  }

  void finalize(engines::DagStreamResult& out, bool has_recorded_comm_times, std::size_t max_rows = 256) const {
    out.provenance.delay_timing_basis = has_recorded_comm_times ? "measured" : "modeled";
    struct Row {
      std::uint32_t rank;
      std::string op_name;
      double direct;
      double propagated;
      std::uint64_t count;
    };
    std::vector<Row> rows;
    rows.reserve(buckets_.size());
    for (const auto& [key, val] : buckets_) {
      rows.push_back({key.first, key.second, val.direct_wait_ns, val.propagated_delay_ns, val.op_count});
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
      if (a.propagated != b.propagated) return a.propagated > b.propagated;
      if (a.direct != b.direct) return a.direct > b.direct;
      if (a.rank != b.rank) return a.rank < b.rank;
      return a.op_name < b.op_name;
    });
    if (rows.size() > max_rows) rows.resize(max_rows);
    out.delay_rank.reserve(rows.size());
    out.delay_op_name.reserve(rows.size());
    out.delay_direct_wait_ns.reserve(rows.size());
    out.delay_propagated_ns.reserve(rows.size());
    out.delay_op_count.reserve(rows.size());
    for (const auto& r : rows) {
      out.delay_rank.push_back(r.rank);
      out.delay_op_name.push_back(r.op_name);
      out.delay_direct_wait_ns.push_back(r.direct);
      out.delay_propagated_ns.push_back(r.propagated);
      out.delay_op_count.push_back(r.count);
    }
  }

 private:
  std::map<std::pair<std::uint32_t, std::string>, DelayBucket> buckets_;
};

}  // namespace cta::streaming::attribution
