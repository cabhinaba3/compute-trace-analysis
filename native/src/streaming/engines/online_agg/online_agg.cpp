#include "streaming/engines/online_agg/online_agg.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "streaming/sketches/ddsketch.hpp"
#include "streaming/sketches/greenwald_khanna.hpp"
#include "streaming/sketches/kll.hpp"
#include "streaming/sketches/quantile.hpp"
#include "util/util.hpp"

namespace cta::streaming::sketches {

std::unique_ptr<QuantileSketch> make_sketch(const std::string& kind, double accuracy) {
  if (kind == "ddsketch") return std::make_unique<DDSketch>(accuracy);
  if (kind == "kll") return std::make_unique<KLLSketch>(accuracy);
  if (kind == "gk" || kind == "greenwald_khanna") return std::make_unique<GKSketch>(accuracy);
  throw std::invalid_argument("unknown quantile_sketch '" + kind + "'; use ddsketch, kll, or gk");
}

}  // namespace cta::streaming::sketches

namespace cta::streaming::engines {
namespace {

struct RankAcc {
  std::uint64_t compute_ns = 0;
  std::uint64_t comm_bytes = 0;
  std::uint64_t op_count = 0;
};

struct LinkAcc {
  std::uint64_t msgs = 0;
  std::uint64_t bytes = 0;
};

struct TypeAcc {
  std::uint64_t bytes = 0;
  std::uint64_t count = 0;
};

struct CollKey {
  std::uint32_t rank = 0;
  std::string comm_type;
  std::uint64_t size_bytes = 0;
  bool operator==(const CollKey& o) const {
    return rank == o.rank && size_bytes == o.size_bytes && comm_type == o.comm_type;
  }
};

struct CollKeyHash {
  std::size_t operator()(const CollKey& k) const {
    std::size_t h = static_cast<std::size_t>(mix64(k.rank * 0x9e3779b97f4a7c15ULL ^ k.size_bytes));
    h ^= std::hash<std::string>{}(k.comm_type) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

bool is_ring_collective(std::string_view ct) {
  return ct == "ALL_GATHER" || ct == "REDUCE_SCATTER" || ct == "REDUCE_SCATTER_BLOCK" || ct == "BROADCAST" ||
         ct == "REDUCE";
}

}  // namespace

OnlineAggResult run_online_aggregates(GraphSource& source, const OnlineAggOptions& opts) {
  OnlineAggResult res;
  res.provenance.engine = "online_agg";
  res.provenance.slack_scope = "unavailable";

  std::map<std::uint32_t, RankAcc> per_rank;
  std::map<std::pair<std::uint32_t, std::uint32_t>, LinkAcc> traffic;
  std::map<std::string, TypeAcc> by_comm_type;
  std::map<std::string, std::uint64_t> coll_bytes;
  std::unordered_map<CollKey, std::uint64_t, CollKeyHash> coll_groups;
  sketches::HybridQuantile msg_sizes(opts.quantile_sketch, opts.quantile_exact_cap, opts.quantile_accuracy);

  std::uint32_t max_rank_seen = 0;
  bool has_nodes = false;
  std::uint64_t total_comm_bytes_u64 = 0;
  std::uint64_t max_msg_bytes = 0;

  GraphChunk chunk;
  const std::size_t batch_limit = opts.chunk_nodes > 0 ? opts.chunk_nodes : 524288;
  while (source.next_chunk(chunk, batch_limit)) {
    const NodeBatch& nb = chunk.nodes;
    const std::size_t n = nb.size();
    res.num_nodes += n;
    res.num_raw_edges += chunk.edges.size();

    for (std::size_t i = 0; i < n; ++i) {
      has_nodes = true;
      const std::uint32_t r = nb.rank[i];
      if (r > max_rank_seen) max_rank_seen = r;

      const std::uint8_t oc = nb.op_class[i];
      const std::uint64_t dur_ns = nb.duration_ns[i];
      const std::uint64_t sz = nb.size_bytes[i];

      RankAcc& ra = per_rank[r];
      ++ra.op_count;
      if (oc == static_cast<std::uint8_t>(OpClass::kCompute)) {
        ra.compute_ns += dur_ns;
        res.total_compute_ns += dur_ns;
      }

      const bool is_send = oc == static_cast<std::uint8_t>(OpClass::kCommSend);
      const bool is_recv = oc == static_cast<std::uint8_t>(OpClass::kCommRecv);
      const bool is_coll = oc == static_cast<std::uint8_t>(OpClass::kCommColl);

      if (is_send) {
        ++res.send_count;
        res.total_send_bytes += sz;
        if (nb.dst_rank[i] >= 0) {
          LinkAcc& la = traffic[{r, static_cast<std::uint32_t>(nb.dst_rank[i])}];
          ++la.msgs;
          la.bytes += sz;
        }
      } else if (is_recv) {
        ++res.recv_count;
      }

      if (is_coll || is_send) {
        ra.comm_bytes += sz;
        total_comm_bytes_u64 += sz;
        ++res.comm_op_count;
        if (sz > max_msg_bytes) max_msg_bytes = sz;
        msg_sizes.add(sz);

        const std::string ctype(nb.comm_type_at(i));
        TypeAcc& ta = by_comm_type[ctype];
        ta.bytes += sz;
        ++ta.count;

        if (is_coll) {
          coll_bytes[ctype] += sz;
          if (opts.include_collectives_in_traffic && sz > 0) {
            ++coll_groups[CollKey{r, ctype, sz}];
          }
        }
      }
    }
  }

  // Fold collective traffic at EOF once exact world_size is known.
  const std::uint32_t traffic_world_size =
      has_nodes ? std::max<std::uint32_t>({opts.min_world_size, max_rank_seen + 1, 1}) : 0;
  if (opts.include_collectives_in_traffic && traffic_world_size > 1) {
    const std::uint64_t P = traffic_world_size;
    for (const auto& [ck, count] : coll_groups) {
      const std::uint32_t next_rank = static_cast<std::uint32_t>((ck.rank + 1) % traffic_world_size);
      if (ck.comm_type == "ALL_REDUCE") {
        LinkAcc& la = traffic[{ck.rank, next_rank}];
        la.msgs += count * (2 * (P - 1));
        la.bytes += count * ((ck.size_bytes * (2 * (P - 1))) / P);
      } else if (is_ring_collective(ck.comm_type)) {
        LinkAcc& la = traffic[{ck.rank, next_rank}];
        la.msgs += count * (P - 1);
        la.bytes += count * ((ck.size_bytes * (P - 1)) / P);
      } else if (ck.comm_type == "ALL_TO_ALL") {
        const std::uint64_t per_peer = count * (ck.size_bytes / P);
        for (std::uint32_t dst = 0; dst < traffic_world_size; ++dst) {
          if (dst == ck.rank) continue;
          LinkAcc& la = traffic[{ck.rank, dst}];
          la.msgs += count;
          la.bytes += per_peer;
        }
      }
    }
  }

  // Finalize per-rank skew statistics (matching src/cta/aspects/distributed/rank_skew.py).
  res.world_size = static_cast<std::uint32_t>(per_rank.size());
  res.rank_ids.reserve(per_rank.size());
  res.rank_compute_ns.reserve(per_rank.size());
  res.rank_comm_bytes.reserve(per_rank.size());
  res.rank_op_count.reserve(per_rank.size());

  long double sum_compute_us = 0.0L;
  long double sum_comm_mb = 0.0L;
  long double sum_comm_mb_sq = 0.0L;
  for (const auto& [r, ra] : per_rank) {
    res.rank_ids.push_back(r);
    res.rank_compute_ns.push_back(ra.compute_ns);
    res.rank_comm_bytes.push_back(ra.comm_bytes);
    res.rank_op_count.push_back(ra.op_count);
    const double comp_us = static_cast<double>(ra.compute_ns) / 1e3;
    const double comm_mb = static_cast<double>(ra.comm_bytes) / 1e6;
    if (comp_us > res.max_rank_compute_us) res.max_rank_compute_us = comp_us;
    if (comm_mb > res.max_rank_comm_mb) res.max_rank_comm_mb = comm_mb;
    sum_compute_us += comp_us;
    sum_comm_mb += comm_mb;
    sum_comm_mb_sq += static_cast<long double>(comm_mb) * static_cast<long double>(comm_mb);
  }
  if (res.world_size > 0) {
    res.mean_rank_compute_us = static_cast<double>(sum_compute_us / static_cast<long double>(res.world_size));
    res.compute_skew_ratio =
        res.mean_rank_compute_us > 1e-12 ? (res.max_rank_compute_us / res.mean_rank_compute_us) : 1.0;
    const long double jains_denom = static_cast<long double>(res.world_size) * sum_comm_mb_sq;
    res.comm_jains_fairness = jains_denom > 1e-12L ? static_cast<double>((sum_comm_mb * sum_comm_mb) / jains_denom)
                                                   : 1.0;
  }

  // Finalize collectives characterization.
  res.total_comm_mb = static_cast<double>(total_comm_bytes_u64) / 1e6;
  if (res.comm_op_count > 0) {
    res.median_msg_kb = msg_sizes.quantile(0.5) / 1e3;
    res.p95_msg_kb = msg_sizes.quantile(0.95) / 1e3;
    res.max_msg_kb = static_cast<double>(max_msg_bytes) / 1e3;
  }
  for (const auto& [name, ta] : by_comm_type) {
    res.comm_type_names.push_back(name);
    res.comm_type_bytes.push_back(ta.bytes);
    res.comm_type_count.push_back(ta.count);
  }
  for (const auto& [name, bytes] : coll_bytes) {
    res.coll_type_names.push_back(name);
    res.coll_type_bytes.push_back(bytes);
  }

  // Finalize traffic matrix summary.
  res.active_rank_links = traffic.size();
  res.traffic_src.reserve(traffic.size());
  res.traffic_dst.reserve(traffic.size());
  res.traffic_msgs.reserve(traffic.size());
  res.traffic_bytes.reserve(traffic.size());
  std::uint64_t total_traffic_bytes = 0;
  std::uint64_t hotspot_bytes = 0;
  for (const auto& [pair, la] : traffic) {
    res.traffic_src.push_back(pair.first);
    res.traffic_dst.push_back(pair.second);
    res.traffic_msgs.push_back(la.msgs);
    res.traffic_bytes.push_back(la.bytes);
    total_traffic_bytes += la.bytes;
    if (la.bytes > hotspot_bytes) hotspot_bytes = la.bytes;
  }
  res.total_p2p_mb = static_cast<double>(total_traffic_bytes) / 1e6;
  res.hotspot_link_mb = static_cast<double>(hotspot_bytes) / 1e6;
  const std::uint32_t unique_ranks = res.world_size > 0 ? res.world_size : 1;
  res.mean_peer_degree = static_cast<double>(res.active_rank_links) / static_cast<double>(unique_ranks);

  // Honest provenance.
  res.provenance.exact = msg_sizes.is_exact();
  res.provenance.quantile_sketch = std::string(msg_sizes.sketch_name());
  res.provenance.quantile_error_bound = msg_sizes.error_bound();
  return res;
}

}  // namespace cta::streaming::engines
