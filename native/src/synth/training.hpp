#pragma once

#include <cstdint>
#include <string>

namespace cta {

// One training job on dp x pp x tp ranks (rank = (d * pp + p) * tp + t), written as
// a GPipe-style schedule: all microbatch forwards, then all backwards (reversed),
// with tensor-parallel all-reduces per layer, optional MoE all-to-alls inside
// expert groups of `ep_size` data-parallel ranks, bucketed data-parallel gradient
// all-reduces that overlap the last microbatch's backward, and an optimizer calc.
struct TrainingSpec {
  int dp = 2, pp = 2, tp = 2;
  int microbatches = 4;
  int layers_per_stage = 4;
  int steps = 2;
  int moe_every = 0;  // 0 = no MoE; N = every N-th layer is an MoE layer
  int ep_size = 1;    // expert group size; must divide dp
  int dp_buckets = 2;
  std::int64_t layer_fwd_ns = 1000000;
  std::int64_t layer_bwd_ns = 2000000;
  std::int64_t optimizer_ns = 500000;
  std::int64_t expert_ns = 800000;
  std::int64_t tp_bytes = 8ll << 20;
  std::int64_t pp_bytes = 16ll << 20;
  std::int64_t dp_bytes = 64ll << 20;
  std::int64_t moe_bytes = 8ll << 20;

  int ranks() const { return dp * pp * tp; }
  // Throws std::invalid_argument for sizes that cannot be generated.
  void validate() const;
};

// GOAL text: collectives are spelled out as send/recv (ring all-reduce, pairwise
// all-to-all). Returns the total node count.
std::uint64_t write_goal_training(const std::string& path, const TrainingSpec& spec);

// The same schedule in the SPCL `.bin` layout, written directly.
std::uint64_t write_goal_training_bin(const std::string& path, const TrainingSpec& spec);

// Chakra `.et` shards, one per rank: a collective is one COMM_COLL node, and
// only pipeline traffic is send/recv. `threads` = 0 uses the available CPUs.
std::uint64_t write_chakra_training(const std::string& out_dir, const TrainingSpec& spec,
                                    int threads);

}  // namespace cta
