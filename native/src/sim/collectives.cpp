#include "sim/collectives.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace cta::sim {
namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr std::array<const char*, kNumAlgorithms> kAlgorithmNames = {
    "ring", "recursive_doubling", "halving_doubling", "binomial_tree", "double_binary_tree", "direct", "hierarchical"};

// Chunk c of a buffer of `total` bytes cut into `parts` chunks.
u64 chunk(u64 total, u32 parts, u32 c) { return total / parts + (c < total % parts ? 1 : 0); }

bool is_pow2(u32 x) { return x != 0 && (x & (x - 1)) == 0; }
u32 floor_pow2(u32 x) {
  u32 p = 1;
  while (p <= x / 2) p *= 2;
  return p;
}
u32 ceil_log2(u32 x) {
  u32 k = 0;
  while ((u64{1} << k) < x) ++k;
  return k;
}
// (a - b) mod n for non-negative a, b.
u32 mod_sub(u64 a, u64 b, u32 n) { return static_cast<u32>((a + (b / n + 1) * n - b) % n); }

// Adds actions with explicit dependencies.
class Builder {
 public:
  explicit Builder(Schedule& s) : s_(s) {}
  u32 add(ActionKind kind, u32 peer, u64 bytes, std::span<const u32> deps) {
    Action a{kind, peer, bytes, 0, static_cast<u32>(s_.deps.size()), 0};
    s_.deps.insert(s_.deps.end(), deps.begin(), deps.end());
    a.deps_end = static_cast<u32>(s_.deps.size());
    s_.actions.push_back(a);
    return static_cast<u32>(s_.actions.size() - 1);
  }

 private:
  Schedule& s_;
};

// Adds actions in steps. A step's sends and recvs wait for the latest step's
// reductions, else its recvs; a reduction waits for its own step's recvs.
// Steps that only send change nothing, so a broadcast root starts every send
// at once and its CPU and NIC order them.
class Steps {
 public:
  Steps(Schedule& s, bool reduce) : b_(s), reduce_(reduce) {}
  void send(u32 peer, u64 bytes) { b_.add(ActionKind::kSend, peer, bytes, prev_); }
  void recv(u32 peer, u64 bytes) { recvs_.push_back(b_.add(ActionKind::kRecv, peer, bytes, prev_)); }
  void reduce(u64 bytes) {
    if (reduce_ && bytes > 0 && !recvs_.empty()) reduces_.push_back(b_.add(ActionKind::kReduce, 0, bytes, recvs_));
  }
  void end() {
    if (!reduces_.empty()) prev_ = reduces_;
    else if (!recvs_.empty()) prev_ = recvs_;
    recvs_.clear();
    reduces_.clear();
  }
  const std::vector<u32>& last() const { return prev_; }

 private:
  Builder b_;
  bool reduce_;
  std::vector<u32> prev_, recvs_, reduces_;
};

// Rings over `ring` (positions); `idx` is this member's place in it and
// sizes[c] the size of chunk c.

void ring_reduce_scatter(Steps& st, std::span<const u32> ring, u32 idx, std::span<const u64> sizes) {
  const u32 n = static_cast<u32>(ring.size());
  if (n < 2) return;
  const u32 next = ring[(idx + 1) % n], prev = ring[(idx + n - 1) % n];
  for (u32 k = 0; k + 1 < n; ++k) {
    st.send(next, sizes[mod_sub(idx, k, n)]);
    const u64 in = sizes[mod_sub(idx, k + 1, n)];
    st.recv(prev, in);
    st.reduce(in);
    st.end();
  }
  // Member idx now holds the reduced chunk (idx + 1) % n.
}

// Member idx starts with chunk (idx + own_offset) % n.
void ring_all_gather(Steps& st, std::span<const u32> ring, u32 idx, std::span<const u64> sizes, u32 own_offset) {
  const u32 n = static_cast<u32>(ring.size());
  if (n < 2) return;
  const u32 next = ring[(idx + 1) % n], prev = ring[(idx + n - 1) % n];
  const u32 own = (idx + own_offset) % n;
  for (u32 k = 0; k + 1 < n; ++k) {
    st.send(next, sizes[mod_sub(own, k, n)]);
    st.recv(prev, sizes[mod_sub(own, k + 1, n)]);
    st.end();
  }
}

std::vector<u32> iota(u32 n) {
  std::vector<u32> v(n);
  for (u32 i = 0; i < n; ++i) v[i] = i;
  return v;
}

std::vector<u64> chunks_of(u64 total, u32 parts) {
  std::vector<u64> v(parts);
  for (u32 c = 0; c < parts; ++c) v[c] = chunk(total, parts, c);
  return v;
}


void ring_algorithm(const CollSpec& s, u32 me, Steps& st) {
  const u32 p = s.size;
  const auto ring = iota(p);
  const auto sizes = chunks_of(s.bytes, p);
  switch (s.type) {
    case CollType::kAllReduce:
    case CollType::kBarrier:
      ring_reduce_scatter(st, ring, me, sizes);
      ring_all_gather(st, ring, me, sizes, 1);
      break;
    case CollType::kReduceScatter:
    case CollType::kReduceScatterBlock:
      ring_reduce_scatter(st, ring, me, sizes);
      break;
    case CollType::kAllGather:
      ring_all_gather(st, ring, me, sizes, 0);
      break;
    case CollType::kAllToAll:  // pairwise exchange, one peer per step
      for (u32 k = 1; k < p; ++k) {
        const u32 to = (me + k) % p;
        st.send(to, s.all_to_all_per_peer ? s.bytes : chunk(s.bytes, p, to));
        st.recv(mod_sub(me, k, p), s.all_to_all_per_peer ? s.bytes : chunk(s.bytes, p, me));
        st.end();
      }
      break;
    case CollType::kBroadcast: {  // a pipelined chain from the root
      const u32 c = std::max<u32>(1, s.chunks), v = mod_sub(me, s.root, p);
      for (u32 step = 0; step + 2 < c + p; ++step) {
        if (v >= 1 && step >= v - 1 && step - (v - 1) < c) st.recv((s.root + v - 1) % p, chunk(s.bytes, c, step - (v - 1)));
        if (v + 1 < p && step >= v && step - v < c) st.send((s.root + v + 1) % p, chunk(s.bytes, c, step - v));
        st.end();
      }
      break;
    }
    case CollType::kReduce: {  // a pipelined chain towards the root
      const u32 c = std::max<u32>(1, s.chunks), u = mod_sub(me, s.root, p);
      for (u32 step = 0; step + 2 < c + p; ++step) {
        if (u + 1 < p && step >= p - 2 - u && step - (p - 2 - u) < c) {
          const u64 in = chunk(s.bytes, c, step - (p - 2 - u));
          st.recv((s.root + u + 1) % p, in);
          st.reduce(in);
        }
        if (u >= 1 && step >= p - 1 - u && step - (p - 1 - u) < c)
          st.send((s.root + u - 1) % p, chunk(s.bytes, c, step - (p - 1 - u)));
        st.end();
      }
      break;
    }
    default:
      throw std::logic_error("ring: unsupported collective");
  }
}

// Members past the largest power of two hand their buffer to a partner and get
// the result back; `core` runs on the others.
template <typename Core>
void with_fold(const CollSpec& s, u32 me, Steps& st, Core&& core) {
  const u32 p2 = floor_pow2(s.size), rem = s.size - p2;
  if (me >= p2) {
    st.send(me - p2, s.bytes);
    st.end();
    st.recv(me - p2, s.bytes);
    st.end();
    return;
  }
  if (me < rem) {
    st.recv(me + p2, s.bytes);
    st.reduce(s.bytes);
    st.end();
  }
  core(p2);
  if (me < rem) {
    st.send(me + p2, s.bytes);
    st.end();
  }
}

void recursive_doubling(const CollSpec& s, u32 me, Steps& st) {
  const u32 p = s.size;
  switch (s.type) {
    case CollType::kAllReduce:
      with_fold(s, me, st, [&](u32 p2) {
        for (u32 d = 1; d < p2; d <<= 1) {
          st.send(me ^ d, s.bytes);
          st.recv(me ^ d, s.bytes);
          st.reduce(s.bytes);
          st.end();
        }
      });
      break;
    case CollType::kAllGather: {  // power-of-two groups only
      const auto sizes = chunks_of(s.bytes, p);
      std::vector<u64> prefix(p + 1, 0);
      for (u32 c = 0; c < p; ++c) prefix[c + 1] = prefix[c] + sizes[c];
      auto block = [&](u32 who, u32 width) {
        const u32 base = who / width * width;
        return prefix[base + width] - prefix[base];
      };
      for (u32 d = 1; d < p; d <<= 1) {
        st.send(me ^ d, block(me, d));
        st.recv(me ^ d, block(me ^ d, d));
        st.end();
      }
      break;
    }
    case CollType::kBarrier:  // dissemination
      for (u32 d = 1; d < p; d <<= 1) {
        st.send((me + d) % p, 0);
        st.recv(mod_sub(me, d, p), 0);
        st.end();
      }
      break;
    default:
      throw std::logic_error("recursive_doubling: unsupported collective");
  }
}

// Recursive halving, then for all-reduce recursive doubling, on p2 = 2^k members.
void halving_doubling_core(const CollSpec& s, u32 me, u32 p2, Steps& st, bool gather_back) {
  const auto sizes = chunks_of(s.bytes, p2);
  std::vector<u64> prefix(p2 + 1, 0);
  for (u32 c = 0; c < p2; ++c) prefix[c + 1] = prefix[c] + sizes[c];
  auto sum = [&](u32 lo, u32 hi) { return prefix[hi] - prefix[lo]; };
  u32 lo = 0, hi = p2;
  std::vector<std::pair<u32, u32>> parents;
  for (u32 d = p2 / 2; d >= 1; d /= 2) {
    const u32 mid = lo + (hi - lo) / 2;
    const bool low = (me & d) == 0;
    parents.emplace_back(lo, hi);
    const u64 out = low ? sum(mid, hi) : sum(lo, mid);
    const u64 kept = low ? sum(lo, mid) : sum(mid, hi);
    st.send(me ^ d, out);
    st.recv(me ^ d, kept);
    st.reduce(kept);
    st.end();
    if (low) hi = mid;
    else lo = mid;
  }
  if (!gather_back) return;
  u32 d = 1;
  for (auto it = parents.rbegin(); it != parents.rend(); ++it, d <<= 1) {
    const auto [plo, phi] = *it;
    const u64 mine = sum(lo, hi), whole = sum(plo, phi);
    st.send(me ^ d, mine);
    st.recv(me ^ d, whole - mine);
    st.end();
    lo = plo;
    hi = phi;
  }
}

void halving_doubling(const CollSpec& s, u32 me, Steps& st) {
  if (s.type == CollType::kAllReduce) {
    with_fold(s, me, st, [&](u32 p2) { halving_doubling_core(s, me, p2, st, true); });
  } else {  // reduce-scatter, power-of-two groups only
    halving_doubling_core(s, me, s.size, st, false);
  }
}

void binomial_bcast(const CollSpec& s, u32 me, u32 root, u64 bytes, Steps& st) {
  const u32 p = s.size, v = mod_sub(me, root, p);
  for (u32 r = 0; r < ceil_log2(p); ++r) {
    const u32 bit = 1u << r;
    if (v < bit && v + bit < p) st.send((v + bit + root) % p, bytes);
    if (v >= bit && v < 2 * bit) st.recv((v - bit + root) % p, bytes);
    st.end();
  }
}

void binomial_reduce(const CollSpec& s, u32 me, u32 root, u64 bytes, Steps& st) {
  const u32 p = s.size, v = mod_sub(me, root, p);
  for (u32 mask = 1; mask < p; mask <<= 1) {
    if (v & mask) {
      st.send((v - mask + root) % p, bytes);
      st.end();
      break;
    }
    if (v + mask < p) {
      st.recv((v + mask + root) % p, bytes);
      st.reduce(bytes);
      st.end();
    }
  }
}

void binomial_tree(const CollSpec& s, u32 me, Steps& st) {
  const u32 p = s.size, root = s.root % p, v = mod_sub(me, root, p);
  // Bytes of the chunks of virtual ranks [lo, hi).
  auto subtree = [&](u32 lo, u32 hi) {
    u64 total = 0;
    for (u32 u = lo; u < std::min(hi, p); ++u) total += chunk(s.bytes, p, (u + root) % p);
    return total;
  };
  switch (s.type) {
    case CollType::kBroadcast:
      binomial_bcast(s, me, root, s.bytes, st);
      break;
    case CollType::kReduce:
      binomial_reduce(s, me, root, s.bytes, st);
      break;
    case CollType::kAllReduce:
    case CollType::kBarrier:
      binomial_reduce(s, me, 0, s.bytes, st);
      binomial_bcast(s, me, 0, s.bytes, st);
      break;
    case CollType::kGather:
      for (u32 mask = 1; mask < p; mask <<= 1) {
        if (v & mask) {
          st.send((v - mask + root) % p, subtree(v, v + mask));
          st.end();
          break;
        }
        if (v + mask < p) {
          st.recv((v + mask + root) % p, subtree(v + mask, v + 2 * mask));
          st.end();
        }
      }
      break;
    case CollType::kScatter:
      for (u32 r = ceil_log2(p); r-- > 0;) {
        const u32 mask = 1u << r;
        if (v % (2 * mask) == 0 && v + mask < p) st.send((v + mask + root) % p, subtree(v + mask, v + 2 * mask));
        else if (v % (2 * mask) == mask) st.recv((v - mask + root) % p, subtree(v, v + mask));
        st.end();
      }
      break;
    default:
      throw std::logic_error("binomial_tree: unsupported collective");
  }
}

// NCCL's binary tree (ncclGetBtree): parent and two children, -1 for none.
struct TreeLinks {
  int up = -1, down0 = -1, down1 = -1;
};
TreeLinks btree(int n, int rank) {
  int bit = 1;
  while (bit < n && !(bit & rank)) bit <<= 1;
  TreeLinks t;
  if (rank == 0) {
    t.down1 = n > 1 ? bit >> 1 : -1;
    return t;
  }
  int up = (rank ^ bit) | (bit << 1);
  if (up >= n) up = rank ^ bit;
  t.up = up;
  int lowbit = bit >> 1;
  t.down0 = lowbit == 0 ? -1 : rank - lowbit;
  int down1 = lowbit == 0 ? -1 : rank + lowbit;
  while (down1 >= n) {
    lowbit >>= 1;
    down1 = lowbit == 0 ? -1 : rank + lowbit;
  }
  t.down1 = down1;
  return t;
}
// ncclGetDtree's second tree: mirrored for an even size, shifted for an odd one.
TreeLinks second_tree(int n, int rank) {
  const bool even = n % 2 == 0;
  const int shifted = even ? n - 1 - rank : (rank - 1 + n) % n;
  TreeLinks t = btree(n, shifted);
  auto back = [&](int x) { return x < 0 ? -1 : even ? n - 1 - x : (x + 1) % n; };
  return {back(t.up), back(t.down0), back(t.down1)};
}

void double_binary_tree(const CollSpec& s, u32 me, Schedule& out) {
  Builder b(out);
  const int n = static_cast<int>(s.size);
  const u32 c = std::max<u32>(1, s.chunks);
  const std::array<TreeLinks, 2> trees = {btree(n, static_cast<int>(me)), second_tree(n, static_cast<int>(me))};
  const std::array<u64, 2> share = {s.bytes - s.bytes / 2, s.bytes / 2};
  for (int t = 0; t < 2; ++t) {
    const TreeLinks& tl = trees[t];
    std::vector<int> children;
    for (int child : {tl.down0, tl.down1})
      if (child >= 0) children.push_back(child);
    for (u32 k = 0; k < c; ++k) {
      const u64 size = chunk(share[t], c, k);
      std::vector<u32> got;
      for (int child : children) got.push_back(b.add(ActionKind::kRecv, static_cast<u32>(child), size, {}));
      if (s.reduce && !got.empty() && size > 0) {
        const u32 red = b.add(ActionKind::kReduce, 0, size * got.size(), got);
        got.assign(1, red);
      }
      std::vector<u32> down_deps = got;
      if (tl.up >= 0) {
        b.add(ActionKind::kSend, static_cast<u32>(tl.up), size, got);
        down_deps.assign(1, b.add(ActionKind::kRecv, static_cast<u32>(tl.up), size, {}));
      }
      for (int child : children) b.add(ActionKind::kSend, static_cast<u32>(child), size, down_deps);
      out.outputs.insert(out.outputs.end(), down_deps.begin(), down_deps.end());
    }
  }
}

void direct(const CollSpec& s, u32 me, Steps& st) {
  const u32 p = s.size, root = s.root % p;
  const u64 b = s.bytes;
  auto each_peer = [&](auto&& f) {
    for (u32 k = 1; k < p; ++k) f((me + k) % p, mod_sub(me, k, p));
  };
  switch (s.type) {
    case CollType::kAllReduce:  // one shot: everyone sends the whole buffer to everyone
      each_peer([&](u32 to, u32 from) {
        st.send(to, b);
        st.recv(from, b);
      });
      st.reduce(b * (p - 1));
      break;
    case CollType::kReduceScatter:
    case CollType::kReduceScatterBlock:
      each_peer([&](u32 to, u32 from) {
        st.send(to, chunk(b, p, to));
        st.recv(from, chunk(b, p, me));
      });
      st.reduce(chunk(b, p, me) * (p - 1));
      break;
    case CollType::kAllGather:
      each_peer([&](u32 to, u32 from) {
        st.send(to, chunk(b, p, me));
        st.recv(from, chunk(b, p, from));
      });
      break;
    case CollType::kAllToAll:
      each_peer([&](u32 to, u32 from) {
        st.send(to, s.all_to_all_per_peer ? b : chunk(b, p, to));
        st.recv(from, s.all_to_all_per_peer ? b : chunk(b, p, me));
      });
      break;
    case CollType::kBarrier:
      each_peer([&](u32 to, u32 from) {
        st.send(to, 0);
        st.recv(from, 0);
      });
      break;
    case CollType::kBroadcast:
      if (me == root) each_peer([&](u32 to, u32) { st.send(to, b); });
      else st.recv(root, b);
      break;
    case CollType::kScatter:
      if (me == root) each_peer([&](u32 to, u32) { st.send(to, chunk(b, p, to)); });
      else st.recv(root, chunk(b, p, me));
      break;
    case CollType::kReduce:
      if (me == root) {
        each_peer([&](u32, u32 from) { st.recv(from, b); });
        st.reduce(b * (p - 1));
      } else {
        st.send(root, b);
      }
      break;
    case CollType::kGather:
      if (me == root) each_peer([&](u32, u32 from) { st.recv(from, chunk(b, p, from)); });
      else st.send(root, chunk(b, p, me));
      break;
  }
  st.end();
}

void hierarchical(const CollSpec& s, u32 me, Steps& st) {
  const u32 p = s.size, m = s.node_size, nodes = p / m;
  const u32 node = me / m, local = me % m;
  std::vector<u32> intra(m), inter(nodes);
  for (u32 l = 0; l < m; ++l) intra[l] = node * m + l;
  for (u32 j = 0; j < nodes; ++j) inter[j] = j * m + local;
  // Bytes of the chunks of positions {j m + l}, the block of local index l.
  auto block = [&](u32 l) {
    u64 total = 0;
    for (u32 j = 0; j < nodes; ++j) total += chunk(s.bytes, p, j * m + l);
    return total;
  };
  switch (s.type) {
    case CollType::kAllReduce: {
      const auto intra_sizes = chunks_of(s.bytes, m);
      ring_reduce_scatter(st, intra, local, intra_sizes);
      const auto inter_sizes = chunks_of(intra_sizes[(local + 1) % m], nodes);
      ring_reduce_scatter(st, inter, node, inter_sizes);
      ring_all_gather(st, inter, node, inter_sizes, 1);
      ring_all_gather(st, intra, local, intra_sizes, 1);
      break;
    }
    case CollType::kAllGather: {
      std::vector<u64> inter_sizes(nodes), intra_sizes(m);
      for (u32 j = 0; j < nodes; ++j) inter_sizes[j] = chunk(s.bytes, p, j * m + local);
      for (u32 l = 0; l < m; ++l) intra_sizes[l] = block(l);
      ring_all_gather(st, inter, node, inter_sizes, 0);
      ring_all_gather(st, intra, local, intra_sizes, 0);
      break;
    }
    case CollType::kReduceScatter:
    case CollType::kReduceScatterBlock: {
      std::vector<u64> intra_sizes(m), inter_sizes(nodes);
      for (u32 l = 0; l < m; ++l) intra_sizes[l] = block(l);
      ring_reduce_scatter(st, intra, local, intra_sizes);
      const u32 owned = (local + 1) % m;
      for (u32 j = 0; j < nodes; ++j) inter_sizes[j] = chunk(s.bytes, p, j * m + owned);
      ring_reduce_scatter(st, inter, node, inter_sizes);
      break;
    }
    default:
      throw std::logic_error("hierarchical: unsupported collective");
  }
}

void number_messages(Schedule& s) {
  std::unordered_map<u32, u32> sent, received;
  for (Action& a : s.actions) {
    if (a.kind == ActionKind::kSend) a.seq = sent[a.peer]++;
    else if (a.kind == ActionKind::kRecv) a.seq = received[a.peer]++;
  }
}

std::uint64_t message_key(u32 src, u32 dst, u32 seq, u32 p) {
  return ((static_cast<u64>(src) * p + dst) << 20) | seq;
}

}  // namespace

Algorithm parse_algorithm(std::string_view name) {
  for (int i = 0; i < kNumAlgorithms; ++i)
    if (name == kAlgorithmNames[i]) return static_cast<Algorithm>(i);
  throw std::invalid_argument("unknown collective algorithm '" + std::string(name) + "'");
}

std::vector<std::uint32_t> Schedule::sinks() const {
  std::vector<bool> needed(actions.size(), false);
  for (u32 d : deps) needed[d] = true;
  std::vector<u32> out;
  for (u32 a = 0; a < actions.size(); ++a)
    if (!needed[a]) out.push_back(a);
  return out;
}

bool supports(CollType type, Algorithm algorithm, std::uint32_t size, std::uint32_t node_size) {
  if (size <= 1) return true;
  using C = CollType;
  switch (algorithm) {
    case Algorithm::kRing:
      return type == C::kAllReduce || type == C::kReduceScatter || type == C::kReduceScatterBlock ||
             type == C::kAllGather || type == C::kBroadcast || type == C::kReduce || type == C::kAllToAll ||
             type == C::kBarrier;
    case Algorithm::kRecursiveDoubling:
      return type == C::kAllReduce || type == C::kBarrier || (type == C::kAllGather && is_pow2(size));
    case Algorithm::kHalvingDoubling:
      return type == C::kAllReduce ||
             ((type == C::kReduceScatter || type == C::kReduceScatterBlock) && is_pow2(size));
    case Algorithm::kBinomialTree:
      return type == C::kBroadcast || type == C::kReduce || type == C::kGather || type == C::kScatter ||
             type == C::kAllReduce || type == C::kBarrier;
    case Algorithm::kDoubleBinaryTree:
      return type == C::kAllReduce;
    case Algorithm::kDirect:
      return true;
    case Algorithm::kHierarchical:
      return (type == C::kAllReduce || type == C::kAllGather || type == C::kReduceScatter ||
              type == C::kReduceScatterBlock) &&
             node_size >= 1 && size % node_size == 0;
  }
  return false;
}

static Algorithm default_algorithm(CollType type) {
  switch (type) {
    case CollType::kAllReduce:
    case CollType::kAllGather:
    case CollType::kReduceScatter:
    case CollType::kReduceScatterBlock:
      return Algorithm::kRing;
    case CollType::kAllToAll:
      return Algorithm::kDirect;
    case CollType::kBarrier:
      return Algorithm::kRecursiveDoubling;
    default:
      return Algorithm::kBinomialTree;
  }
}

Algorithm resolve_algorithm(CollType type, Algorithm wanted, std::uint32_t size, std::uint32_t node_size,
                            bool* fell_back) {
  const bool ok = supports(type, wanted, size, node_size);
  if (fell_back) *fell_back = !ok;
  return ok ? wanted : default_algorithm(type);
}

void build_schedule(const CollSpec& spec, std::uint32_t me, Schedule& out) {
  out.actions.clear();
  out.deps.clear();
  out.outputs.clear();
  if (me >= spec.size) throw std::invalid_argument("collective member out of range");
  if (spec.size <= 1) return;
  if (!supports(spec.type, spec.algorithm, spec.size, spec.node_size))
    throw std::invalid_argument(std::string(kAlgorithmNames[static_cast<std::size_t>(spec.algorithm)]) +
                                " does not support " + kCollTypeNames[static_cast<std::size_t>(spec.type)] + " on " +
                                std::to_string(spec.size) + " members");
  if (spec.algorithm == Algorithm::kDoubleBinaryTree) {
    double_binary_tree(spec, me, out);
  } else {
    Steps st(out, spec.reduce);
    switch (spec.algorithm) {
      case Algorithm::kRing: ring_algorithm(spec, me, st); break;
      case Algorithm::kRecursiveDoubling: recursive_doubling(spec, me, st); break;
      case Algorithm::kHalvingDoubling: halving_doubling(spec, me, st); break;
      case Algorithm::kBinomialTree: binomial_tree(spec, me, st); break;
      case Algorithm::kDirect: direct(spec, me, st); break;
      case Algorithm::kHierarchical: hierarchical(spec, me, st); break;
      case Algorithm::kDoubleBinaryTree: break;
    }
    out.outputs = st.last();
  }
  number_messages(out);
}

std::vector<Time> evaluate_collective(const CollSpec& spec, std::span<const std::uint32_t> members,
                                      const CostModel& cost, bool data_completion, std::uint64_t exact_limit) {
  const u32 p = spec.size;
  std::vector<Time> finish(p, 0);
  if (p <= 1) return finish;
  if (members.size() != p) throw std::invalid_argument("collective group size does not match its members");

  auto finish_of = [&](const Schedule& s, std::span<const Time> done) {
    Time t = 0;
    if (data_completion) {
      for (u32 a : s.outputs) t = std::max(t, done[a]);
    } else {
      for (Time d : done) t = std::max(t, d);
    }
    return t;
  };

  Schedule first;
  build_schedule(spec, 0, first);
  if (static_cast<u64>(first.actions.size()) * p > exact_limit) {
    std::vector<Time> done(first.actions.size(), 0);
    for (u32 a = 0; a < first.actions.size(); ++a) {
      Time ready = 0;
      for (u32 d : first.deps_of(a)) ready = std::max(ready, done[d]);
      const Action& act = first.actions[a];
      const u32 peer = members[act.peer];
      switch (act.kind) {
        case ActionKind::kSend: done[a] = ready + cost.send_busy(members[0], peer, act.bytes); break;
        case ActionKind::kRecv: done[a] = ready + cost.deliver(peer, members[0], act.bytes); break;
        case ActionKind::kReduce: done[a] = ready + cost.reduce(act.bytes); break;
      }
    }
    std::fill(finish.begin(), finish.end(), finish_of(first, done));
    return finish;
  }

  std::vector<Schedule> all(p);
  all[0] = std::move(first);
  for (u32 m = 1; m < p; ++m) build_schedule(spec, m, all[m]);
  std::vector<u64> base(p + 1, 0);
  for (u32 m = 0; m < p; ++m) base[m + 1] = base[m] + all[m].actions.size();
  const u64 total = base[p];
  std::vector<Time> done(total, kNever);
  std::vector<u32> pending(total, 0);
  std::vector<std::vector<u32>> dependents(total);
  std::vector<std::pair<u32, u32>> stack;  // (member, action)
  for (u32 m = 0; m < p; ++m) {
    for (u32 a = 0; a < all[m].actions.size(); ++a) {
      const auto deps = all[m].deps_of(a);
      pending[base[m] + a] = static_cast<u32>(deps.size());
      for (u32 d : deps) dependents[base[m] + d].push_back(a);
      if (deps.empty()) stack.emplace_back(m, a);
    }
  }
  std::unordered_map<u64, Time> arrivals;                      // message -> when the receiver has it
  std::unordered_map<u64, std::pair<u32, Time>> waiting;      // message -> (recv action, post time)

  auto complete = [&](u32 m, u32 a, Time t) {
    done[base[m] + a] = t;
    for (u32 next : dependents[base[m] + a])
      if (--pending[base[m] + next] == 0) stack.emplace_back(m, next);
  };
  while (!stack.empty()) {
    const auto [m, a] = stack.back();
    stack.pop_back();
    const Schedule& s = all[m];
    Time ready = 0;
    for (u32 d : s.deps_of(a)) ready = std::max(ready, done[base[m] + d]);
    const Action& act = s.actions[a];
    switch (act.kind) {
      case ActionKind::kSend: {
        const Time arrival = ready + cost.deliver(members[m], members[act.peer], act.bytes);
        const u64 key = message_key(m, act.peer, act.seq, p);
        if (auto it = waiting.find(key); it != waiting.end()) {
          const auto [recv, post] = it->second;
          waiting.erase(it);
          complete(act.peer, recv, std::max(post, arrival));
        } else {
          arrivals[key] = arrival;
        }
        complete(m, a, ready + cost.send_busy(members[m], members[act.peer], act.bytes));
        break;
      }
      case ActionKind::kRecv: {
        const u64 key = message_key(act.peer, m, act.seq, p);
        if (auto it = arrivals.find(key); it != arrivals.end()) {
          const Time arrival = it->second;
          arrivals.erase(it);
          complete(m, a, std::max(ready, arrival));
        } else {
          waiting[key] = {a, ready};
        }
        break;
      }
      case ActionKind::kReduce:
        complete(m, a, ready + cost.reduce(act.bytes));
        break;
    }
  }
  for (u64 i = 0; i < total; ++i)
    if (done[i] == kNever) throw std::logic_error("collective schedule has unmatched messages");
  for (u32 m = 0; m < p; ++m)
    finish[m] = finish_of(all[m], std::span<const Time>(done).subspan(base[m], all[m].actions.size()));
  return finish;
}

}  // namespace cta::sim
