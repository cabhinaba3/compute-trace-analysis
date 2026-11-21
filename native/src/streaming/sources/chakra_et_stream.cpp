#include "streaming/sources/source.hpp"

#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "et_def.pb.h"

namespace cta::streaming {
namespace {

namespace pb = ChakraProtoMsg;
using Wide = __int128;

OpClass op_class_of(int node_type) {
  switch (node_type) {
    case pb::MEM_LOAD_NODE:
    case pb::MEM_STORE_NODE:
      return OpClass::kMemory;
    case pb::COMP_NODE:
      return OpClass::kCompute;
    case pb::COMM_SEND_NODE:
      return OpClass::kCommSend;
    case pb::COMM_RECV_NODE:
      return OpClass::kCommRecv;
    case pb::COMM_COLL_NODE:
      return OpClass::kCommColl;
    case pb::STORAGE_NODE:
      return OpClass::kStorage;
    case pb::DATA_OP_NODE:
      return OpClass::kDataOp;
    default:
      return OpClass::kOther;
  }
}

Wide text_as_int(const std::string& s) {
  if (s.empty() || s.size() > 30) return 0;
  char* end = nullptr;
  const long long v = std::strtoll(s.c_str(), &end, 10);
  return (end && *end == '\0') ? v : 0;
}

Wide attr_as_int(const pb::AttributeProto& a) {
  switch (a.value_case()) {
    case pb::AttributeProto::kInt32Val: return a.int32_val();
    case pb::AttributeProto::kInt64Val: return a.int64_val();
    case pb::AttributeProto::kUint32Val: return a.uint32_val();
    case pb::AttributeProto::kUint64Val: return a.uint64_val();
    case pb::AttributeProto::kSint32Val: return a.sint32_val();
    case pb::AttributeProto::kSint64Val: return a.sint64_val();
    case pb::AttributeProto::kFixed32Val: return a.fixed32_val();
    case pb::AttributeProto::kFixed64Val: return a.fixed64_val();
    case pb::AttributeProto::kSfixed32Val: return a.sfixed32_val();
    case pb::AttributeProto::kSfixed64Val: return a.sfixed64_val();
    case pb::AttributeProto::kBoolVal: return a.bool_val() ? 1 : 0;
    case pb::AttributeProto::kDoubleVal: {
      const double d = a.double_val();
      return std::isfinite(d) && std::fabs(d) < 9.2e18 ? static_cast<Wide>(d) : 0;
    }
    case pb::AttributeProto::kFloatVal: {
      const double d = a.float_val();
      return std::isfinite(d) && std::fabs(d) < 9.2e18 ? static_cast<Wide>(d) : 0;
    }
    case pb::AttributeProto::kStringVal: return text_as_int(a.string_val());
    case pb::AttributeProto::kBytesVal: return text_as_int(a.bytes_val());
    default: return 0;
  }
}

bool attr_is_integer(const pb::AttributeProto& a) {
  switch (a.value_case()) {
    case pb::AttributeProto::kInt32Val:
    case pb::AttributeProto::kInt64Val:
    case pb::AttributeProto::kUint32Val:
    case pb::AttributeProto::kUint64Val:
    case pb::AttributeProto::kSint32Val:
    case pb::AttributeProto::kSint64Val:
    case pb::AttributeProto::kFixed32Val:
    case pb::AttributeProto::kFixed64Val:
    case pb::AttributeProto::kSfixed32Val:
    case pb::AttributeProto::kSfixed64Val:
    case pb::AttributeProto::kBoolVal:
      return true;
    default:
      return false;
  }
}

std::string wide_to_string(Wide v) {
  if (v == 0) return "0";
  const bool negative = v < 0;
  std::string digits;
  unsigned __int128 mag =
      negative ? static_cast<unsigned __int128>(-(v + 1)) + 1 : static_cast<unsigned __int128>(v);
  while (mag > 0) {
    digits.push_back(static_cast<char>('0' + static_cast<int>(mag % 10)));
    mag /= 10;
  }
  if (negative) digits.push_back('-');
  std::reverse(digits.begin(), digits.end());
  return digits;
}

std::uint64_t clamp_u64(Wide v) {
  if (v < 0) return 0;
  return v > static_cast<Wide>(std::numeric_limits<std::uint64_t>::max()) ? std::numeric_limits<std::uint64_t>::max()
                                                                         : static_cast<std::uint64_t>(v);
}

std::uint32_t clamp_u32(Wide v) {
  if (v < 0) return 0;
  return v > std::numeric_limits<std::uint32_t>::max() ? std::numeric_limits<std::uint32_t>::max()
                                                       : static_cast<std::uint32_t>(v);
}

std::int32_t clamp_i32(Wide v) {
  if (v < std::numeric_limits<std::int32_t>::min()) return std::numeric_limits<std::int32_t>::min();
  if (v > std::numeric_limits<std::int32_t>::max()) return std::numeric_limits<std::int32_t>::max();
  return static_cast<std::int32_t>(v);
}

std::uint64_t storage_bytes(const pb::Node& node) {
  const pb::StorageInfo& storage = node.storage();
  switch (storage.detail_case()) {
    case pb::StorageInfo::kFs: return storage.fs().size_bytes();
    case pb::StorageInfo::kBlock: return static_cast<std::uint64_t>(storage.block().nr_sectors()) * 512;
    case pb::StorageInfo::kPage: return static_cast<std::uint64_t>(storage.page().nr_pages()) * 4096;
    default: return 0;
  }
}

struct Attrs {
  bool comm_type_is_int = false, comm_type_is_text = false;
  Wide comm_type_int = 0;
  std::string comm_type_text;
  bool has_comm_size = false, has_tensor_size = false, has_src = false, has_dst = false;
  Wide comm_size = 0, tensor_size = 0, src = 0, dst = 0, tag = 0, num_ops = 0;
  std::string comm_group;
};

Attrs read_attrs(const pb::Node& node) {
  Attrs out;
  for (const pb::AttributeProto& a : node.attr()) {
    const std::string& name = a.name();
    if (name == "comm_type") {
      out.comm_type_is_int = attr_is_integer(a);
      out.comm_type_is_text = a.value_case() == pb::AttributeProto::kStringVal;
      out.comm_type_int = out.comm_type_is_int ? attr_as_int(a) : 0;
      out.comm_type_text = out.comm_type_is_text ? a.string_val() : std::string();
    } else if (name == "comm_size") {
      out.has_comm_size = true;
      out.comm_size = attr_as_int(a);
    } else if (name == "tensor_size") {
      out.has_tensor_size = true;
      out.tensor_size = attr_as_int(a);
    } else if (name == "comm_src") {
      out.has_src = true;
      out.src = attr_as_int(a);
    } else if (name == "comm_dst") {
      out.has_dst = true;
      out.dst = attr_as_int(a);
    } else if (name == "comm_tag") {
      out.tag = attr_as_int(a);
    } else if (name == "num_ops") {
      out.num_ops = attr_as_int(a);
    } else if (name == "pg_name" || name == "comm_group" || name == "process_group") {
      if (a.value_case() == pb::AttributeProto::kStringVal) {
        out.comm_group = a.string_val();
      } else if (attr_is_integer(a)) {
        out.comm_group = wide_to_string(attr_as_int(a));
      }
    }
  }
  return out;
}

std::string resolve_comm_type(const Attrs& at, OpClass oc) {
  if (at.comm_type_is_int) {
    const Wide v = at.comm_type_int;
    if (v >= std::numeric_limits<int>::min() && v <= std::numeric_limits<int>::max() &&
        pb::CollectiveCommType_IsValid(static_cast<int>(v))) {
      return std::string(pb::CollectiveCommType_Name(static_cast<pb::CollectiveCommType>(static_cast<int>(v))));
    }
    return "COLL_" + wide_to_string(v);
  }
  if (at.comm_type_is_text) {
    std::string up = at.comm_type_text;
    for (char& c : up)
      if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 32);
    return up;
  }
  if (oc == OpClass::kCommSend) return "SEND";
  if (oc == OpClass::kCommRecv) return "RECV";
  return "";
}

void append_chakra_node(const pb::Node& node, std::uint32_t rank, GraphChunk& out) {
  const Attrs at = read_attrs(node);
  const OpClass oc = op_class_of(static_cast<int>(node.type()));
  Wide size = at.has_comm_size ? at.comm_size : (at.has_tensor_size ? at.tensor_size : 0);
  std::uint64_t size_bytes = clamp_u64(size);
  if (node.type() == pb::STORAGE_NODE && size_bytes == 0) size_bytes = storage_bytes(node);

  const Wide default_src = oc == OpClass::kCommSend ? static_cast<Wide>(rank) : Wide{-1};
  const Wide default_dst = oc == OpClass::kCommRecv ? static_cast<Wide>(rank) : Wide{-1};
  const std::string comm_type = resolve_comm_type(at, oc);

  out.nodes.push(rank, node.id(), node.name(), oc, comm_type, node.duration_micros() * 1000, size_bytes,
                 clamp_u64(at.num_ops), clamp_i32(at.has_src ? at.src : default_src),
                 clamp_i32(at.has_dst ? at.dst : default_dst), clamp_u32(at.tag), at.comm_group);

  // Data dependencies come before control dependencies, matching native/src/readers/chakra/reader.cpp.
  for (std::uint64_t dep : node.data_deps()) out.edges.push(rank, dep, node.id(), EdgeKind::kData);
  for (std::uint64_t dep : node.ctrl_deps()) out.edges.push(rank, dep, node.id(), EdgeKind::kCtrl);
}

// Streaming source over one or more Chakra .et protobuf files (at most one
// file descriptor open at a time, so arbitrarily many ranks never hit EMFILE).
class ChakraEtSource final : public GraphSource {
 public:
  ChakraEtSource(std::vector<std::string> paths, std::vector<std::uint32_t> ranks)
      : paths_(std::move(paths)), ranks_(std::move(ranks)) {
    if (paths_.size() != ranks_.size()) throw std::invalid_argument("paths and ranks must have the same length");
    seekable_ = true;
    for (const auto& p : paths_) {
      if (!is_seekable_path(p)) seekable_ = false;
      est_bytes_ += regular_file_size(p);
    }
  }

  bool next_chunk(GraphChunk& out, std::size_t max_nodes) override {
    out.clear();
    if (max_nodes == 0) max_nodes = 1;

    while (out.nodes.size() < max_nodes) {
      if (!stream_) {
        if (!open_next_file()) break;
        continue;
      }
      bool clean_eof = false;
      node_.Clear();
      if (!google::protobuf::util::ParseDelimitedFromZeroCopyStream(&node_, stream_.get(), &clean_eof)) {
        if (clean_eof) {
          if (stream_->GetErrno() != 0) throw std::runtime_error("read failed: " + current_path_);
          stream_.reset();
          continue;
        }
        throw std::invalid_argument(current_path_ + ": truncated or corrupt Chakra node");
      }
      append_chakra_node(node_, current_rank_, out);
    }
    return !out.empty();
  }

  bool is_seekable() const override { return seekable_; }

  void reset() override {
    if (!seekable_) throw std::runtime_error("cannot reset unseekable Chakra .et stream");
    stream_.reset();
    file_idx_ = 0;
  }

  std::string format_name() const override { return "chakra_et"; }
  std::uint64_t estimated_bytes() const override { return est_bytes_; }

 private:
  bool open_next_file() {
    while (file_idx_ < paths_.size()) {
      const std::size_t i = file_idx_++;
      current_path_ = paths_[i];
      current_rank_ = ranks_[i];
      int fd = -1;
      if (current_path_ == "-") {
        fd = ::dup(STDIN_FILENO);
        if (fd < 0) throw std::runtime_error("cannot dup stdin for Chakra .et stream");
      } else {
        fd = ::open(current_path_.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + current_path_);
      }
      auto s = std::make_unique<google::protobuf::io::FileInputStream>(fd, 1 << 20);
      s->SetCloseOnDelete(true);

      bool clean_eof = false;
      pb::GlobalMetadata meta;
      if (!google::protobuf::util::ParseDelimitedFromZeroCopyStream(&meta, s.get(), &clean_eof)) {
        if (clean_eof) {
          if (s->GetErrno() != 0) throw std::runtime_error("read failed: " + current_path_);
          continue;  // empty file
        }
        throw std::invalid_argument(current_path_ + ": truncated or corrupt Chakra metadata");
      }
      stream_ = std::move(s);
      return true;
    }
    return false;
  }

  std::vector<std::string> paths_;
  std::vector<std::uint32_t> ranks_;
  bool seekable_ = true;
  std::uint64_t est_bytes_ = 0;
  std::size_t file_idx_ = 0;
  std::string current_path_;
  std::uint32_t current_rank_ = 0;
  std::unique_ptr<google::protobuf::io::FileInputStream> stream_;
  pb::Node node_;
};

}  // namespace

std::unique_ptr<GraphSource> open_chakra_et_stream(const std::vector<std::string>& paths,
                                                   const std::vector<std::uint32_t>& ranks) {
  return std::make_unique<ChakraEtSource>(paths, ranks);
}

}  // namespace cta::streaming
