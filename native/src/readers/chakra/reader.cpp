#include "readers/chakra/reader.hpp"

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

#include "util/cpus.hpp"
#include "util/util.hpp"
#include "util/interner.hpp"
#include "util/ordered_pool.hpp"
#include "et_def.pb.h"

namespace cta::readers {
namespace {

namespace pb = ChakraProtoMsg;
using Wide = __int128;  // wide enough for every integer kind an attribute can hold

// NodeType numbers (stable protobuf wire values) to the codes in ChakraColumns.
std::uint8_t op_class_of(int node_type) {
  switch (node_type) {
    case pb::MEM_LOAD_NODE:
    case pb::MEM_STORE_NODE:
      return 1;
    case pb::COMP_NODE:
      return 2;
    case pb::COMM_SEND_NODE:
      return 3;
    case pb::COMM_RECV_NODE:
      return 4;
    case pb::COMM_COLL_NODE:
      return 5;
    case pb::STORAGE_NODE:
      return 6;
    case pb::DATA_OP_NODE:
      return 7;
    default:
      return 0;
  }
}

constexpr std::uint8_t kSend = 3, kRecv = 4;

Wide text_as_int(const std::string& s) {
  if (s.empty() || s.size() > 30) return 0;
  char* end = nullptr;
  const long long v = std::strtoll(s.c_str(), &end, 10);
  return (end && *end == '\0') ? v : 0;
}

// An attribute's value as an integer: integers and bools as they are, floats
// truncated, text parsed when it is a number. Lists and unset values give 0.
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
  unsigned __int128 mag = negative ? static_cast<unsigned __int128>(-(v + 1)) + 1 : static_cast<unsigned __int128>(v);
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

// Bytes moved by a storage node: the file-system size, or sectors / pages times
// their size, as the node's storage detail says.
std::uint64_t storage_bytes(const pb::Node& node) {
  const pb::StorageInfo& storage = node.storage();
  switch (storage.detail_case()) {
    case pb::StorageInfo::kFs: return storage.fs().size_bytes();
    case pb::StorageInfo::kBlock: return static_cast<std::uint64_t>(storage.block().nr_sectors()) * 512;
    case pb::StorageInfo::kPage: return static_cast<std::uint64_t>(storage.page().nr_pages()) * 4096;
    default: return 0;
  }
}

// One file's rows, with strings numbered locally; renumbered when merged.
struct FileRows {
  Interner names, comm_types;
  std::vector<std::uint64_t> node_id;
  std::vector<std::int32_t> name;
  std::vector<std::uint8_t> op_class;
  std::vector<std::int32_t> comm_type;
  std::vector<std::uint64_t> duration_ns, size_bytes, num_ops;
  std::vector<std::int32_t> src_rank, dst_rank;
  std::vector<std::uint32_t> message_tag;
  std::vector<std::uint64_t> edge_src, edge_dst;
  std::vector<std::uint8_t> edge_kind;
  std::uint32_t rank = 0;
};

// The attributes the readers use; a later attribute with the same name wins.
struct Attrs {
  bool comm_type_set = false, comm_type_is_int = false, comm_type_is_text = false;
  Wide comm_type_int = 0;
  std::string comm_type_text;
  bool has_comm_size = false, has_tensor_size = false, has_src = false, has_dst = false;
  Wide comm_size = 0, tensor_size = 0, src = 0, dst = 0, tag = 0, num_ops = 0;
};

Attrs read_attrs(const pb::Node& node) {
  Attrs out;
  for (const pb::AttributeProto& a : node.attr()) {
    const std::string& name = a.name();
    if (name == "comm_type") {
      out.comm_type_set = true;
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
    }
  }
  return out;
}

// The comm_type column: an integer names a collective ("COLL_<n>" if the
// number is unknown), text is upper-cased, and a send or receive without the
// attribute is "SEND" or "RECV".
std::string resolve_comm_type(const Attrs& at, std::uint8_t op_class) {
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
  if (op_class == kSend) return "SEND";
  if (op_class == kRecv) return "RECV";
  return "";
}

void add_node(const pb::Node& node, FileRows& rows) {
  const Attrs at = read_attrs(node);
  const std::uint8_t op_class = op_class_of(static_cast<int>(node.type()));

  Wide size = at.has_comm_size ? at.comm_size : (at.has_tensor_size ? at.tensor_size : 0);
  std::uint64_t size_bytes = clamp_u64(size);
  if (node.type() == pb::STORAGE_NODE && size_bytes == 0) size_bytes = storage_bytes(node);

  const Wide default_src = op_class == kSend ? static_cast<Wide>(rows.rank) : Wide{-1};
  const Wide default_dst = op_class == kRecv ? static_cast<Wide>(rows.rank) : Wide{-1};

  rows.node_id.push_back(node.id());
  rows.name.push_back(rows.names.intern(node.name()));
  rows.op_class.push_back(op_class);
  rows.comm_type.push_back(rows.comm_types.intern(resolve_comm_type(at, op_class)));
  rows.duration_ns.push_back(node.duration_micros() * 1000);
  rows.size_bytes.push_back(size_bytes);
  rows.src_rank.push_back(clamp_i32(at.has_src ? at.src : default_src));
  rows.dst_rank.push_back(clamp_i32(at.has_dst ? at.dst : default_dst));
  rows.message_tag.push_back(clamp_u32(at.tag));
  rows.num_ops.push_back(clamp_u64(at.num_ops));

  // Data dependencies come before control dependencies, as in the files.
  for (std::uint64_t dep : node.data_deps()) {
    rows.edge_src.push_back(dep);
    rows.edge_dst.push_back(node.id());
    rows.edge_kind.push_back(0);
  }
  for (std::uint64_t dep : node.ctrl_deps()) {
    rows.edge_src.push_back(dep);
    rows.edge_dst.push_back(node.id());
    rows.edge_kind.push_back(1);
  }
}

// Reads one .et file: a length-prefixed GlobalMetadata message, then
// length-prefixed Node messages. An empty file has no rows.
std::unique_ptr<FileRows> read_file(const std::string& path, std::uint32_t rank) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  google::protobuf::io::FileInputStream stream(fd, 1 << 20);
  stream.SetCloseOnDelete(true);

  auto result = std::make_unique<FileRows>();
  FileRows& rows = *result;
  rows.rank = rank;
  auto corrupt = [&](const char* what) { return std::invalid_argument(path + ": " + what); };

  bool clean_eof = false;
  pb::GlobalMetadata metadata;
  if (!google::protobuf::util::ParseDelimitedFromZeroCopyStream(&metadata, &stream, &clean_eof)) {
    if (clean_eof) return result;
    throw corrupt("truncated or corrupt Chakra metadata");
  }
  pb::Node node;
  while (true) {
    node.Clear();  // parsing merges into the message
    if (!google::protobuf::util::ParseDelimitedFromZeroCopyStream(&node, &stream, &clean_eof)) {
      if (clean_eof) break;
      throw corrupt("truncated or corrupt Chakra node");
    }
    add_node(node, rows);
  }
  if (stream.GetErrno() != 0) throw std::runtime_error("read failed: " + path);
  return result;
}

// Appends files' rows to the columns in the order they arrive.
class Merger {
 public:
  void add(std::unique_ptr<FileRows> file) {
    FileRows& f = *file;
    std::vector<std::int32_t> name_map, type_map;
    for (const std::string& s : f.names.items()) name_map.push_back(names_.intern(s));
    for (const std::string& s : f.comm_types.items()) type_map.push_back(comm_types_.intern(s));

    const std::size_t n = f.node_id.size();
    out_.rank.insert(out_.rank.end(), n, f.rank);
    append_vector(out_.node_id, f.node_id);
    for (std::int32_t c : f.name) out_.name.push_back(name_map[static_cast<std::size_t>(c)]);
    append_vector(out_.op_class, f.op_class);
    for (std::int32_t c : f.comm_type) out_.comm_type.push_back(type_map[static_cast<std::size_t>(c)]);
    append_vector(out_.duration_ns, f.duration_ns);
    append_vector(out_.size_bytes, f.size_bytes);
    append_vector(out_.num_ops, f.num_ops);
    append_vector(out_.src_rank, f.src_rank);
    append_vector(out_.dst_rank, f.dst_rank);
    append_vector(out_.message_tag, f.message_tag);

    out_.edge_rank.insert(out_.edge_rank.end(), f.edge_src.size(), f.rank);
    append_vector(out_.edge_src, f.edge_src);
    append_vector(out_.edge_dst, f.edge_dst);
    append_vector(out_.edge_kind, f.edge_kind);
  }

  ChakraColumns finish() {
    out_.names.assign(names_.items().begin(), names_.items().end());
    out_.comm_types.assign(comm_types_.items().begin(), comm_types_.items().end());
    return std::move(out_);
  }

 private:
  ChakraColumns out_;
  Interner names_, comm_types_;
};

}  // namespace

ChakraColumns read_chakra(const std::vector<std::string>& paths,
                                const std::vector<std::uint32_t>& ranks, int threads) {
  if (paths.size() != ranks.size()) throw std::invalid_argument("paths and ranks must have the same length");
  if (threads < 0) throw std::invalid_argument("threads must be >= 0");
  if (threads == 0) threads = usable_cpus();
  threads = static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(static_cast<std::size_t>(threads), paths.size())));

  Merger merger;
  OrderedPool<std::size_t, std::unique_ptr<FileRows>> pool(
      threads,
      [&]() -> OrderedPool<std::size_t, std::unique_ptr<FileRows>>::Work {
        return [&](std::size_t& i) { return read_file(paths[i], ranks[i]); };
      },
      [&](std::unique_ptr<FileRows>&& rows) { merger.add(std::move(rows)); }, static_cast<std::size_t>(threads) + 2);
  for (std::size_t i = 0; i < paths.size(); ++i) pool.submit(std::size_t(i));
  pool.finish();
  return merger.finish();
}

}  // namespace cta::readers
