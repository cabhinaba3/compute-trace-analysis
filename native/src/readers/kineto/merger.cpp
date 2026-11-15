#include "readers/kineto/merger.hpp"

#include <limits>

namespace cta::kineto_reader {

void Merger::add(ChunkOut& c) {
  std::vector<int32_t> name_map;
  name_map.reserve(c.names.size());
  for (const std::string& s : c.names.items()) name_map.push_back(names_.intern(s));

  std::vector<int32_t> thread_map;
  thread_map.reserve(c.threads.size());
  for (const ThreadKey& key : c.threads) {
    const std::string text = encode(key.pid, c.tags) + encode(key.tid, c.tags);
    auto it = thread_ids_.try_emplace(text, static_cast<int32_t>(thread_ids_.size())).first;
    thread_map.push_back(it->second);
  }

  const int64_t row_base = static_cast<int64_t>(out_.gpu_name.size());
  for (std::size_t j = 0; j < c.c_row.size(); ++j) {
    out_.comm_row.push_back(row_base + c.c_row[j]);
    out_.comm_op.push_back(name_map[static_cast<std::size_t>(c.c_op[j])]);
    out_.comm_dtype.push_back(name_map[static_cast<std::size_t>(c.c_dtype[j])]);
    out_.comm_group.push_back(name_map[static_cast<std::size_t>(c.c_group[j])]);
  }
  append_vector(out_.comm_in, c.c_in);
  append_vector(out_.comm_out, c.c_out);
  append_vector(out_.comm_size, c.c_size);
  for (int32_t n : c.g_name) out_.gpu_name.push_back(name_map[static_cast<std::size_t>(n)]);
  append_vector(out_.gpu_category, c.g_cat);
  append_vector(out_.gpu_start, c.g_start);
  append_vector(out_.gpu_end, c.g_end);
  append_vector(out_.gpu_bytes, c.g_bytes);
  append_vector(out_.gpu_stream, c.g_stream);
  append_vector(out_.gpu_device, c.g_device);
  append_vector(gpu_corr_, c.g_corr);

  append_vector(out_.op_start, c.o_start);
  append_vector(out_.op_end, c.o_end);
  for (int32_t n : c.o_name) out_.op_name.push_back(name_map[static_cast<std::size_t>(n)]);
  for (int32_t t : c.o_thread) out_.op_thread.push_back(thread_map[static_cast<std::size_t>(t)]);

  append_vector(out_.step_id, c.s_id);
  append_vector(out_.step_start, c.s_start);
  append_vector(out_.step_end, c.s_end);

  for (std::size_t j = 0; j < c.r_name.size(); ++j) {
    const std::size_t code = static_cast<std::size_t>(name_map[static_cast<std::size_t>(c.r_name[j])]);
    if (runtime_slot_.size() <= code) runtime_slot_.resize(names_.size(), -1);
    if (runtime_slot_[code] < 0) {
      runtime_slot_[code] = static_cast<int32_t>(out_.runtime_name.size());
      out_.runtime_name.push_back(static_cast<int32_t>(code));
      out_.runtime_duration.push_back(0);
      out_.runtime_calls.push_back(0);
    }
    const std::size_t slot = static_cast<std::size_t>(runtime_slot_[code]);
    out_.runtime_duration[slot] += c.r_dur[j];
    out_.runtime_calls[slot] += c.r_calls[j];
  }

  for (std::size_t j = 0; j < c.l_corr.size(); ++j)
    launches_.put(c.l_corr[j], c.l_start[j], thread_map[static_cast<std::size_t>(c.l_thread[j])]);
}

KinetoColumns Merger::finish() {
  // A kernel can come before its launch in the file, so launches are
  // matched only now that every chunk is in.
  const std::size_t n = gpu_corr_.size();
  out_.gpu_launch_ns.resize(n);
  out_.gpu_launch_thread.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    LaunchMap::Hit hit{};
    if (launches_.find(gpu_corr_[i], hit)) {
      out_.gpu_launch_ns[i] = static_cast<double>(hit.start);
      out_.gpu_launch_thread[i] = hit.thread;
    } else {
      out_.gpu_launch_ns[i] = std::numeric_limits<double>::quiet_NaN();
      out_.gpu_launch_thread[i] = -1;
    }
  }
  out_.names.assign(names_.items().begin(), names_.items().end());
  return std::move(out_);
}

std::string Merger::encode(const Tag& tag, const Interner& tags) {
  std::string s(1, static_cast<char>(tag.kind));
  if (tag.kind == 1) {
    s.append(reinterpret_cast<const char*>(&tag.v), sizeof tag.v);
  } else if (tag.kind == 2) {
    const std::string& text = tags.items()[static_cast<std::size_t>(tag.v)];
    const std::uint64_t len = text.size();
    s.append(reinterpret_cast<const char*>(&len), sizeof len);
    s.append(text);
  }
  return s;
}

}  // namespace cta::kineto_reader
