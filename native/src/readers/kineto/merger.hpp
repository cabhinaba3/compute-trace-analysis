#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "readers/kineto/chunk.hpp"
#include "common/interner.hpp"
#include "readers/kineto/kineto_read.hpp"
#include "readers/kineto/launch_map.hpp"

namespace cta::kineto_reader {

// Folds parsed chunks, in file order, into one set of columns. Strings and
// thread keys are renumbered here, so ids follow first appearance in the file.
class Merger {
 public:
  void add(ChunkOut& c);

  // Matches GPU activities to launches once every chunk is in, and hands over
  // the columns.
  KinetoColumns finish();

 private:
  template <typename T>
  static void append(std::vector<T>& to, const std::vector<T>& from) {
    to.insert(to.end(), from.begin(), from.end());
  }
  static std::string encode(const Tag& tag, const Interner& tags);

  KinetoColumns out_;
  Interner names_;
  std::unordered_map<std::string, int32_t> thread_ids_;
  std::vector<int32_t> runtime_slot_;
  std::vector<int64_t> gpu_corr_;
  LaunchMap launches_;
};

}  // namespace cta::kineto_reader
