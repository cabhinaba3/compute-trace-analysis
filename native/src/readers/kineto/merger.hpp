#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/interner.hpp"
#include "common/launch_map.hpp"
#include "common/util.hpp"
#include "readers/kineto/chunk.hpp"
#include "readers/kineto/kineto_read.hpp"

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
  static std::string encode(const Tag& tag, const Interner& tags);

  KinetoColumns out_;
  Interner names_;
  std::unordered_map<std::string, int32_t> thread_ids_;
  std::vector<int32_t> runtime_slot_;
  std::vector<int64_t> gpu_corr_;
  LaunchMap launches_;
};

}  // namespace cta::kineto_reader
