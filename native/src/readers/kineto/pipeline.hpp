#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "readers/kineto/chunk.hpp"
#include "readers/kineto/kineto_read.hpp"
#include "readers/kineto/merger.hpp"

namespace cta::kineto_reader {

// Parses chunks on worker threads and merges them in order on one more thread.
// At most `workers + 4` chunks (queued, parsing or waiting to merge) exist at a
// time, which bounds memory however fast the splitter runs.
class Pipeline {
 public:
  explicit Pipeline(int workers);
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  // Hands over the next chunk; blocks while the pipeline is full. Rethrows the
  // error of a failed worker.
  void submit(std::string&& data);

  // Waits for every chunk to be parsed and merged, and returns the columns.
  KinetoColumns finish();

 private:
  void fail(std::exception_ptr e);
  void stop();
  void work();
  void merge();

  const std::size_t max_in_flight_;
  std::mutex m_;
  std::condition_variable work_, room_, result_;
  std::deque<Chunk> queue_;
  std::map<std::size_t, std::unique_ptr<ChunkOut>> done_;
  std::size_t next_submit_ = 0, next_merge_ = 0;
  bool closed_ = false, aborted_ = false;
  std::exception_ptr error_;
  std::vector<std::thread> workers_;
  std::thread merger_thread_;
  Merger merger_;
};

}  // namespace cta::kineto_reader
