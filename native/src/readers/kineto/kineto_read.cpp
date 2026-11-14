#include "readers/kineto/kineto_read.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>

#include "common/cpus.hpp"
#include "readers/kineto/chunk_parser.hpp"
#include "readers/kineto/pipeline.hpp"
#include "readers/kineto/source.hpp"
#include "readers/kineto/splitter.hpp"

namespace cta {

KinetoColumns read_kineto_columns(const std::string& path, int threads, std::uint64_t chunk_bytes,
                                  std::uint64_t block_bytes) {
  if (threads < 0) throw std::invalid_argument("threads must be >= 0");
  if (block_bytes < 1) throw std::invalid_argument("block_bytes must be at least 1");
  if (chunk_bytes < 64) throw std::invalid_argument("chunk_bytes must be at least 64");
  if (threads == 0) threads = usable_cpus();

  auto source = kineto_reader::open_source(path);

  using Clock = std::chrono::steady_clock;
  auto seconds = [](Clock::duration d) { return std::chrono::duration<double>(d).count(); };
  double read_s = 0, split_s = 0, wait_s = 0;

  kineto_reader::Pipeline pipeline(threads);
  kineto_reader::Splitter splitter(static_cast<std::size_t>(chunk_bytes), [&](std::string&& chunk) {
    const auto t0 = Clock::now();
    pipeline.submit(std::move(chunk));
    wait_s += seconds(Clock::now() - t0);
  });
  std::vector<char> block(static_cast<std::size_t>(std::min<std::uint64_t>(block_bytes, std::uint64_t{1} << 30)));
  while (true) {
    const auto t0 = Clock::now();
    const std::size_t n = source->read(block.data(), block.size());
    const auto t1 = Clock::now();
    read_s += seconds(t1 - t0);
    if (n == 0) break;
    splitter.feed(block.data(), n);
    split_s += seconds(Clock::now() - t1);
  }
  splitter.finish();
  const auto t_finish = Clock::now();
  KinetoColumns columns = pipeline.finish();
  columns.finish_seconds = seconds(Clock::now() - t_finish);
  columns.read_seconds = read_s;
  columns.submit_wait_seconds = wait_s;
  columns.split_seconds = split_s - wait_s;
  columns.rank = kineto_reader::parse_rank(splitter.info());
  return columns;
}

}  // namespace cta