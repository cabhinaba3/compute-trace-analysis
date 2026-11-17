#include "readers/kineto/reader.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>

#include "util/cpus.hpp"
#include "util/ordered_pool.hpp"
#include "util/util.hpp"
#include "readers/kineto/chunk_parser.hpp"
#include "readers/kineto/merger.hpp"
#include "readers/kineto/source.hpp"
#include "readers/kineto/splitter.hpp"

namespace cta::readers {

KinetoColumns read_kineto(const std::string& path, int threads, std::uint64_t chunk_bytes,
                                  std::uint64_t block_bytes) {
  if (threads < 0) throw std::invalid_argument("threads must be >= 0");
  if (block_bytes < 1) throw std::invalid_argument("block_bytes must be at least 1");
  if (chunk_bytes < 64) throw std::invalid_argument("chunk_bytes must be at least 64");
  if (threads == 0) threads = usable_cpus();

  auto source = kineto::open_source(path);

  using Clock = std::chrono::steady_clock;
  double read_s = 0, split_s = 0, wait_s = 0;

  // Chunks are parsed on `threads` workers, each with its own parser, and merged in
  // file order. At most `threads + 4` chunks exist at a time.
  using kineto::ChunkOut;
  kineto::Merger merger;
  using Pool = OrderedPool<std::string, std::unique_ptr<ChunkOut>>;
  Pool pool(
      threads,
      []() -> Pool::Work {
        auto parser = std::make_shared<kineto::ChunkParser>();
        return [parser](std::string& text) {
          auto out = std::make_unique<ChunkOut>();
          parser->parse(text, *out);
          return out;
        };
      },
      [&](std::unique_ptr<ChunkOut>&& out) { merger.add(*out); }, static_cast<std::size_t>(threads) + 4);
  kineto::Splitter splitter(static_cast<std::size_t>(chunk_bytes), [&](std::string&& chunk) {
    const auto t0 = Clock::now();
    pool.submit(std::move(chunk));
    wait_s += duration_seconds(Clock::now() - t0);
  });
  std::vector<char> block(static_cast<std::size_t>(std::min<std::uint64_t>(block_bytes, std::uint64_t{1} << 30)));
  while (true) {
    const auto t0 = Clock::now();
    const std::size_t n = source->read(block.data(), block.size());
    const auto t1 = Clock::now();
    read_s += duration_seconds(t1 - t0);
    if (n == 0) break;
    splitter.feed(block.data(), n);
    split_s += duration_seconds(Clock::now() - t1);
  }
  splitter.finish();
  const auto t_finish = Clock::now();
  pool.finish();
  KinetoColumns columns = merger.finish();
  columns.timing.finish = duration_seconds(Clock::now() - t_finish);
  columns.timing.read = read_s;
  columns.timing.submit_wait = wait_s;
  columns.timing.split = split_s - wait_s;
  columns.rank = kineto::parse_rank(splitter.info());
  return columns;
}

}  // namespace cta::readers