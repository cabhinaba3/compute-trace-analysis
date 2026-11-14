#include "readers/kineto/pipeline.hpp"

#include <stdexcept>

#include "readers/kineto/chunk_parser.hpp"

namespace cta::kineto_reader {

Pipeline::Pipeline(int workers) : max_in_flight_(static_cast<std::size_t>(workers) + 4) {
  for (int i = 0; i < workers; ++i) workers_.emplace_back([this] { work(); });
  merger_thread_ = std::thread([this] { merge(); });
}

Pipeline::~Pipeline() { stop(); }

void Pipeline::submit(std::string&& data) {
  std::unique_lock<std::mutex> lock(m_);
  room_.wait(lock, [&] { return aborted_ || next_submit_ - next_merge_ < max_in_flight_; });
  if (aborted_) {
    if (error_) std::rethrow_exception(error_);
    throw std::runtime_error("trace reader stopped");
  }
  queue_.push_back(Chunk{next_submit_++, std::move(data)});
  work_.notify_one();
}

KinetoColumns Pipeline::finish() {
  {
    std::lock_guard<std::mutex> lock(m_);
    closed_ = true;
  }
  work_.notify_all();
  result_.notify_all();
  for (auto& t : workers_) t.join();
  merger_thread_.join();
  if (error_) std::rethrow_exception(error_);
  return merger_.finish();
}

void Pipeline::fail(std::exception_ptr e) {
  {
    std::lock_guard<std::mutex> lock(m_);
    if (!error_) error_ = e;
    aborted_ = true;
  }
  work_.notify_all();
  room_.notify_all();
  result_.notify_all();
}

void Pipeline::stop() {
  {
    std::lock_guard<std::mutex> lock(m_);
    aborted_ = true;
    closed_ = true;
  }
  work_.notify_all();
  room_.notify_all();
  result_.notify_all();
  for (auto& t : workers_)
    if (t.joinable()) t.join();
  if (merger_thread_.joinable()) merger_thread_.join();
}

void Pipeline::work() {
  ChunkParser parser;
  while (true) {
    Chunk chunk;
    {
      std::unique_lock<std::mutex> lock(m_);
      work_.wait(lock, [&] { return aborted_ || !queue_.empty() || closed_; });
      if (aborted_ || queue_.empty()) return;
      chunk = std::move(queue_.front());
      queue_.pop_front();
    }
    try {
      auto out = std::make_unique<ChunkOut>();
      parser.parse(chunk, *out);
      chunk.data = std::string();
      {
        std::lock_guard<std::mutex> lock(m_);
        done_.emplace(chunk.seq, std::move(out));
      }
      result_.notify_all();
    } catch (...) {
      fail(std::current_exception());
      return;
    }
  }
}

void Pipeline::merge() {
  while (true) {
    std::unique_ptr<ChunkOut> out;
    {
      std::unique_lock<std::mutex> lock(m_);
      result_.wait(lock, [&] {
        return aborted_ || done_.count(next_merge_) > 0 || (closed_ && next_merge_ == next_submit_);
      });
      if (aborted_) return;
      auto it = done_.find(next_merge_);
      if (it == done_.end()) return;  // closed and everything merged
      out = std::move(it->second);
      done_.erase(it);
      ++next_merge_;
    }
    room_.notify_all();
    try {
      merger_.add(*out);
    } catch (...) {
      fail(std::current_exception());
      return;
    }
  }
}

}  // namespace cta::kineto_reader
