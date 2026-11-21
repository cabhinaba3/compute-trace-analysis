// RAII temporary scratch directory for out-of-core spill files.
#pragma once

#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace cta::streaming {

// Creates an isolated run directory under `base_dir` (defaulting to
// `<cache_dir>/_scratch` or the system temp dir) and removes it on destruction.
class ScratchDir {
 public:
  explicit ScratchDir(const std::string& base_dir = {}) {
    static std::atomic<std::uint64_t> counter{0};
    std::filesystem::path root = base_dir.empty() ? std::filesystem::temp_directory_path() / "cta_scratch"
                                                  : std::filesystem::path(base_dir);
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) throw std::runtime_error("cannot create scratch root " + root.string() + ": " + ec.message());
    const std::uint64_t seq = counter.fetch_add(1, std::memory_order_relaxed);
    dir_ = root / ("run_" + std::to_string(::getpid()) + "_" + std::to_string(seq));
    std::filesystem::create_directories(dir_, ec);
    if (ec) throw std::runtime_error("cannot create scratch dir " + dir_.string() + ": " + ec.message());
  }

  ~ScratchDir() { cleanup(); }

  ScratchDir(const ScratchDir&) = delete;
  ScratchDir& operator=(const ScratchDir&) = delete;

  ScratchDir(ScratchDir&& other) noexcept : dir_(std::move(other.dir_)), keep_(other.keep_) {
    other.dir_.clear();
  }

  ScratchDir& operator=(ScratchDir&& other) noexcept {
    if (this != &other) {
      cleanup();
      dir_ = std::move(other.dir_);
      keep_ = other.keep_;
      other.dir_.clear();
    }
    return *this;
  }

  const std::filesystem::path& path() const { return dir_; }

  std::string file(const std::string& name) const { return (dir_ / name).string(); }

  void set_keep_on_exit(bool keep) { keep_ = keep; }

 private:
  void cleanup() noexcept {
    if (keep_ || dir_.empty()) return;
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
    dir_.clear();
  }

  std::filesystem::path dir_;
  bool keep_ = false;
};

}  // namespace cta::streaming
