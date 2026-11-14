#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>

namespace cta {

// Distinct strings numbered in order of first appearance.
class Interner {
 public:
  std::int32_t intern(std::string_view s, bool* is_new = nullptr) {
    auto it = index_.find(s);
    if (it != index_.end()) {
      if (is_new) *is_new = false;
      return it->second;
    }
    const std::int32_t id = static_cast<std::int32_t>(items_.size());
    items_.emplace_back(s);  // a deque never moves its elements, so the views stay valid
    index_.emplace(std::string_view(items_.back()), id);
    if (is_new) *is_new = true;
    return id;
  }
  const std::deque<std::string>& items() const { return items_; }
  std::size_t size() const { return items_.size(); }

 private:
  std::deque<std::string> items_;
  std::unordered_map<std::string_view, std::int32_t> index_;
};

}  // namespace cta
