// Converting C++ results to Python objects without copying the arrays.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <concepts>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cta::bindings {

// Hands a vector to NumPy without a copy: the array owns the vector.
template <typename T>
nanobind::object to_numpy(std::vector<T>&& values) {
  namespace nb = nanobind;
  if (values.empty()) {
    nb::ndarray<nb::numpy, T, nb::ndim<1>> array(nullptr, {0});
    return nb::cast(array);
  }
  auto* owned = new std::vector<T>(std::move(values));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  nb::ndarray<nb::numpy, T, nb::ndim<1>> array(owned->data(), {owned->size()}, owner);
  return nb::cast(array);
}

template <typename T>
inline constexpr bool kIsVector = false;
template <typename T, typename Allocator>
inline constexpr bool kIsVector<std::vector<T, Allocator>> = true;

template <typename T>
inline constexpr bool kIsOptional = false;
template <typename T>
inline constexpr bool kIsOptional<std::optional<T>> = true;

// A struct that lists its members with for_each_column(visit), such as the
// readers' column structs.
template <typename T>
concept Columns = requires(T& value) { value.for_each_column([](const char*, auto&) {}); };

template <Columns T>
nanobind::dict to_dict(T&& columns);

// One member as a Python object: a nested column struct becomes a dict, a
// string table a list, any other vector a NumPy array, an empty optional None,
// and a scalar itself.
template <typename T>
nanobind::object to_python(T&& value) {
  namespace nb = nanobind;
  using V = std::remove_cvref_t<T>;
  if constexpr (Columns<V>) {
    return to_dict(std::move(value));
  } else if constexpr (std::same_as<V, std::vector<std::string>>) {
    return nb::cast(value);
  } else if constexpr (kIsVector<V>) {
    return to_numpy(std::move(value));
  } else if constexpr (kIsOptional<V>) {
    return value ? nb::cast(*value) : nb::none();
  } else {
    return nb::cast(value);
  }
}

// A column struct as a dict of its members, moving the arrays out.
template <Columns T>
nanobind::dict to_dict(T&& columns) {
  nanobind::dict out;
  columns.for_each_column([&](const char* name, auto& member) { out[name] = to_python(std::move(member)); });
  return out;
}

}  // namespace cta::bindings
