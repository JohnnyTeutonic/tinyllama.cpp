#pragma once

#include <algorithm>
#include <cmath>

#ifdef _MSC_VER
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

namespace detail {
template <typename T>
inline T safe_min(T a, T b) {
    return (std::min)(a, b);  // Parentheses prevent macro expansion
}

template <typename T>
inline T safe_max(T a, T b) {
    return (std::max)(a, b);  // Parentheses prevent macro expansion
}

inline float safe_sqrt(float x) {
    return (std::sqrt)(x);  // Parentheses prevent macro expansion
}
}  // namespace detail

#define SAFE_MIN(a, b) detail::safe_min((a), (b))
#define SAFE_MAX(a, b) detail::safe_max((a), (b))
#define SAFE_SQRT(x) detail::safe_sqrt(x) 