#pragma once

#include <cstdint>

// BFloat16 constants
namespace bfloat16 {
    constexpr uint16_t EXPONENT_MASK = 0x7F80;
    constexpr uint16_t MANTISSA_MASK = 0x007F;
    constexpr uint16_t SIGN_BIT = 0x8000;
    constexpr uint16_t ZERO = 0x0000;
    constexpr uint16_t NEG_ZERO = 0x8000;
    constexpr int SHIFT_BITS = 16;
}

// Attention and scaling constants
namespace attention {
    constexpr float ATTENTION_SCALE_BASE = std::sqrt(1.0f / 8.0f);  // 1/sqrt(head_dim) for head_dim=8
    constexpr float MIN_SCALE = 1e-4f;
    constexpr float MAX_SCALE = 1e4f;
}

// RoPE constants
namespace rope {
    constexpr float ROPE_THETA = 10000.0f;
    constexpr int MAX_SEQUENCE_LENGTH = 2048;  // Default max sequence length
}

// Numeric stability constants
namespace numeric {
    constexpr float MIN_NORM_EPS = 1e-5f;
    constexpr float DEFAULT_EPS = 1e-6f;
    constexpr float MAX_LOGIT_THRESHOLD = 100.0f;
} 