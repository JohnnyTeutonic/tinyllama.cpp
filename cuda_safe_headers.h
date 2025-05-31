#ifndef CUDA_SAFE_HEADERS_H
#define CUDA_SAFE_HEADERS_H

/**
 * @file cuda_safe_headers.h
 * @brief Safe CUDA header inclusion wrapper for Windows CUDA 12.1+ compatibility
 * 
 * This header provides a workaround for Windows CUDA 12.1+ where cuda_fp16.hpp
 * and cuda_bf16.hpp try to include non-existent 'nv/target' files.
 */

#ifdef HAS_CUDA

// For Windows CUDA 12.1+, we need to be very careful about header inclusion
#if defined(WINDOWS_CUDA_12_1_WORKAROUND) && defined(_WIN32)

// Only include the essential CUDA headers that are known to work
#include <cuda_runtime.h>

// Conditionally include cuBLAS - this should work without fp16/bf16 headers
#ifdef __cplusplus
extern "C" {
#endif
#include <cublas_v2.h>
#ifdef __cplusplus
}
#endif

// Forward declare essential types that might be missing
#ifndef __CUDA_FP16_TYPES_EXIST__
typedef struct {
    unsigned short __x;
} __half;

typedef struct __align__(4) {
    __half x, y;
} __half2;
#endif

#ifndef __CUDA_BF16_TYPES_EXIST__
typedef struct {
    unsigned short __x;
} __nv_bfloat16;
#endif

#else
// For non-problematic platforms, include headers normally
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Only include these if they're not explicitly blocked
#ifndef __CUDA_FP16_H__
#include <cuda_fp16.h>
#endif

#ifndef __CUDA_BF16_H__
#include <cuda_bf16.h>
#endif

#endif // WINDOWS_CUDA_12_1_WORKAROUND

#endif // HAS_CUDA

#endif // CUDA_SAFE_HEADERS_H 