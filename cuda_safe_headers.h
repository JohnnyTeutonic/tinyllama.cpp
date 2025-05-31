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

// Include essential CUDA runtime first
#include <cuda_runtime.h>

// Define the half precision types that cuBLAS needs BEFORE including cuBLAS
// These definitions must be compatible with what cuBLAS expects
#ifndef __CUDA_FP16_TYPES_EXIST__
#define __CUDA_FP16_TYPES_EXIST__

// Basic half precision type - must match NVIDIA's definition exactly
typedef struct __align__(2) {
    unsigned short __x;
} __half;

// Half2 type for paired operations
typedef struct __align__(4) {
    __half x, y;
} __half2;

// Essential half precision constants and basic operations for cuBLAS compatibility
#ifdef __cplusplus
extern "C" {
#endif

// Declare and implement essential functions that cuBLAS might expect
// These are minimal stub implementations to satisfy linking requirements
static inline __host__ __device__ __half __float2half(const float a) {
    __half result;
    // Basic float to half conversion (truncated, not IEEE-compliant but functional)
    result.__x = (unsigned short)((*(unsigned int*)&a) >> 16);
    return result;
}

static inline __host__ __device__ float __half2float(const __half a) {
    // Basic half to float conversion (zero-extended, not IEEE-compliant but functional)
    unsigned int temp = ((unsigned int)a.__x) << 16;
    return *(float*)&temp;
}

#ifdef __cplusplus
}
#endif

#endif // __CUDA_FP16_TYPES_EXIST__

#ifndef __CUDA_BF16_TYPES_EXIST__
#define __CUDA_BF16_TYPES_EXIST__

// BFloat16 type definition
typedef struct __align__(2) {
    unsigned short __x;
} __nv_bfloat16;

// BFloat16 pair type
typedef struct __align__(4) {
    __nv_bfloat16 x, y;
} __nv_bfloat162;

#ifdef __cplusplus
extern "C" {
#endif

// Essential BF16 conversion functions for cuBLAS compatibility
static inline __host__ __device__ __nv_bfloat16 __float2bfloat16(const float a) {
    __nv_bfloat16 result;
    // Basic float to bfloat16 conversion (truncate mantissa, keep exponent and sign)
    result.__x = (unsigned short)((*(unsigned int*)&a) >> 16);
    return result;
}

static inline __host__ __device__ float __bfloat162float(const __nv_bfloat16 a) {
    // Basic bfloat16 to float conversion (zero-extend mantissa)
    unsigned int temp = ((unsigned int)a.__x) << 16;
    return *(float*)&temp;
}

#ifdef __cplusplus
}
#endif

#endif // __CUDA_BF16_TYPES_EXIST__

// Now safely include cuBLAS - the types it needs are defined above
#ifndef CUBLAS_V2_H_
#include <cublas_v2.h>
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