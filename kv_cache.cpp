#include "kv_cache.h"
#include "logger.h"

#ifdef HAS_CUDA
#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

void KVCache::initialize(const ModelConfig& config, 
                         int total_num_model_layers, int num_gpu_layers_to_allocate, 
                         int max_seq_len_arg, int num_kv_heads,
                         int head_dim, int max_batch_size_arg) {
  this->total_model_layers_ = total_num_model_layers;
  this->max_seq_len_config_ = max_seq_len_arg;
  this->max_batch_size = max_batch_size_arg;
  this->current_batch_size = 0;
  this->batch_seq_lens.clear();
  this->batch_seq_lens.resize(max_batch_size_arg, 0);
  layers.resize(total_num_model_layers);
  seq_len = 0;
  Logger::info("Allocating KVCache host vectors...");
  size_t cache_size_per_layer = static_cast<size_t>(max_seq_len_arg) *
                                static_cast<size_t>(max_batch_size_arg) *
                                static_cast<size_t>(num_kv_heads) *
                                static_cast<size_t>(head_dim);
  if (cache_size_per_layer == 0 && max_seq_len_arg > 0 && total_num_model_layers > 0) {
    throw std::runtime_error(
        "KVCache (CPU): Calculated cache size is zero for non-empty model. Check parameters.");
  }

  for (int l = 0; l < total_num_model_layers; ++l) {
    try {
      layers[l].k.assign(cache_size_per_layer, 0.0f);
      layers[l].v.assign(cache_size_per_layer, 0.0f);
    } catch (const std::bad_alloc& e) {
      Logger::error("Failed to allocate CPU KVCache for layer " +
                    std::to_string(l) + ": " + e.what());
      throw;
    }
  }
  Logger::info("KVCache (CPU) vectors allocated for " +
               std::to_string(total_num_model_layers) + " layers.");

#ifdef HAS_CUDA
  this->allocated_num_layers = num_gpu_layers_to_allocate;
  this->allocated_max_seq_len = max_seq_len_arg;
  this->allocated_num_kv_heads = num_kv_heads;
  this->allocated_head_dim = head_dim;

  if (num_gpu_layers_to_allocate > 0) {
      if (num_gpu_layers_to_allocate > total_num_model_layers) {
          Logger::warning("KVCache::initialize: num_gpu_layers_to_allocate (" + std::to_string(num_gpu_layers_to_allocate) +
                          ") > total_num_model_layers (" + std::to_string(total_num_model_layers) + 
                          "). Clamping to total_num_model_layers.");
          this->allocated_num_layers = total_num_model_layers;
          num_gpu_layers_to_allocate = total_num_model_layers;
      }

      size_t cache_elems_per_layer_gpu = static_cast<size_t>(max_seq_len_arg) *
                                 static_cast<size_t>(num_kv_heads) *
                                 static_cast<size_t>(head_dim);
      
      size_t fp32_cache_bytes_per_layer_gpu = cache_elems_per_layer_gpu * sizeof(float);
      size_t int8_cache_bytes_per_layer_gpu = cache_elems_per_layer_gpu * sizeof(int8_t);
      size_t num_scales_per_layer_gpu = static_cast<size_t>(max_seq_len_arg) * static_cast<size_t>(num_kv_heads);
      size_t scales_bytes_per_layer_gpu = num_scales_per_layer_gpu * sizeof(float);

      if (cache_elems_per_layer_gpu == 0 && config.use_kvcache_quantization) {
        throw std::runtime_error(
            "KVCache (CUDA INT8): Calculated cache elements per layer is zero. Check parameters.");
      } else if (cache_elems_per_layer_gpu == 0) {
        throw std::runtime_error(
            "KVCache (CUDA FP32): Calculated cache elements per layer is zero. Check parameters.");
      }

      if (config.use_kvcache_quantization) {
        Logger::info("Allocating INT8 KVCache + FP32 Scales on GPU for " + std::to_string(num_gpu_layers_to_allocate) +
                 " layers. Data size per layer: " +
                     std::to_string(int8_cache_bytes_per_layer_gpu / (1024.0 * 1024.0)) +
                 " MB. Scales size per layer: " + 
                     std::to_string(scales_bytes_per_layer_gpu / (1024.0 * 1024.0)) + " MB");
      } else {
        Logger::info("Allocating FP32 KVCache on GPU for " + std::to_string(num_gpu_layers_to_allocate) +
                 " layers, size per layer: " +
                     std::to_string(fp32_cache_bytes_per_layer_gpu / (1024.0 * 1024.0)) +
                 " MB");
      }

      int gpu_layer_start_model_idx = this->total_model_layers_ - num_gpu_layers_to_allocate;
      Logger::info("KVCache GPU allocation will target model layers from index " + std::to_string(gpu_layer_start_model_idx) +
                   " to " + std::to_string(gpu_layer_start_model_idx + num_gpu_layers_to_allocate - 1));

      for (int i = 0; i < num_gpu_layers_to_allocate; ++i) {
        int current_model_idx_for_gpu = gpu_layer_start_model_idx + i;

        if (current_model_idx_for_gpu < 0 || static_cast<size_t>(current_model_idx_for_gpu) >= layers.size()) {
            Logger::error("KVCache::initialize: Calculated current_model_idx_for_gpu (" + std::to_string(current_model_idx_for_gpu) + ") is out of bounds for layers vector (size " + std::to_string(layers.size()) + "). Skipping this layer.");
            continue;
        }

        if (layers[current_model_idx_for_gpu].k_dev_fp32) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " K dev fp32 pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].k_dev_fp32));
          layers[current_model_idx_for_gpu].k_dev_fp32 = nullptr;
        }
        if (layers[current_model_idx_for_gpu].v_dev_fp32) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " V dev fp32 pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].v_dev_fp32));
          layers[current_model_idx_for_gpu].v_dev_fp32 = nullptr;
        }
        if (layers[current_model_idx_for_gpu].k_dev_quantized) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " K dev quantized pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].k_dev_quantized));
          layers[current_model_idx_for_gpu].k_dev_quantized = nullptr;
        }
        if (layers[current_model_idx_for_gpu].v_dev_quantized) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " V dev quantized pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].v_dev_quantized));
          layers[current_model_idx_for_gpu].v_dev_quantized = nullptr;
        }
        if (layers[current_model_idx_for_gpu].k_dev_scales) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " K dev scales pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].k_dev_scales));
          layers[current_model_idx_for_gpu].k_dev_scales = nullptr;
        }
        if (layers[current_model_idx_for_gpu].v_dev_scales) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " V dev scales pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].v_dev_scales));
          layers[current_model_idx_for_gpu].v_dev_scales = nullptr;
        }
        
        if (config.use_kvcache_quantization) {
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].k_dev_quantized, int8_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].v_dev_quantized, int8_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].k_dev_scales, scales_bytes_per_layer_gpu));
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].v_dev_scales, scales_bytes_per_layer_gpu));

            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].k_dev_quantized, 0, int8_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].v_dev_quantized, 0, int8_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].k_dev_scales, 0, scales_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].v_dev_scales, 0, scales_bytes_per_layer_gpu));
        } else {
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].k_dev_fp32, fp32_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].v_dev_fp32, fp32_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].k_dev_fp32, 0, fp32_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].v_dev_fp32, 0, fp32_cache_bytes_per_layer_gpu));
        }
  }
      Logger::info("KVCache GPU allocation and zeroing complete for " + std::to_string(num_gpu_layers_to_allocate) + " layers.");
  } else {
      Logger::info("KVCache: No GPU layers requested for allocation (num_gpu_layers_to_allocate is 0). Skipping GPU KVCache allocation.");
      this->allocated_num_layers = 0; 
  }

#else
  Logger::info("KVCache (CPU-only build) initialized with dimensions for " +
               std::to_string(total_num_model_layers) + " layers, " +
               std::to_string(max_seq_len_arg) + " seq len, " +
               std::to_string(num_kv_heads) + " KV heads, " +
               std::to_string(head_dim) + " head dim");
#endif
} 