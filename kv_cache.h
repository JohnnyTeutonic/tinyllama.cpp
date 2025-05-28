#ifndef KV_CACHE_H
#define KV_CACHE_H

#include "model.h"

void initialize_kv_cache(KVCache* kv_cache, const ModelConfig& config, 
                         int total_num_model_layers, int num_gpu_layers_to_allocate, 
                         int max_seq_len_arg, int num_kv_heads,
                         int head_dim, int max_batch_size_arg);

#endif // KV_CACHE_H 