#pragma once

#include "model.h"
#include "kv_cache.h"
#include <vector>

class CPUBatchProcessor {
public:
    explicit CPUBatchProcessor(TinyLlamaModel* model);
    
    std::vector<float> forward_cpu_batch(
        const std::vector<float>& batch_input_activations,
        int num_tokens_in_batch,
        int num_cpu_layers_to_process,
        int start_pos_in_sequence,
        KVCache* kv_cache,
        const std::vector<int>& prompt_lengths);

private:
    TinyLlamaModel* model_;
}; 