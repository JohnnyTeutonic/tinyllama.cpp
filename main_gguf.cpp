#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include "model.h"
#include "logger.h"
#include "gguf_parser.h"

int main(int argc, char **argv) {
    Logger::info("GGUF Loader - Starting...");

    // TODO: Add file path argument parsing (e.g., from argv)
    std::string filename = "data/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"; // Hardcoded for now

    try {
        TinyLlamaModel model({}, filename); // Model auto-loads config from GGUF
        const auto& config = model.get_config();
        std::stringstream ss;
        ss << "--- Model Summary ---\n";
        ss << "Hidden size: " << config.hidden_size << "\n";
        ss << "Intermediate size: " << config.intermediate_size << "\n";
        ss << "Num attention heads: " << config.num_attention_heads << "\n";
        ss << "Num key/value heads: " << config.num_key_value_heads << "\n";
        ss << "Num hidden layers: " << config.num_hidden_layers << "\n";
        ss << "Vocab size: " << config.vocab_size << "\n";
        ss << "Max position embeddings: " << config.max_position_embeddings << "\n";
        ss << "---------------------\n";
        ss << "Model loaded successfully from: " << filename;
        Logger::info(ss.str());

        // For debugging: log GGUFData tensor info and metadata from the model
        if (const GGUFData* gguf_data = model.get_gguf_data()) {
            Logger::info("--- GGUF Tensor Info ---");
            for (const auto& t : gguf_data->tensor_infos) {
                std::stringstream ts;
                ts << "Tensor: '" << t.name << "', Type=" << t.type
                   << ", Shape=[";
                for (size_t i = 0; i < t.shape.size(); ++i) ts << t.shape[i] << (i + 1 < t.shape.size() ? ", " : "");
                ts << "], Offset=" << t.offset << ", Size=" << t.size_in_bytes << " bytes";
                Logger::info(ts.str());
            }
            Logger::info("------------------------");
            // Optionally log some metadata
            Logger::info("--- GGUF Metadata ---");
            for (const auto& kv : gguf_data->metadata) {
                std::stringstream ms;
                ms << kv.first << ": ";
                // Print the value as string if possible
                if (std::holds_alternative<std::string>(kv.second)) {
                    ms << std::get<std::string>(kv.second);
                } else if (std::holds_alternative<int32_t>(kv.second)) {
                    ms << std::get<int32_t>(kv.second);
                } else if (std::holds_alternative<uint32_t>(kv.second)) {
                    ms << std::get<uint32_t>(kv.second);
                } else if (std::holds_alternative<int64_t>(kv.second)) {
                    ms << std::get<int64_t>(kv.second);
                } else if (std::holds_alternative<uint64_t>(kv.second)) {
                    ms << std::get<uint64_t>(kv.second);
                } else if (std::holds_alternative<float>(kv.second)) {
                    ms << std::get<float>(kv.second);
                } else if (std::holds_alternative<double>(kv.second)) {
                    ms << std::get<double>(kv.second);
                } else if (std::holds_alternative<bool>(kv.second)) {
                    ms << (std::get<bool>(kv.second) ? "true" : "false");
                } else {
                    ms << "(unhandled type)";
                }
                Logger::info(ms.str());
            }
            Logger::info("---------------------");
        }
    } catch (const std::exception& e) {
        Logger::error(std::string("*** Error loading model ***\n") + e.what());
        return 1;
    }

    Logger::info("GGUF Loader - Finished successfully.");
    return 0;
} 