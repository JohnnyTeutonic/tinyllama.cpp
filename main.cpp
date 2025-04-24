#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>
#include <map>
#include "safetensors_loader.h"
#include "tokenizer.h"
#include "logger.h"
#include "prompt.h"
#include "model.h"

// TODO: Implement safetensors loader
// TODO: Implement TinyLlama model and inference

// Utility: Read file into string
std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file: " + path);
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    // Accept data directory as argument (default: "data")
    std::string data_dir = "data";
    if (argc > 1) {
        data_dir = argv[1];
    }
    std::string config_path = data_dir + "/config.json";
    std::string tokenizer_path = data_dir + "/tokenizer.model";
    std::string safetensors_path = data_dir + "/model.safetensors";

    Logger::info("Using data directory: " + data_dir);
    Logger::info("Loading config: " + config_path);

    // 1. Load config.json
    nlohmann::json config;
    try {
        std::string config_str = read_file(config_path);
        config = nlohmann::json::parse(config_str);
    } catch (const std::exception& e) {
        Logger::error(std::string("Error loading config.json: ") + e.what());
        return 1;
    }

    // 2. Load tokenizer.model (SentencePiece)
    sentencepiece::SentencePieceProcessor sp;
    auto sp_status = sp.Load(tokenizer_path);
    if (!sp_status.ok()) {
        Logger::error("Failed to load SentencePiece model: " + sp_status.ToString());
        return 1;
    }

    // Tokenizer usage example
    try {
        Tokenizer tokenizer(data_dir);
        std::string sample = "Hello, world!";
        auto ids = tokenizer.tokenize(sample);
        std::string idstr;
        for (int id : ids) idstr += std::to_string(id) + " ";
        Logger::info("Tokenized: " + idstr);
        std::string detok = tokenizer.detokenize(ids);
        Logger::info("Detokenized: " + detok);

        // Prompt formatting example
        std::string system = "You are a helpful assistant.";
        std::vector<std::string> user_msgs = {"What is the capital of France?"};
        std::vector<std::string> assistant_msgs; // empty for next turn
        std::string prompt = format_prompt(system, user_msgs, assistant_msgs, tokenizer.chat_template());
        Logger::info("Formatted prompt: " + prompt);
    } catch (const std::exception& e) {
        Logger::error(std::string("Tokenizer error: ") + e.what());
        return 1;
    }

    // 3. Load model.safetensors
    try {
        SafeTensorsLoader st_loader(safetensors_path);
        auto names = st_loader.tensor_names();
        Logger::info("Loaded " + std::to_string(names.size()) + " tensors from model.safetensors:");
        for (const auto& n : names) {
            const auto& info = st_loader.get_tensor_info(n);
            std::string shape_str = "[";
            for (size_t i = 0; i < info.shape.size(); ++i) {
                shape_str += std::to_string(info.shape[i]);
                if (i + 1 < info.shape.size()) shape_str += ", ";
            }
            shape_str += "]";
            Logger::info("  " + n + " | dtype: " + info.dtype + ", shape: " + shape_str);
        }

        // Model weight loading example
        try {
            ModelConfig mcfg = parse_model_config(config);
            TinyLlamaModel model(mcfg, st_loader);
            Logger::info("TinyLlamaModel weights loaded successfully.");

            // Forward pass test
            std::string sample = "Hello, world!";
            Tokenizer tokenizer(data_dir);
            auto ids = tokenizer.tokenize(sample);
            if (!ids.empty()) {
                int token_id = ids[0];
                auto logits = model.forward(token_id);
                Logger::info("Logits size: " + std::to_string(logits.size()));
                std::string logit_str;
                for (int i = 0; i < 10 && i < logits.size(); ++i) logit_str += std::to_string(logits[i]) + " ";
                Logger::info("First 10 logits: " + logit_str);
                float minv = *std::min_element(logits.begin(), logits.end());
                float maxv = *std::max_element(logits.begin(), logits.end());
                float mean = std::accumulate(logits.begin(), logits.end(), 0.0f) / logits.size();
                Logger::info("Logits min: " + std::to_string(minv) + ", max: " + std::to_string(maxv) + ", mean: " + std::to_string(mean));
                bool all_finite = std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); });
                Logger::info(std::string("All logits finite: ") + (all_finite ? "yes" : "no"));
                bool all_same = std::all_of(logits.begin(), logits.end(), [&](float v) { return v == logits[0]; });
                Logger::info(std::string("All logits same: ") + (all_same ? "yes" : "no"));
            } else {
                Logger::error("Tokenizer produced no tokens for forward pass test.");
            }
        } catch (const std::exception& e) {
            Logger::error(std::string("Model weight loading error: ") + e.what());
            return 1;
        }
    } catch (const std::exception& e) {
        Logger::error(std::string("Error loading model.safetensors: ") + e.what());
        return 1;
    }

    // 4. TODO: Format prompt using chat template
    // 5. TODO: Tokenize prompt
    // 6. TODO: Run inference
    // 7. TODO: Detokenize output

    Logger::info("Pipeline skeleton initialized.\n");
    return 0;
} 