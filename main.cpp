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