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

    std::cout << "Using data directory: " << data_dir << std::endl;
    std::cout << "Loading config: " << config_path << std::endl;

    // 1. Load config.json
    nlohmann::json config;
    try {
        std::string config_str = read_file(config_path);
        config = nlohmann::json::parse(config_str);
    } catch (const std::exception& e) {
        std::cerr << "Error loading config.json: " << e.what() << std::endl;
        return 1;
    }

    // 2. Load tokenizer.model (SentencePiece)
    sentencepiece::SentencePieceProcessor sp;
    auto sp_status = sp.Load(tokenizer_path);
    if (!sp_status.ok()) {
        std::cerr << "Failed to load SentencePiece model: " << sp_status.ToString() << std::endl;
        return 1;
    }

    // 3. Load model.safetensors
    try {
        SafeTensorsLoader st_loader(safetensors_path);
        auto names = st_loader.tensor_names();
        std::cout << "Loaded " << names.size() << " tensors from model.safetensors:\n";
        for (const auto& n : names) {
            const auto& info = st_loader.get_tensor_info(n);
            std::cout << "  " << n << " | dtype: " << info.dtype << ", shape: [";
            for (size_t i = 0; i < info.shape.size(); ++i) {
                std::cout << info.shape[i];
                if (i + 1 < info.shape.size()) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading model.safetensors: " << e.what() << std::endl;
        return 1;
    }

    // 4. TODO: Format prompt using chat template
    // 5. TODO: Tokenize prompt
    // 6. TODO: Run inference
    // 7. TODO: Detokenize output

    std::cout << "Pipeline skeleton initialized.\n";
    return 0;
} 