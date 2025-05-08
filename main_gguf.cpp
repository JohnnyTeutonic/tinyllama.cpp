#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iomanip> // For std::setw, std::fixed, std::setprecision

#include "logger.h"
#include "api.h"


int main(int argc, char** argv) {
    if (argc < 2) {
        Logger::error("Usage: " + std::string(argv[0]) + " <path_to_gguf_file> [prompt] [steps] [temperature]");
        Logger::error("Example: " + std::string(argv[0]) + " ./models/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf \"Who is awesome?\" 64 0.7");
        std::cerr << "Usage: " << argv[0] << " <path_to_gguf_file> [prompt] [steps] [temperature]" << std::endl;
        return 1;
    }

    std::string gguf_model_path = argv[1];
    std::string prompt = "What is the capital of France?"; // Default raw prompt
    int steps = 64;             // Default max new tokens
    float temperature = 0.7f;   // Default temperature

    if (argc > 2) {
        prompt = argv[2]; // User provides raw prompt
    }
    if (argc > 3) {
        try {
            steps = std::stoi(argv[3]);
        } catch (const std::exception& e) {
            Logger::error("Invalid steps argument: " + std::string(argv[3]) + ". Using default: " + std::to_string(steps));
        }
    }
    if (argc > 4) {
        try {
            temperature = std::stof(argv[4]);
        } catch (const std::exception& e) {
            Logger::error("Invalid temperature argument: " + std::string(argv[4]) + ". Using default: " + std::to_string(temperature));
        }
    }

    // Log inputs
    Logger::info("Using GGUF model file: " + gguf_model_path);
    Logger::info("Raw Prompt (to be formatted by API): \"" + prompt + "\"");
    Logger::info("Steps (max new tokens): " + std::to_string(steps));
    Logger::info("Temperature: " + std::to_string(temperature));

    try {
        // 1. Create TinyLlamaSession
        // The constructor handles loading the GGUF model and its associated tokenizer.
        tinyllama::TinyLlamaSession session(gguf_model_path);
        Logger::info("TinyLlamaSession initialized successfully for GGUF model.");

        // 2. Generate text
        // The session.generate() method internally applies the "Q: prompt\nA:" formatting.
        std::string generated_text = session.generate(prompt, steps, temperature);

        // 3. Print the generated text
        // We print the original prompt and then the generated part.
        std::cout << "Q: " << prompt << "\nA:" << generated_text << std::endl;

    } catch (const std::exception& e) {
        Logger::error("An error occurred: " + std::string(e.what()));
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 
