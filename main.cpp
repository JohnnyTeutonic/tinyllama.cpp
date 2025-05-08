#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include "logger.h"
#include <cstdio> // For std::remove
#include <sstream>
#include <iomanip> // Include for std::setw, std::fixed, std::setprecision
#include "api.h"

int main(int argc, char** argv) {
    // Accept data directory as argument (default: "data")
    std::string model_path_or_dir = "data"; // Changed variable name for clarity
    std::string prompt = "Hello, world!";
    int steps = 64;
    float temperature = 0.7f; // Default temperature

    if (argc > 1) {
        model_path_or_dir = argv[1];
    }
    if (argc > 2) {
        prompt = argv[2];
    }
    if (argc > 3) {
        try {
            steps = std::stoi(argv[3]);
        } catch (const std::exception& e) {
            Logger::error(std::string("Invalid steps argument: ") + argv[3] + ". Using default: " + std::to_string(steps));
        }
    }
    if (argc > 4) {
        try {
            temperature = std::stof(argv[4]);
        } catch (const std::exception& e) {
            Logger::error(std::string("Invalid temperature argument: ") + argv[4] + ". Using default: " + std::to_string(temperature));
        }
    }

    Logger::info("Using model path/directory: " + model_path_or_dir);
    Logger::info("Prompt: \"" + prompt + "\"");
    Logger::info("Steps: " + std::to_string(steps));
    Logger::info("Temperature: " + std::to_string(temperature));

    try {
        // 1. Create TinyLlamaSession
        // The constructor will handle loading model, tokenizer, etc.
        // It will also detect if model_path_or_dir is a directory or a .gguf file.
        tinyllama::TinyLlamaSession session(model_path_or_dir);
        Logger::info("TinyLlamaSession initialized successfully.");

        // 2. Generate text
        std::string generated_text = session.generate(prompt, steps, temperature);

        // 3. Print the generated text
        // The prompt is not included in the output of session.generate by default design.
        // If you want to print the prompt, do it here.
        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Generated: " << generated_text << std::endl;

    } catch (const std::exception& e) {
        Logger::error(std::string("An error occurred: ") + e.what());
        // You might want to print to cerr as well for command-line tools
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 