#include "api.h"
#include "logger.h" // For logging setup
#include <iostream>

int main(int argc, char** argv) {
    std::string model_dir = "data";
    if (argc > 1) model_dir = argv[1];

    try {
        Logger::info("Creating TinyLlamaSession...");
        tinyllama::TinyLlamaSession session(model_dir);
        Logger::info("Session created.");

        std::string prompt = "Q: What is the capital of France?\nA:";
        std::cout << "Prompt: " << prompt << std::endl;

        Logger::info("Calling generate...");
        std::string result = session.generate(prompt, 50); // Generate up to 50 tokens
        Logger::info("Generation complete.");

        std::cout << "Generated: " << result << std::endl;

    } catch (const std::exception& e) {
        Logger::error(std::string("API Error: ") + e.what());
        std::cerr << "API Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}