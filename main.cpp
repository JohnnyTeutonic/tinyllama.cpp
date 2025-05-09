#include <algorithm>  // For std::find_if_not
#include <cctype>     // For std::isspace
#include <cstdio>     // For std::remove
#include <iomanip>    // Include for std::setw, std::fixed, std::setprecision
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "api.h"
#include "logger.h"

// Helper function to trim leading/trailing whitespace
std::string trim_whitespace(const std::string& s) {
  auto start = std::find_if_not(
      s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
  auto end = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) {
               return std::isspace(c);
             }).base();
  return (start < end ? std::string(start, end) : std::string());
}

int main(int argc, char** argv) {
  // Accept data directory as argument (default: "data")
  std::string model_path_or_dir = "data";  // Changed variable name for clarity
  std::string prompt = "Hello, world!";    // Default prompt
  int steps = 64;                          // Default max new tokens
  float temperature = 0.7f;                // Default temperature

  if (argc > 1) {
    model_path_or_dir = argv[1];
  }
  // --- Read prompt from argv (with trimming) ---
  if (argc > 2) {
    std::string raw_prompt_from_argv = argv[2];
    prompt = trim_whitespace(raw_prompt_from_argv);  // Trim whitespace
  }
  // --- Read steps from argv ---
  if (argc > 3) {
    try {
      steps = std::stoi(argv[3]);
    } catch (const std::exception& e) {
      Logger::error("Invalid steps argument: " + std::string(argv[3]) +
                    ". Using default: " + std::to_string(steps));
    }
  }
  // --------------------------
  if (argc > 4) {  // Keep temperature from argv if provided
    try {
      temperature = std::stof(argv[4]);
    } catch (const std::exception& e) {
      Logger::error("Invalid temperature argument: " + std::string(argv[4]) +
                    ". Using default: " + std::to_string(temperature));
    }
  }

  // --- Remove forced values ---
  // prompt = "What is the capital of France?";
  // steps = 30;
  // ----------------------------

  // Log inputs
  Logger::info("Using model path/directory: " + model_path_or_dir);
  Logger::info("Raw Prompt (from argv, trimmed): \"" + prompt + "\"");
  Logger::info("Steps (from argv): " + std::to_string(steps));
  Logger::info("Temperature Used (currently ignored by API sampling): " +
               std::to_string(temperature));

  try {
    // 1. Create TinyLlamaSession
    tinyllama::TinyLlamaSession session(model_path_or_dir);
    Logger::info("TinyLlamaSession initialized successfully.");

    // 2. Generate text using prompt and steps from argv
    std::string generated_text = session.generate(prompt, steps, temperature);

    // 3. Print the generated text
    std::cout << "Prompt: " + prompt << std::endl;
    std::cout << "Generated: " << generated_text << std::endl;

  } catch (const std::exception& e) {
    Logger::error(std::string("An error occurred: ") + e.what());
    // You might want to print to cerr as well for command-line tools
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}