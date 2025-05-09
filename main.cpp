#include <algorithm>
#include <cctype>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "api.h"
#include "logger.h"

std::string trim_whitespace(const std::string& s) {
  auto start = std::find_if_not(
      s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
  auto end = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) {
               return std::isspace(c);
             }).base();
  return (start < end ? std::string(start, end) : std::string());
}

void print_usage() {
  std::cout << "Usage: tinyllama [model_path] [prompt] [steps] [temperature] [top_k] [top_p]\n"
            << "  model_path: Path to model directory or .gguf file (default: data)\n"
            << "  prompt: Input text (default: \"Hello, world!\")\n"
            << "  steps: Number of tokens to generate (default: 64)\n"
            << "  temperature: Sampling temperature, lower is more deterministic (default: 0.7)\n"
            << "  top_k: Limit sampling to top K tokens (default: 40)\n"
            << "  top_p: Limit sampling to top P probability mass (default: 0.9)\n";
}

int main(int argc, char** argv) {
  if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    print_usage();
    return 0;
  }

  std::string model_path_or_dir = "data";
  std::string prompt = "Hello, world!";
  int steps = 64;
  float temperature = 0.1f;
  int top_k = 40;
  float top_p = 0.9f;

  if (argc > 1) {
    model_path_or_dir = argv[1];
  }

  if (argc > 2) {
    std::string raw_prompt_from_argv = argv[2];
    prompt = trim_whitespace(raw_prompt_from_argv);
  }

  if (argc > 3) {
    try {
      steps = std::stoi(argv[3]);
    } catch (const std::exception& e) {
      Logger::error("Invalid steps argument: " + std::string(argv[3]) +
                    ". Using default: " + std::to_string(steps));
    }
  }

  if (argc > 4) {
    try {
      temperature = std::stof(argv[4]);
    } catch (const std::exception& e) {
      Logger::error("Invalid temperature argument: " + std::string(argv[4]) +
                    ". Using default: " + std::to_string(temperature));
    }
  }

  if (argc > 5) {
    try {
      top_k = std::stoi(argv[5]);
    } catch (const std::exception& e) {
      Logger::error("Invalid top_k argument: " + std::string(argv[5]) +
                    ". Using default: " + std::to_string(top_k));
    }
  }

  if (argc > 6) {
    try {
      top_p = std::stof(argv[6]);
    } catch (const std::exception& e) {
      Logger::error("Invalid top_p argument: " + std::string(argv[6]) +
                    ". Using default: " + std::to_string(top_p));
    }
  }

  Logger::info("Using model path/directory: " + model_path_or_dir);
  Logger::info("Raw Prompt (from argv, trimmed): \"" + prompt + "\"");
  Logger::info("Steps (from argv): " + std::to_string(steps));
  Logger::info("Temperature: " + std::to_string(temperature));
  Logger::info("Top-K: " + std::to_string(top_k));
  Logger::info("Top-P: " + std::to_string(top_p));

  try {
    tinyllama::TinyLlamaSession session(model_path_or_dir);
    Logger::info("TinyLlamaSession initialized successfully.");

    std::string generated_text =
        session.generate(prompt, steps, temperature, top_k, top_p, "", true);
    std::cout << generated_text << std::endl;
  } catch (const std::exception& e) {
    Logger::error("Error: " + std::string(e.what()));
    return 1;
  }

  return 0;
}