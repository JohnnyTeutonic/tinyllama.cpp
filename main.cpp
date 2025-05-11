/**
 * @file main.cpp
 * @brief Command-line interface for TinyLlama text generation.
 *
 * This program provides a command-line interface for generating text using
 * TinyLlama models. It supports both GGUF and SafeTensors model formats.
 * The program automatically applies Q:A formatting to prompts, which is
 * required for proper model responses in both formats.
 *
 * Usage:
 *   tinyllama [model_path] [prompt] [steps] [temperature] [top_k] [top_p] [cpu_layers]
 *
 * Arguments:
 *   model_path: Path to model directory or .gguf file (default: data)
 *   prompt: Input text (default: "Hello, world!")
 *   steps: Number of tokens to generate (default: 64)
 *   temperature: Sampling temperature, lower is more deterministic (default: 0.1)
 *   top_k: Limit sampling to top K tokens (default: 40)
 *   top_p: Limit sampling to top P probability mass (default: 0.9)
 *   cpu_layers: Number of layers to offload to CPU (default: 0)
 *
 * Example:
 *   ./tinyllama data "What is the capital of France?" 64 0.1 40 0.9 0
 *
 * Note:
 *   The program always applies Q:A formatting to prompts ("Q: [prompt]\nA:")
 *   as this is required for proper responses from both GGUF and SafeTensors models.
 */

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
  std::cout << "Usage: tinyllama [model_path] [prompt] [steps] [temperature] [top_k] [top_p] [cpu_layers]\n"
            << "  model_path: Path to model directory or .gguf file (default: data)\n"
            << "  prompt: Input text (default: \"Hello, world!\")\n"
            << "  steps: Number of tokens to generate (default: 64)\n"
            << "  temperature: Sampling temperature, lower is more deterministic (default: 0.7)\n"
            << "  top_k: Limit sampling to top K tokens (default: 40)\n"
            << "  top_p: Limit sampling to top P probability mass (default: 0.9)\n"
            << "  cpu_layers: Number of layers to offload to CPU (default: 0)\n";
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
  int cpu_layers = 0;

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

  if (argc > 7) {
    try {
      cpu_layers = std::stoi(argv[7]);
    } catch (const std::exception& e) {
      Logger::error("Invalid cpu_layers argument: " + std::string(argv[7]) +
                    ". Using default: " + std::to_string(cpu_layers));
    }
  }

  Logger::info("Using model path/directory: " + model_path_or_dir);
  Logger::info("Raw Prompt (from argv, trimmed): \"" + prompt + "\"");
  Logger::info("Steps (from argv): " + std::to_string(steps));
  Logger::info("Temperature: " + std::to_string(temperature));
  Logger::info("Top-K: " + std::to_string(top_k));
  Logger::info("Top-P: " + std::to_string(top_p));
  Logger::info("CPU Layers: " + std::to_string(cpu_layers));

  try {
    tinyllama::TinyLlamaSession session(model_path_or_dir, cpu_layers);
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