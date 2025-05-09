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

int main(int argc, char** argv) {
  
  std::string model_path_or_dir = "data";  
  std::string prompt = "Hello, world!";    
  int steps = 64;                          
  float temperature = 0.7f;                

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
  
  Logger::info("Using model path/directory: " + model_path_or_dir);
  Logger::info("Raw Prompt (from argv, trimmed): \"" + prompt + "\"");
  Logger::info("Steps (from argv): " + std::to_string(steps));
  Logger::info("Temperature Used (currently ignored by API sampling): " +
               std::to_string(temperature));

  try {
    
    tinyllama::TinyLlamaSession session(model_path_or_dir);
    Logger::info("TinyLlamaSession initialized successfully.");

    // Ensure Q: A: format is applied by session.generate for models run via main.cpp
    // The system_prompt argument ("") is unused by api.cpp's generate when apply_q_a_format is true.
    std::string generated_text = session.generate(prompt, steps, temperature, "", true);

    // Output the raw prompt and the generated text
    std::cout << "Prompt: " + prompt << std::endl;
    std::cout << "Generated: " << generated_text << std::endl;

  } catch (const std::exception& e) {
    Logger::error(std::string("An error occurred: ") + e.what());
    
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}