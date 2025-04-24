cmake_minimum_required(VERSION 3.14)
project(tinyllama_inference CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main executable (add all .cpp files here as you modularize)
add_executable(tinyllama main.cpp safetensors_loader.cpp tokenizer.cpp logger.cpp)
# In the future, add model.cpp, prompt.cpp, etc.

# nlohmann_json (header-only)
find_package(nlohmann_json 3.2.0 REQUIRED)
target_link_libraries(tinyllama PRIVATE nlohmann_json::nlohmann_json)

# SentencePiece discovery (manual)
# You can set these variables manually if needed:
#   -DSentencePiece_INCLUDE_DIR=/path/to/sentencepiece/include
#   -DSentencePiece_LIBRARY=/path/to/sentencepiece/libs/libsentencepiece.so
find_path(SentencePiece_INCLUDE_DIR sentencepiece_processor.h)
find_library(SentencePiece_LIBRARY sentencepiece)

if(NOT SentencePiece_INCLUDE_DIR OR NOT SentencePiece_LIBRARY)
    message(FATAL_ERROR "Could not find SentencePiece. Please install it or set SentencePiece_INCLUDE_DIR and SentencePiece_LIBRARY.")
endif()

# Link SentencePiece
target_include_directories(tinyllama PRIVATE ${SentencePiece_INCLUDE_DIR})
target_link_libraries(tinyllama PRIVATE ${SentencePiece_LIBRARY})

# If nlohmann_json is not found, provide a hint
if(NOT nlohmann_json_FOUND)
    message(FATAL_ERROR "nlohmann_json not found. You can install it via your package manager or add the single header to your project.")
endif()

# Optionally, add include directories if you have local copies
# target_include_directories(tinyllama PRIVATE /path/to/json/include) 