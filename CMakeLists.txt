cmake_minimum_required(VERSION 3.14)
project(tinyllama_inference CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP and set compiler flags
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "Found OpenMP, enabling parallel execution.")
    # Add appropriate compiler flags for OpenMP
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found. Performance will be suboptimal.")
endif()

# Main executable (add all .cpp files here as you modularize)
add_executable(tinyllama main.cpp safetensors_loader.cpp tokenizer.cpp logger.cpp prompt.cpp model.cpp)
# In the future, add more modules as needed.

# nlohmann_json (header-only)
find_package(nlohmann_json 3.2.0 REQUIRED)
# Explicitly add include directory for nlohmann_json
target_include_directories(tinyllama PRIVATE ${nlohmann_json_INCLUDE_DIRS})
# Link library (might be interface library, handles includes too usually)
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

# Add SentencePiece include directory and link library
target_include_directories(tinyllama PRIVATE ${SentencePiece_INCLUDE_DIR})
target_link_libraries(tinyllama PRIVATE ${SentencePiece_LIBRARY})

# Find and link BLAS (e.g., OpenBLAS)
find_package(BLAS REQUIRED)
# Explicitly add include directory for BLAS
if(BLAS_INCLUDE_DIRS)
    target_include_directories(tinyllama PRIVATE ${BLAS_INCLUDE_DIRS})
endif()
target_link_libraries(tinyllama PRIVATE ${BLAS_LIBRARIES})

# Link OpenMP if found (modern CMake often uses target_link_libraries)
# Check if OpenMP::OpenMP_CXX target exists, otherwise use flags (handled above)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(tinyllama PRIVATE OpenMP::OpenMP_CXX)
endif()

# If nlohmann_json is not found, provide a hint
if(NOT nlohmann_json_FOUND)
    message(FATAL_ERROR "nlohmann_json not found. You can install it via your package manager or add the single header to your project.")
endif()

# Optionally, add include directories if you have local copies
# target_include_directories(tinyllama PRIVATE /path/to/json/include) 