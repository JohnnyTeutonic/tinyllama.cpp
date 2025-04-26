cmake_minimum_required(VERSION 3.16)
project(TinyLlamaCPP CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find necessary libraries
find_package(Threads REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED) # Find BLAS library

# SentencePiece requires manual finding or pkg-config
# Assuming SentencePiece is installed system-wide or in a known prefix
# Adjust paths if necessary
find_library(SENTENCEPIECE_LIBRARY NAMES sentencepiece)
find_path(SENTENCEPIECE_INCLUDE_DIR NAMES sentencepiece_processor.h)

if(NOT SENTENCEPIECE_LIBRARY OR NOT SENTENCEPIECE_INCLUDE_DIR)
    message(FATAL_ERROR "SentencePiece library or headers not found. Please install SentencePiece or set SENTENCEPIECE_LIBRARY and SENTENCEPIECE_INCLUDE_DIR.")
endif()

# Add source files
add_executable(tinyllama main.cpp model.cpp safetensors_loader.cpp tokenizer.cpp logger.cpp prompt.cpp)

# Include directories
target_include_directories(tinyllama PRIVATE
    ${nlohmann_json_INCLUDE_DIRS}
    ${SENTENCEPIECE_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(tinyllama PRIVATE
    Threads::Threads
    nlohmann_json::nlohmann_json
    ${SENTENCEPIECE_LIBRARY}
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES} # Link against BLAS
)

# Set compiler flags if needed (e.g., for optimizations)
# target_compile_options(tinyllama PRIVATE -O3)

# Install target (optional)
# install(TARGETS tinyllama DESTINATION bin)

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

# In the future, add more modules as needed.

# Add CMAKE_CURRENT_SOURCE_DIR for local headers
target_include_directories(tinyllama PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Explicitly add include directory for nlohmann_json
target_include_directories(tinyllama PRIVATE ${nlohmann_json_INCLUDE_DIRS})

# If nlohmann_json is not found, provide a hint
if(NOT nlohmann_json_FOUND)
    message(FATAL_ERROR "nlohmann_json not found. You can install it via your package manager or add the single header to your project.")
endif()

# Optionally, add include directories if you have local copies
# target_include_directories(tinyllama PRIVATE /path/to/json/include) 

# --- START: Test executable for matvec_bf16_f32 ---
add_executable(test_matvec test_matvec.cpp safetensors_loader.cpp logger.cpp model.cpp)

# Add CMAKE_CURRENT_SOURCE_DIR for local headers
target_include_directories(test_matvec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link BLAS (not strictly needed for matvec_bf16_f32, but model.cpp might depend on it)
if(BLAS_INCLUDE_DIRS)
    target_include_directories(test_matvec PRIVATE ${BLAS_INCLUDE_DIRS})
endif()
target_link_libraries(test_matvec PRIVATE ${BLAS_LIBRARIES})

# Link OpenMP (matvec_bf16_f32 uses OpenMP pragmas)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(test_matvec PRIVATE OpenMP::OpenMP_CXX)
endif()
# --- END: Test executable --- 