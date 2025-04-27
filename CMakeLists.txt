cmake_minimum_required(VERSION 3.11)
project(TinyLlamaCppWithLibtorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Download and configure libtorch (CPU version)
# Use the latest without-deps version (GAMBLE!)
# set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-without-deps-latest.zip") # LATEST
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.2.2%2Bcpu.zip") # 2.2.2 with deps
# set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-without-deps-2.2.2%2Bcpu.zip") # INVALID
# set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.2.2%2Bcpu.zip") # OLD
FetchContent_Declare(
  libtorch
  URL      ${LIBTORCH_URL}
)
FetchContent_MakeAvailable(libtorch)

# Set Torch_DIR for find_package
set(Torch_DIR ${libtorch_SOURCE_DIR}/share/cmake/Torch)
find_package(Torch REQUIRED)

# Find necessary libraries
find_package(Threads REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenMP REQUIRED)

# Add source files
add_executable(tinyllama main.cpp model.cpp safetensors_loader.cpp tokenizer.cpp logger.cpp prompt.cpp vocab_loader.cpp)

# Include directories
target_include_directories(tinyllama PRIVATE
    ${nlohmann_json_INCLUDE_DIRS}
    # ${libtorch_SOURCE_DIR}/include  # Let find_package handle Torch includes
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(tinyllama PRIVATE
    Threads::Threads
    nlohmann_json::nlohmann_json
    OpenMP::OpenMP_CXX
    "${TORCH_LIBRARIES}" # Revert to using the variable as Torch::torch target wasn't found
    # Torch::torch 
)

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

# Add CMAKE_CURRENT_SOURCE_DIR for local headers
target_include_directories(tinyllama PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Explicitly add include directory for nlohmann_json
target_include_directories(tinyllama PRIVATE ${nlohmann_json_INCLUDE_DIRS})

# If nlohmann_json is not found, provide a hint
if(NOT nlohmann_json_FOUND)
    message(FATAL_ERROR "nlohmann_json not found. You can install it via your package manager or add the single header to your project.")
endif()

# For Windows: avoid conflicting runtime
if (MSVC)
  set_property(TARGET tinyllama PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug$>")
endif()

# Set library output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Print debug info
message(STATUS "Libtorch directory: ${libtorch_SOURCE_DIR}")
message(STATUS "Project source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
