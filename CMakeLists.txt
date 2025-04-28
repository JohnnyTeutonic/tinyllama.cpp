cmake_minimum_required(VERSION 3.11)
project(TinyLlamaCpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find necessary libraries
find_package(Threads REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenMP REQUIRED)

# Handle CUDA
find_package(CUDA)
set(CUDA_SOURCES cuda_kernels.cu) # Define CUDA source file
if(CUDA_FOUND)
    message(STATUS "Found CUDA toolkit version: ${CUDA_VERSION}")
    set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 CACHE STRING "CUDA architectures") 
    message(STATUS "CUDA Found. Will define HAS_CUDA for tinyllama_core library.")
    set(CUDA_ENABLED TRUE)
else()
    message(WARNING "CUDA Toolkit not found. Building CPU-only version.")
    set(CUDA_ENABLED FALSE)
endif()

# --- Core Library Definition ---
add_library(tinyllama_core STATIC
    model.cpp 
    safetensors_loader.cpp 
    tokenizer.cpp 
    logger.cpp 
    api.cpp # Added API implementation
    vocab_loader.cpp
    ${CUDA_SOURCES} # Include CUDA source file here
)

# Set properties for the core library
target_include_directories(tinyllama_core PUBLIC # Use PUBLIC so executables inherit includes
    ${nlohmann_json_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(tinyllama_core PRIVATE # Dependencies needed to BUILD the library
    Threads::Threads
    nlohmann_json::nlohmann_json
    OpenMP::OpenMP_CXX
)

# Conditionally add CUDA definition and link CUDA libs for the library
if(CUDA_ENABLED)
    target_compile_definitions(tinyllama_core PRIVATE HAS_CUDA)
    target_link_libraries(tinyllama_core PRIVATE ${CUDA_LIBRARIES})
endif()

# --- Original Executable (Now uses the library) ---
add_executable(tinyllama 
    main.cpp 
    prompt.cpp # Keep prompt helper with main for now?
)

# Link tinyllama executable against the core library
target_link_libraries(tinyllama PRIVATE tinyllama_core)

# --- New API Example Executable ---
add_executable(tinyllama_api_example
    main_api_example.cpp # Will be created next
)

# Link the example against the core library
target_link_libraries(tinyllama_api_example PRIVATE tinyllama_core)

# --- Common Settings --- 

# Add OpenMP flags (applies to all targets using CXX)
if(OpenMP_FOUND)
    message(STATUS "Found OpenMP, enabling parallel execution.")
    # Add appropriate compiler flags for OpenMP - Apply globally or per target?
    # Applying globally for simplicity here
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found. Performance will be suboptimal.")
endif()

# If nlohmann_json is not found, provide a hint
if(NOT nlohmann_json_FOUND)
    message(FATAL_ERROR "nlohmann_json not found. You can install it via your package manager or add the single header to your project.")
endif()

# For Windows: avoid conflicting runtime (Apply to all targets)
if (MSVC)
  foreach(target tinyllama tinyllama_api_example tinyllama_core)
    set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug$>")
  endforeach()
endif()

# Set library output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

message(STATUS "Project source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
