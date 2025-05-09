cmake_minimum_required(VERSION 3.11)
project(TinyLlamaCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent module
include(FetchContent)

# --- Fetch nlohmann/json --- 
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json not found via find_package. Fetching with FetchContent.")
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
  )
  FetchContent_MakeAvailable(nlohmann_json)
  set(NLOHMANN_JSON_INCLUDE_DIRS ${nlohmann_json_SOURCE_DIR}/include CACHE INTERNAL "nlohmann_json include directory")
  message(STATUS "nlohmann_json fetched. Include directory: ${NLOHMANN_JSON_INCLUDE_DIRS}")
else()
  message(STATUS "Found nlohmann_json system-wide or via CMake package config. Version: ${nlohmann_json_VERSION}")
  set(NLOHMANN_JSON_INCLUDE_DIRS ${nlohmann_json_INCLUDE_DIRS} CACHE INTERNAL "nlohmann_json include directory")
endif()

# --- Fetch cpp-httplib --- 
FetchContent_Declare(
  cpp-httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG        v0.15.3
)
FetchContent_MakeAvailable(cpp-httplib)

# Find necessary system libraries
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Handle CUDA
OPTION(HAS_CUDA "Enable CUDA support (requires CUDA Toolkit and cuBLAS)" ON) # Default to ON, user can override with -DHAS_CUDA=OFF

set(CUDA_ENABLED FALSE) # Default to disabled
set(CUDA_SOURCES "")   # Default to no CUDA sources

if(HAS_CUDA)
    enable_language(CUDA)
    find_package(CUDA)
    # Modern CMake approach to find specific CUDA components
    if(CUDA_FOUND)
        find_package(CUDAToolkit COMPONENTS cublas) 
    endif()
    
    if(CUDA_FOUND AND CUDAToolkit_FOUND)
        message(STATUS "CUDA toolkit and cuBLAS found, and HAS_CUDA option is ON.")
        message(STATUS "Found CUDA toolkit version: ${CUDA_VERSION}")
        set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 CACHE STRING "CUDA architectures") 
        message(STATUS "CUDA Enabled. Will define HAS_CUDA for tinyllama_core library.")
        set(CUDA_ENABLED TRUE)
        set(CUDA_SOURCES cuda_kernels.cu) # Only include CUDA source if enabled
    else()
        message(WARNING "HAS_CUDA option is ON, but CUDA Toolkit or cuBLAS not found. Disabling CUDA support.")
        set(HAS_CUDA FALSE) # Force the option OFF if prerequisites are missing
    endif()
else()
    message(STATUS "HAS_CUDA option is OFF. Building CPU-only version.")
endif()

# --- Core Library Definition ---
add_library(tinyllama_core STATIC
    model.cpp 
    safetensors_loader.cpp 
    tokenizer.cpp 
    logger.cpp 
    api.cpp # Added API implementation
    vocab_loader.cpp
    gguf_parser.cpp
    quantization.cpp
    model_constants.h
    ${CUDA_SOURCES} # Conditionally include CUDA source file
)

# Set properties for the core library
target_include_directories(tinyllama_core PUBLIC
    ${NLOHMANN_JSON_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    # ${CUDA_INCLUDE_DIRS}
    ${cpp-httplib_SOURCE_DIR}/
)

target_link_libraries(tinyllama_core PRIVATE
    Threads::Threads
    nlohmann_json::nlohmann_json
    OpenMP::OpenMP_CXX
)

# Conditionally add CUDA definition and link CUDA libs for the library
if(CUDA_ENABLED)
    target_compile_definitions(tinyllama_core PUBLIC HAS_CUDA)
    target_include_directories(tinyllama_core PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(tinyllama_core PRIVATE ${CUDA_LIBRARIES} CUDA::cublas)
endif()

add_executable(tinyllama 
    main.cpp 
)

# Link tinyllama executable against the core library
target_link_libraries(tinyllama PRIVATE tinyllama_core)

add_executable(tinyllama_server
    server.cpp
)

target_link_libraries(tinyllama_server PRIVATE tinyllama_core httplib)

# Link OpenSSL if found (needed by httplib if SSL support is enabled)
find_package(OpenSSL QUIET) # Try to find OpenSSL for cpp-httplib to potentially use
if(OPENSSL_FOUND AND TARGET OpenSSL::SSL AND TARGET OpenSSL::Crypto)
    message(STATUS "Found OpenSSL. Enabling SSL support in cpp-httplib.")
    message(STATUS "OpenSSL Include Directories: ${OpenSSL_INCLUDE_DIRS}")
    target_compile_definitions(tinyllama_server PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    # Add OpenSSL include directories
    target_include_directories(tinyllama_server PRIVATE ${OpenSSL_INCLUDE_DIRS})
    # target_link_libraries(tinyllama_server PRIVATE OpenSSL::SSL OpenSSL::Crypto)
else()
    message(STATUS "OpenSSL not found or targets not available. HTTPS support in server will be disabled.")
endif()

# Add OpenMP flags (applies to all targets using CXX)
if(OpenMP_FOUND)
    message(STATUS "Found OpenMP, enabling parallel execution.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found. Performance will be suboptimal.")
endif()

# For Windows: avoid conflicting runtime (Apply to all targets)
if (MSVC)
  # Need to apply to tinyllama_core as well
  foreach(target tinyllama tinyllama_server tinyllama_core)
    set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug$>")
    target_compile_options(${target} PRIVATE "/openmp:llvm" "/DNOMINMAX" "/D_USE_MATH_DEFINES")
    target_compile_definitions(${target} PRIVATE NOMINMAX _USE_MATH_DEFINES)
  endforeach()
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

message(STATUS "Project source directory: ${CMAKE_CURRENT_SOURCE_DIR}")