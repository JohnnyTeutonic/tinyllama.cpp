cmake_minimum_required(VERSION 3.11)
project(TinyLlamaCpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent module
include(FetchContent)

# --- Fetch cpp-httplib --- 
FetchContent_Declare(
  cpp-httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG        v0.15.3
)
FetchContent_MakeAvailable(cpp-httplib)
# --------------------------

# Find necessary system libraries
find_package(Threads REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenSSL)
find_package(Boost REQUIRED COMPONENTS regex)

# Handle CUDA
find_package(CUDA)
# Modern CMake approach to find specific CUDA components
if(CUDA_FOUND)
    find_package(CUDAToolkit COMPONENTS cublas) 
endif()

# --- USER OPTION TO CONTROL CUDA --- 
OPTION(HAS_CUDA "Enable CUDA support (requires CUDA Toolkit and cuBLAS)" ON) # Default to ON, user can override with -DHAS_CUDA=OFF

set(CUDA_ENABLED FALSE) # Default to disabled
set(CUDA_SOURCES "")   # Default to no CUDA sources

if(HAS_CUDA)
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
    quantization.cpp # ADDED QUANTIZATION SOURCE
    ${CUDA_SOURCES} # Conditionally include CUDA source file
)

# Set properties for the core library
target_include_directories(tinyllama_core PUBLIC # Use PUBLIC so executables inherit includes
    ${nlohmann_json_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${cpp-httplib_SOURCE_DIR} # Add include directory for httplib
    ${Boost_INCLUDE_DIRS}
)

# Link system/other libraries
target_link_libraries(tinyllama_core PRIVATE # Dependencies needed to BUILD the library
    Threads::Threads
    nlohmann_json::nlohmann_json
    OpenMP::OpenMP_CXX
)

# Conditionally add CUDA definition and link CUDA libs for the library
if(CUDA_ENABLED)
    target_compile_definitions(tinyllama_core PUBLIC HAS_CUDA)
    # Also add CUDA include dirs if needed specifically here, though PUBLIC in previous block is okay
    target_include_directories(tinyllama_core PRIVATE ${CUDA_INCLUDE_DIRS}) # Add includes needed to *build* the lib if CUDA is on
    # Link core CUDA runtime and the cuBLAS library
    target_link_libraries(tinyllama_core PRIVATE ${CUDA_LIBRARIES} CUDA::cublas)
endif()

# --- Original Executable (Now uses the library) ---
add_executable(tinyllama 
    main.cpp 
)

# Link tinyllama executable against the core library
target_link_libraries(tinyllama PRIVATE tinyllama_core)

# --- New API Example Executable ---
add_executable(tinyllama_api_example
    main_api_example.cpp # File exists now
)
# Link the example against the core library
target_link_libraries(tinyllama_api_example PRIVATE tinyllama_core)

# --- New Chat Server Executable ---
add_executable(tinyllama_server
    server.cpp
)

target_link_libraries(tinyllama_server PRIVATE tinyllama_core)

# --- New GGUF Loader Executable ---
add_executable(tinyllamagguf
    main_gguf.cpp
)

# Link GGUF loader against the core library (inherits CUDA/OpenMP/etc links)
target_link_libraries(tinyllamagguf PRIVATE tinyllama_core)

# --- ADDED: Tokenizer Test Executable ---
add_executable(test_tokenizer
    test_tokenizer.cpp
)
# Link the test executable against the core library
target_link_libraries(test_tokenizer PRIVATE tinyllama_core Boost::regex)
# --- END ADDED ---

# Link OpenSSL if found (needed by httplib if SSL support is enabled)
if(OPENSSL_FOUND)
    message(STATUS "Found OpenSSL, linking ssl and crypto libraries to server.")
    target_link_libraries(tinyllama_server PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_link_libraries(tinyllama_core PRIVATE OpenSSL::Crypto)
else()
    message(WARNING "OpenSSL not found. HTTPS support in server will likely fail to link if enabled via CPPHTTPLIB_OPENSSL_SUPPORT.")
endif()

# --- Common Settings --- 

# Add OpenMP flags (applies to all targets using CXX)
if(OpenMP_FOUND)
    message(STATUS "Found OpenMP, enabling parallel execution.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found. Performance will be suboptimal.")
endif()

if(NOT nlohmann_json_FOUND)
    message(FATAL_ERROR "nlohmann_json not found. You can install it via your package manager or add the single header to your project.")
endif()

# For Windows: avoid conflicting runtime (Apply to all targets)
if (MSVC)
  # Need to apply to tinyllama_core as well
  foreach(target tinyllama tinyllama_server tinyllama_core)
    set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug$>")
  endforeach()
endif()

# Set library output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

message(STATUS "Project source directory: ${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(dequantize_roundtrip dequantize_roundtrip.cpp)
target_link_libraries(dequantize_roundtrip PRIVATE tinyllama_core)

add_executable(matvec_test matvec_test.cpp)
target_link_libraries(matvec_test PRIVATE tinyllama_core)

add_executable(compare_quant_fp16 compare_quant_fp16.cpp)
target_link_libraries(compare_quant_fp16 PRIVATE tinyllama_core)