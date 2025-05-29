cmake_minimum_required(VERSION 3.15)
project(TinyLlamaCpp LANGUAGES CXX)

# Set CMake policies
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW) # DOWNLOAD_EXTRACT_TIMESTAMP
endif()
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW) # find_package uses upper-case <PACKAGENAME>_ROOT variables
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json not found via find_package. Fetching with FetchContent.")
  FetchContent_Declare(
    nlohmann_json
    URL        https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    URL_HASH   SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406 # Verified hash
  )
  FetchContent_MakeAvailable(nlohmann_json)
  set(NLOHMANN_JSON_INCLUDE_DIRS ${nlohmann_json_SOURCE_DIR}/include CACHE INTERNAL "nlohmann_json include directory from FetchedContent source")
  message(STATUS "nlohmann_json fetched via URL. Include directory: ${NLOHMANN_JSON_INCLUDE_DIRS}")
else()
  message(STATUS "Found nlohmann_json system-wide or via CMake package config. Version: ${nlohmann_json_VERSION}")
  set(NLOHMANN_JSON_INCLUDE_DIRS ${nlohmann_json_INCLUDE_DIRS} CACHE INTERNAL "nlohmann_json include directory from find_package")
endif()

FetchContent_Declare(
  cpp-httplib
  URL        https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.15.3.tar.gz
  URL_HASH   SHA256=2121bbf38871bb2aafb5f7f2b9b94705366170909f434428352187cb0216124e # Verified hash for v0.15.3
)
FetchContent_MakeAvailable(cpp-httplib)

set(HTTPLIB_HEADER_PATH "${cpp-httplib_SOURCE_DIR}/httplib.h")
message(STATUS "Attempting to patch httplib.h at: ${HTTPLIB_HEADER_PATH} to disable Brotli.")

if(EXISTS "${HTTPLIB_HEADER_PATH}")
    file(READ "${HTTPLIB_HEADER_PATH}" HTTPLIB_CONTENT_ORIGINAL)
    set(HTTPLIB_CONTENT "${HTTPLIB_CONTENT_ORIGINAL}") # Work on a copy

    string(REPLACE "// #include <brotli/decode.h> // Patched by TinyLlamaCpp CMakeLists.txt" "#include <brotli/decode.h>" HTTPLIB_CONTENT "${HTTPLIB_CONTENT}")
    string(REPLACE "// #include <brotli/encode.h> // Patched by TinyLlamaCpp CMakeLists.txt" "#include <brotli/encode.h>" HTTPLIB_CONTENT "${HTTPLIB_CONTENT}")

    string(REPLACE "#ifdef CPPHTTPLIB_BROTLI_SUPPORT" "#if 0 // Patched by TinyLlamaCpp CMakeLists.txt (was #ifdef CPPHTTPLIB_BROTLI_SUPPORT)" HTTPLIB_CONTENT "${HTTPLIB_CONTENT}")
    
    if(NOT "${HTTPLIB_CONTENT}" STREQUAL "${HTTPLIB_CONTENT_ORIGINAL}")
        file(WRITE "${HTTPLIB_HEADER_PATH}" "${HTTPLIB_CONTENT}")
        message(STATUS "Patched ${HTTPLIB_HEADER_PATH} to disable Brotli code block via '#if 0'.")
    else()
        message(STATUS "No changes made to ${HTTPLIB_HEADER_PATH} during Brotli patching (possibly already patched or string not found).")
    endif()
else()
    message(WARNING "Could not find httplib.h at ${HTTPLIB_HEADER_PATH} for patching. Brotli issues may persist.")
endif()
FetchContent_Declare(
  pybind11
  URL        https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.tar.gz
  URL_HASH   SHA256=bf8f242abd1abcd375d516a7067490fb71abd79519a282d22b6e4d19282185a7 # Verified hash for v2.12.0 from build error
)
FetchContent_MakeAvailable(pybind11)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

find_package(Boost 1.70.0 REQUIRED COMPONENTS regex)
if(Boost_FOUND)
    message(STATUS "Found Boost version: ${Boost_VERSION_STRING}")
    message(STATUS "Boost include directories: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost regex library: ${Boost_REGEX_LIBRARY}")
else()
    message(FATAL_ERROR "Boost (version 1.70.0 or newer) with the regex component is required but was not found.")
endif()

OPTION(HAS_CUDA "Enable CUDA support (requires CUDA Toolkit and cuBLAS)" ON) # Default to ON, user can override with -DHAS_CUDA=OFF

set(CUDA_ENABLED FALSE) # Default to disabled
set(CUDA_SOURCES "")   # Default to no CUDA sources

if(HAS_CUDA)
    enable_language(CUDA)
    find_package(CUDA)
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
    model_config.cpp
    kv_cache.cpp
    weight_management.cpp
    cpu_attention.cpp
    model_utils.cpp
    gpu_initialization.cpp
    gpu_memory_optimization.cpp
    cpu_batch_processor.cpp
    utils.cpp
    safetensors_loader.cpp 
    tokenizer.cpp 
    logger.cpp 
    api.cpp
    gguf_parser.cpp
    quantization.cpp
    model_constants.h
    ${CUDA_SOURCES} # Conditionally include CUDA source file
)

set_target_properties(tinyllama_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set properties for the core library
target_include_directories(tinyllama_core PUBLIC
    ${NLOHMANN_JSON_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    # ${CUDA_INCLUDE_DIRS}
    ${cpp-httplib_SOURCE_DIR}/
    ${Boost_INCLUDE_DIRS} # Add Boost include directory
)

target_link_libraries(tinyllama_core PRIVATE
    Threads::Threads
    nlohmann_json::nlohmann_json
    OpenMP::OpenMP_CXX
    Boost::regex # Add Boost regex library target
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

target_link_libraries(tinyllama PRIVATE tinyllama_core)

if(NOT SKBUILD)
    add_executable(tinyllama_server
        server.cpp
    )
    target_link_libraries(tinyllama_server PRIVATE tinyllama_core httplib)

    target_compile_definitions(tinyllama_server PRIVATE CPPHTTPLIB_BROTLI_SUPPORT=0)

    find_package(OpenSSL QUIET) # Try to find OpenSSL for cpp-httplib to potentially use
    if(OPENSSL_FOUND AND TARGET OpenSSL::SSL AND TARGET OpenSSL::Crypto)
        message(STATUS "Found OpenSSL. Enabling SSL support in cpp-httplib for tinyllama_server.")
        target_compile_definitions(tinyllama_server PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
        target_include_directories(tinyllama_server PRIVATE ${OpenSSL_INCLUDE_DIRS})
    else()
        message(STATUS "OpenSSL not found for tinyllama_server. HTTPS support will be disabled.")
    endif()
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

pybind11_add_module(tinyllama_bindings
    bindings.cpp
)

target_link_libraries(tinyllama_bindings PRIVATE tinyllama_core)

# For Windows: ensure the Python extension has the .pyd extension
if (MSVC)
    set_target_properties(tinyllama_bindings PROPERTIES SUFFIX ".pyd")
endif()

install(TARGETS tinyllama_bindings
    LIBRARY DESTINATION .
    ARCHIVE DESTINATION .
)


if(OpenMP_FOUND)
    message(STATUS "Found OpenMP, enabling parallel execution.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found. Performance will be suboptimal.")
endif()

if (MSVC)
  foreach(target tinyllama tinyllama_server tinyllama_core)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDebug") # /MTd
    else()
      set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded") # /MT
    endif()

    target_compile_options(${target} PRIVATE "/openmp:llvm" "/DNOMINMAX" "/D_USE_MATH_DEFINES")
    target_compile_definitions(${target} PRIVATE NOMINMAX _USE_MATH_DEFINES)
  endforeach()
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

message(STATUS "Project source directory: ${CMAKE_CURRENT_SOURCE_DIR}")