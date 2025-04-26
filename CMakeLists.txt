cmake_minimum_required(VERSION 3.11)
project(TinyLlamaCppWithLibtorch)

include(FetchContent)

# Download and configure libtorch (CPU version)
# Use the latest without-deps version (GAMBLE!)
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-without-deps-latest.zip") # LATEST
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
# find_package(PkgConfig REQUIRED) # No longer needed for deps
# find_package(BLAS REQUIRED) # No longer needed

# Use find_package for SentencePiece for robustness (or FetchContent)
# find_package(SentencePiece REQUIRED) # Keep this commented or remove

# Force shared library build for SentencePiece - REMOVE THIS
# set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)

# --- Add Protobuf Dependency (Needed by SentencePiece) using FetchContent ---
message(STATUS "Setting up Protobuf dependency (FetchContent)...")
FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v21.12 # Use specific tag compatible with SP v0.2.0
  GIT_SHALLOW    TRUE
)
# Disable protobuf tests to speed up build
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Disable Protobuf tests" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "Build static Protobuf" FORCE)
FetchContent_MakeAvailable(protobuf)
message(STATUS "Protobuf available via FetchContent.")
# --- End Protobuf Dependency ---

# --- Configure and Add Submodules ---
# Set options BEFORE adding the subdirectories
# set(protobuf_BUILD_TESTS OFF CACHE BOOL "Disable Protobuf tests" FORCE) # Moved to FetchContent
# set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "Build static Protobuf" FORCE) # Moved to FetchContent
set(SPM_ENABLE_SHARED OFF CACHE BOOL "Build static SentencePiece" FORCE)

# Add the submodules' source directories to the build
# add_subdirectory(deps/protobuf) # REMOVED - Use FetchContent instead
add_subdirectory(deps/sentencepiece)

# message(STATUS "Added Protobuf and SentencePiece submodules to build.")
message(STATUS "Added SentencePiece submodule to build.")

# --- Set path for locally built dependencies AND pkg-config --- REMOVED ---
# ...
# --- Find Protobuf --- REMOVED ---
# ...
# --- Find SentencePiece --- REMOVED ---
# ...
# --- Fetch SentencePiece using FetchContent --- REMOVED ---
# ...

# Check if dependencies were found/fetched successfully
# The pkg_check_modules commands handle this implicitly

# Add source files
add_executable(tinyllama main.cpp model.cpp safetensors_loader.cpp tokenizer.cpp logger.cpp prompt.cpp)

# Include directories
# Includes from FetchContent (Protobuf) and add_subdirectory (SentencePiece)
# might be added automatically via target_link_libraries.
# Add manually if needed.
target_include_directories(tinyllama PRIVATE
    ${nlohmann_json_INCLUDE_DIRS}
    # deps/sentencepiece/src # Add if needed
    # ${protobuf_SOURCE_DIR}/src # Add if needed
    ${libtorch_SOURCE_DIR}/include
)

# Link libraries
# Link against targets from FetchContent (protobuf) and add_subdirectory (sentencepiece)
target_link_libraries(tinyllama PRIVATE
    Threads::Threads
    nlohmann_json::nlohmann_json
    sentencepiece # Target from sentencepiece submodule
    protobuf::libprotobuf # Target from protobuf FetchContent
    OpenMP::OpenMP_CXX
    "${TORCH_LIBRARIES}"
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

# For Windows: avoid conflicting runtime
if (MSVC)
  set_property(TARGET tinyllama PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug$>")
endif()

# Remove redundant linking lines at the end if they exist
# target_include_directories(tinyllama PRIVATE ${SentencePiece_INCLUDE_DIRS})
# target_link_libraries(tinyllama PRIVATE ${SentencePiece_LIBRARIES})
