cmake_minimum_required(VERSION 3.11)
project(TinyLlamaCpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find necessary libraries
find_package(Threads REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

if(CUDA_FOUND)
    message(STATUS "Found CUDA toolkit version: ${CUDA_VERSION}")
    set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 CACHE STRING "CUDA architectures") 
    # set(CMAKE_CUDA_STANDARD 17)
    # set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Set the HAS_CUDA preprocessor definition
    add_compile_definitions(HAS_CUDA)
    message(STATUS "CUDA Found. Defining HAS_CUDA macro.")
else()
    message(WARNING "CUDA Toolkit not found. Building CPU-only version.")
endif()

# Add source files
add_executable(tinyllama 
    main.cpp 
    model.cpp 
    safetensors_loader.cpp 
    tokenizer.cpp 
    logger.cpp 
    prompt.cpp 
    vocab_loader.cpp
    cuda_kernels.cu
)

# Include directories
target_include_directories(tinyllama PRIVATE
    ${nlohmann_json_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(tinyllama PRIVATE
    Threads::Threads
    nlohmann_json::nlohmann_json
    OpenMP::OpenMP_CXX
)

# Conditionally link CUDA libraries
if(CUDA_FOUND)
    target_link_libraries(tinyllama PRIVATE ${CUDA_LIBRARIES})
endif()

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

message(STATUS "Project source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
