# TinyLlama.cpp - Minimal C++ Chat Inference

## Purpose

This project provides a streamlined, end-to-end C++ inference pipeline for running TinyLlama-based models for chat applications. The primary goal is to achieve this with minimal dependencies, allowing for deployment without requiring Python or large frameworks like LibTorch at runtime (unless using the optional CUDA backend which requires the CUDA Toolkit).

It includes a simple web server and UI for interactive chatting.

## Features

*   Pure C++ inference core (CPU-based).
*   Optional CUDA backend for GPU acceleration.
*   Loads model configuration (`config.json`), tokenizer (`tokenizer.json`), and weights (`model.safetensors`).
*   Built-in web server (`cpp-httplib`) for easy interaction via a web UI.
*   Minimal external dependencies managed via CMake.
*   Cross-platform (tested on Linux, Windows - requires C++17 compiler).

## Dependencies

### C++ Runtime Dependencies

These are needed to **build and run** the C++ application:

1.  **CMake (>= 3.11):** Used for building the project.
2.  **C++17 Compliant Compiler:** Such as g++, Clang, or MSVC.
3.  **nlohmann/json:** For parsing JSON configuration files. (Fetched automatically by CMake).
4.  **cpp-httplib:** For the web server backend. (Fetched automatically by CMake).
5.  **OpenMP (Optional):** For multi-threading CPU acceleration. CMake will try to find it; performance will be lower without it.
6.  **CUDA Toolkit (Optional):** Required **only** if you want GPU-accelerated inference. CMake will detect it if available and build the CUDA kernels. You'll need a compatible NVIDIA GPU and drivers.
7.  **OpenSSL (Optional):** Required by `cpp-httplib` if HTTPS support is needed (currently not explicitly used, but linked if found).

### Python Setup Dependencies

These are needed **only once** to **download and prepare** the model files (`config.json`, `tokenizer.json`, `model.safetensors`) from sources like Hugging Face. They are **NOT** needed to build or run the C++ application itself.

1.  **Python (>= 3.8)**
2.  **PyTorch (`torch`)**
3.  **Hugging Face `transformers`**
4.  **`safetensors`**
5.  **`sentencepiece`** (often required by the tokenizer)

You would typically use a Python script to load a Hugging Face model and tokenizer and then save the configuration (`config.json`), tokenizer data (`tokenizer.json`), and weights (`model.safetensors`) into a local directory.

## Getting Started

### 1. Obtain Model Files

You need three files from a compatible TinyLlama model (e.g., from Hugging Face):

*   `config.json`
*   `tokenizer.json`
*   `model.safetensors`

Place these three files into a directory, for example, named `data/`.

*(Note: This project expects BF16 weights in `model.safetensors`. Ensure your conversion script saves them in this format if converting from another source).*

### 2. Build the C++ Application

Use CMake to build the project:

```bash
# Clone the repository (if you haven't already)
# git clone <repository-url>
# cd tinyllama.cpp

# Create a build directory
mkdir build
cd build

# Configure with CMake
# (CMake should automatically detect CUDA if available)
cmake ..

# Build the executables
# On Linux/macOS:
make -j$(nproc) 
# On Windows (using MSVC, for example, from a Developer Command Prompt):
# cmake --build . --config Release 
# Or open the generated solution file in Visual Studio.

```

This will create several executables in the `build/bin/` directory (or directly in `build/` or `build/Release/` depending on your system and generator), including `tinyllama_server`.

### 3. Run the Chat Server (Primary Example)

The main way to use this project is via the web server:

```bash
# Navigate back to the project root or ensure paths are correct
# Run the server, pointing it to your model data directory
./build/bin/tinyllama_server ./data 

# Example if executable is directly in build:
# ./build/tinyllama_server ./data

# Example on Windows Release build:
# ./build/Release/tinyllama_server.exe ./data
```

*   Replace `./data` with the actual path to the directory containing your `config.json`, `tokenizer.json`, and `model.safetensors`.
*   The server will start, load the model, and listen on `http://localhost:8080` by default.
*   Open your web browser and navigate to `http://localhost:8080`.
*   You should see a chat interface where you can interact with the model.

### 4. Other Executables (Optional)

*   **`tinyllama`:** The original command-line interface. Less sophisticated than the web UI.
    ```bash
    ./build/bin/tinyllama ./data 
    ```
*   **`tinyllama_api_example`:** A minimal example showing how to use the `TinyLlamaSession` C++ API programmatically (see `main_api_example.cpp`).
    ```bash
    ./build/bin/tinyllama_api_example ./data 
    ```

## Project Structure

*   `CMakeLists.txt`: Defines the build process, dependencies, and targets.
*   `*.cpp`, `*.h`, `*.cu`: C++ source, header, and CUDA kernel files.
    *   `api.cpp`/`api.h`: Defines the `TinyLlamaSession` class for easier interaction.
    *   `model.cpp`/`model.h`: Core model logic (transformer layers, attention, etc.).
    *   `tokenizer.cpp`/`tokenizer.h`: Tokenizer loading and BPE logic wrapper.
    *   `safetensors_loader.cpp`/`safetensors_loader.h`: Handles loading weights.
    *   `server.cpp`: Implements the HTTP server using `cpp-httplib`.
    *   `cuda_kernels.cu`: Contains CUDA kernels for GPU acceleration.
    *   `logger.cpp`/`logger.h`: Simple logging utility.
*   `www/`: Contains the static files (HTML, CSS, JavaScript) for the web chat UI.
*   `data/` (Example): You need to create this directory and place your model files inside it.

## Future Enhancements

*   Support for other model weight formats like GGUF.
*   More sophisticated sampling methods.
*   Streaming responses in the web UI. 