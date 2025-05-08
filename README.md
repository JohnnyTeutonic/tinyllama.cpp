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
# Default: CMake will try to automatically detect CUDA if available 
# and if found, it will enable CUDA support (HAS_CUDA=ON).
cmake ..

# --- Compiling WITH CUDA (Optional) ---
# If CUDA is installed but you want to be explicit or if auto-detection fails:
# cmake .. -DHAS_CUDA=ON
# Ensure your CUDA Toolkit is installed and visible to CMake (e.g., in your PATH).

# --- Compiling WITHOUT CUDA --- 
# If you have CUDA installed but want to force a CPU-only build:
# cmake .. -DHAS_CUDA=OFF

# If you do NOT have CUDA installed, CMake will automatically set HAS_CUDA=OFF.

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

### 4. Other Executables / Command-Line Usage

Besides the web server, you can interact with the models directly via command-line executables. These are typically found in `./build/bin/`.

*   **`tinyllama`**:
    *   **Description**: Command-line interface for chat. Can load models from a SafeTensors model directory (containing `config.json`, `model.safetensors`, `tokenizer.json`) OR by providing a direct path to a `.gguf` model file.
    *   **Usage**: `./build/bin/tinyllama <model_path_or_dir> [prompt] [steps] [temperature]`
        *   `<model_path_or_dir>`: Path to model directory or `.gguf` file. (Default: `data`)
        *   `[prompt]`: The text prompt to send to the model. (Default: "Hello, world!")
        *   `[steps]`: Maximum number of new tokens to generate. (Default: `64`)
        *   `[temperature]`: Sampling temperature. (Default: `0.7`)
    *   **Example (SafeTensors directory)**:
        ```bash
        ./build/bin/tinyllama ./data "Who is the prime minister of Australia?" 32 0.1
        ```
    *   **Example (GGUF file)**:
        ```bash
        ./build/bin/tinyllama ./models/my_model.Q4_K_M.gguf "Explain black holes." 128 0.5
        ```

*   **`tinyllamagguf`**:
    *   **Description**: Command-line interface specifically for loading `.gguf` model files for chat.
    *   **Usage**: `./build/bin/tinyllamagguf <path_to_gguf_file> [prompt] [steps] [temperature]`
        *   `<path_to_gguf_file>`: Direct path to the `.gguf` model file. (Required)
        *   `[prompt]`: The text prompt. (Default: "What is the capital of France?")
        *   `[steps]`: Maximum number of new tokens. (Default: `64`)
        *   `[temperature]`: Sampling temperature. (Default: `0.7`)
    *   **Example**:
        ```bash
        ./build/bin/tinyllamagguf ./models/another_model.Q8_0.gguf "Tell me a story." 256 0.8
        ```
        *(Note: Ensure `tokenizer.json` is present in the same directory as the `.gguf` file for `tinyllamagguf` and when using `.gguf` with `tinyllama`.)*

## PyTorch SafeTensors Inference

For users interested in a Python-based reference or for direct PyTorch inference with SafeTensors models (compatible with Llama 2 / TinyLlama architecture), a dedicated implementation is available in the `pytorch/` directory.

This directory contains:

*   `run_inference.py`: The main script to execute inference using PyTorch.
*   `tinyllama.py`: Contains the PyTorch model definition (e.g., for TinyLlama).
*   `utils.py`: Utility helper functions.
*   `requirements.txt`: Lists the necessary Python packages to run the PyTorch inference scripts. Install these using `pip install -r pytorch/requirements.txt`.
*   `README.md`: A dedicated README within the `pytorch/` directory provides more specific instructions on how to set up and run the PyTorch-based inference.

This can be useful for:
*   Verifying model outputs against a pure PyTorch implementation.
*   Experimenting with the model in a Python environment before using the C++ application.
*   Users who prefer or require a PyTorch-based workflow for SafeTensors models.

Please refer to the `pytorch/README.md` for detailed usage instructions for this PyTorch implementation.

## Project Structure

*   `CMakeLists.txt`: Defines the build process, dependencies, and targets.
*   `*.cpp`, `*.h`, `*.cu`: C++ source, header, and CUDA kernel files.
    *   `main_gguf.cpp`: Main entry point for GGUF model loading and generation tests.
    *   `api.cpp`/`api.h`: Defines the `TinyLlamaSession` class for easier interaction.
    *   `model.cpp`/`model.h`: Core model logic (transformer layers, attention, etc.).
    *   `tokenizer.cpp`/`tokenizer.h`: Tokenizer loading and BPE logic wrapper.
    *   `safetensors_loader.cpp`/`safetensors_loader.h`: Handles loading weights from `.safetensors` files.
    *   `gguf_parser.cpp`/`gguf_parser.h`: Handles loading metadata and weights from `.gguf` files.
    *   `server.cpp`: Implements the HTTP server using `cpp-httplib`.
    *   `cuda_kernels.cu`: Contains CUDA kernels for GPU acceleration.
    *   `logger.cpp`/`logger.h`: Simple logging utility.
    *   `quantization.h`: Contains structures and functions for quantized types (Q4_K, Q6_K, Q8_0 etc.).
*   `www/`: Contains the static files (HTML, CSS, JavaScript) for the web chat UI.
*   `data/` (Example): You need to create this directory and place your model files inside it.

## Future Enhancements
*   Streaming responses in the web UI. 