# TinyLlama.cpp - A C++ Chat Inference
#### Author: Jonathan Reich

This codebase supports inference for Llama 2 architecture models (including TinyLlama variants) using Safetensors model format as well as GGUF format.

The GGUF format support includes loading models with various tensor types such as BF16, FP16, FP32, and Q8_0. Nominal support for Q4_K and Q6_K quantization types is also present, though Q8_0 is the most extensively tested quantized format in this project.

## Purpose

This project provides a streamlined, end-to-end C++ inference pipeline for running Llama-2 models for chat applications. The primary goal is to achieve this with minimal dependencies, allowing for deployment without requiring Python or large frameworks like LibTorch at runtime (unless using the optional CUDA backend which requires the CUDA Toolkit).

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
3.  **nlohmann/json:** For parsing JSON configuration files. (Fetched automatically by CMake if not found system-wide).
4.  **cpp-httplib:** For the web server backend. (Fetched automatically by CMake).
5.  **OpenMP (Optional):** For multi-threading CPU acceleration. CMake will try to find it; performance will be lower without it.
6.  **CUDA Toolkit (Optional):** Required **only** if you want GPU-accelerated inference. CMake will detect it if available and build the CUDA kernels. You'll need a compatible NVIDIA GPU and drivers.
7.  **Boost.Regex (Optional):** Was required only for the now-removed `test_tokenizer` executable. No longer needed.

#### Installing C++ Dependencies on Linux (Debian/Ubuntu Example)

This section provides example commands for installing the necessary C++ dependencies on Debian-based Linux distributions like Ubuntu. Package names and commands may vary for other distributions (e.g., Fedora, Arch Linux, CentOS).

```bash
# 1. Update package lists
sudo apt update

# 2. Essential build tools (includes g++ compiler) and CMake
sudo apt install build-essential cmake

# 3. OpenMP (for parallel processing)
#    Usually comes with modern g++, but can be installed explicitly if needed.
sudo apt install libomp-dev

```
*   **cpp-httplib:** This library is fetched directly by CMake ...
*   **Other Distributions:** For non-Debian/Ubuntu systems, please use your distribution's package manager ...

##### CUDA Toolkit (Optional - For GPU Acceleration)

There are two main ways to install the CUDA Toolkit:

**1. General Method (Recommended for latest versions or specific version requirements):**

*   **NVIDIA Drivers:** First, ensure you have the proprietary NVIDIA drivers installed for your GPU. These are often available through your distribution's package manager (e.g., `nvidia-driver` package on Debian/Ubuntu) or directly from the [NVIDIA website](https://www.nvidia.com/Download/index.aspx).
*   **CUDA Toolkit Download:** Download and install the CUDA Toolkit from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive). It's generally recommended to install a version that is compatible with your NVIDIA driver.
    *   Follow the official NVIDIA installation guide for your operating system meticulously.
    *   Ensure that `nvcc` (the CUDA compiler) is in your `PATH` after installation, and that CMake can find the toolkit (often helped by setting `CUDA_TOOLKIT_ROOT_DIR` or ensuring standard installation paths are used).

**2. Using Debian/Ubuntu Package Manager (`apt` - for convenience, may not be the latest version):**

For Debian/Ubuntu systems, you can install CUDA components using `apt`. This method is often simpler but might provide an older version of the CUDA Toolkit than available directly from NVIDIA.

```bash
# Ensure your NVIDIA drivers are installed first.
# You might have done this via "Additional Drivers" or by installing a package like 'nvidia-driver-XXX'.

# Install the CUDA Toolkit and development libraries (cuBLAS is crucial for this project)
sudo apt update
sudo apt install nvidia-cuda-toolkit libcublas-dev

# Other potentially useful CUDA development libraries (optional, depending on broader needs):
# sudo apt install libcufft-dev libcurand-dev libcusolver-dev libcusparse-dev
```
*   After installation via `apt`, `nvcc` and other tools should generally be in the system `PATH`.

**CMake Detection:**

Regardless of the installation method, CMake will attempt to find the CUDA Toolkit. If `HAS_CUDA` is `ON` (either by default due to detection or set explicitly by you with `-DHAS_CUDA=ON`) and the toolkit is not found or key components like `nvcc` or `cublas` are missing, the CMake configuration step will fail with an error message.

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

### Where to Download Tested Model Weights

This project has been tested with specific model weights. You can download them from the following locations:

*   **For SafeTensors (BF16 format expected):**
    *   **TinyLlama 1.1B Chat v1.0:** [TinyLlama/TinyLlama-1.1B-Chat-v1.0 on Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
        *   Ensure you download `config.json`, `tokenizer.json`, and `model.safetensors` (which should be BF16).

*   **For GGUF:**
    *   **TinyLlama 1.1B Chat v1.0 (Q8_0 GGUF):** [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF - Q8_0.gguf](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf)
        *   This is a quantized version and has been tested.
    *   **Llama 2 7B (Q8_0 GGUF):** [TheBloke/Llama-2-7B-GGUF - llama-2-7b.Q8_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q8_0.gguf)
        *   This larger model has also been tested. Performance will vary based on your hardware.
    *   When using GGUF files, typically only the `.gguf` file itself is needed, as it should contain the necessary metadata. However, for some GGUF files or if using the `tinyllama` executable with a GGUF that doesn't fully embed tokenizer information, ensure a `tokenizer.json` (compatible with Llama 2) is present in the same directory or the model directory specified.

Place downloaded GGUF files in a directory (e.g., `models/`) or directly provide the path to the `tinyllama` executable.

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

This will create several executables in the `build/` directory (or `build/bin/` or `build/Release/` depending on your system and generator), including `tinyllama_server` (for SafeTensors models) and `tinyllama` (general purpose CLI).

For detailed insight into the operations performed by the executables (e.g., `tinyllama`, `tinyllama_server`), you can inspect the `debugging.log` file generated in the application's working directory. This log provides a step-by-step account of model loading, tokenization, and generation processes.

### Using the Management Scripts

For ease of use, comprehensive scripts are provided in the project root to automate common development and project tasks. These scripts simplify building, cleaning, running the applications, formatting code, generating documentation, and packaging releases.

#### `manage.sh` (for Linux/macOS)

**First, make the script executable:**

```bash
chmod +x manage.sh
```

**Available Commands:**

*   `./manage.sh build`: Compiles the project with configurable options for build type and CUDA support.
*   `./manage.sh clean`: Removes build artifacts and generated documentation.
*   `./manage.sh run-server`: Starts the chat server with configurable model directory, host, and port.
*   `./manage.sh run-chat`: Starts the command-line chat client with a configurable model directory and generation parameters.
*   `./manage.sh format`: Formats C++/CUDA source code using `clang-format`.
*   `./manage.sh docs`: Generates API documentation using Doxygen.
*   `./manage.sh docs-serve`: Serves generated documentation locally.
*   `./manage.sh docs-clean`: Removes generated documentation.
*   `./manage.sh package`: Creates a release tarball (`.tar.gz`).
*   `./manage.sh help`: Displays detailed help and options for all commands.

It is recommended to use this script for most routine operations. For detailed options for each command, please run `./manage.sh help`.

#### `manage.ps1` (for Windows PowerShell)

This script provides equivalent functionality to `manage.sh` for Windows users.

**Running the script:**

Open PowerShell, navigate to the project root directory, and then you can run commands like:

```powershell
.\manage.ps1 build
.\manage.ps1 build -BuildType Debug -Cuda OFF
.\manage.ps1 run-server -ModelDir .\data\TinyLlama-1.1B-Chat-v1.0
```

**Available Commands:**

*   `.\manage.ps1 build`: Compiles the project. Options: `-BuildType <Release|Debug>`, `-Cuda <ON|OFF>`.
*   `.\manage.ps1 clean`: Removes build artifacts and generated documentation.
*   `.\manage.ps1 run-server`: Starts the chat server. Options: `-ModelDir <path>`, `-Host <hostname>`, `-Port <port_number>`.
*   `.\manage.ps1 run-chat`: Starts the command-line chat client. Options: `-ModelDir <path>`, `-Temperature <float>`, `-TopK <int>`, `-TopP <float>`, `-Prompt <text>`.
*   `.\manage.ps1 format`: Formats C++/CUDA source code using `clang-format.exe` (must be in PATH).
*   `.\manage.ps1 docs`: Generates API documentation using `doxygen.exe` (must be in PATH).
*   `.\manage.ps1 docs-serve`: Serves generated documentation locally using Python's HTTP server (Python must be in PATH).
*   `.\manage.ps1 docs-clean`: Removes generated documentation.
*   `.\manage.ps1 package`: Creates a release ZIP archive. Options: `-Version <semver>`, `-BuildType <Release|Debug>`.
*   `.\manage.ps1 help`: Displays detailed help and options for all commands.

For detailed options for each command, run `.\manage.ps1 help`.

### 3. Run the Chat Server (Primary Example)

The main way to use this project is via the web server:

```bash
# Navigate back to the project root or ensure paths are correct
# Run the server, pointing it to your model data directory
./build/tinyllama_server ./data 

# Example if executable is directly in build:
# ./build/tinyllama_server ./data

# Example on Windows Release build:
# ./build/Release/tinyllama_server.exe ./data
```

*   Replace `./data` with the actual path to the directory containing your `config.json`, `tokenizer.json`, and `model.safetensors`.
*   The server will start, load the model, and listen on `http://localhost:8080` by default.
*   Open your web browser and navigate to `http://localhost:8080`.
*   You should see a basic chat interface where you can interact with the model.

### 4. Other Executables / Command-Line Usage

Besides the web server, you can interact with the models directly via command-line executables. These are typically found in `./build/bin/`.

*   **`tinyllama`**:
    *   **Description**: Command-line interface for chat. Can load models from a SafeTensors model directory (containing `config.json`, `model.safetensors`, `tokenizer.json`) OR by providing a direct path to a `.gguf` model file.
    *   **Usage**: `./build/tinyllama <model_path_or_dir> [prompt] [steps] [temperature]`
        *   `<model_path_or_dir>`: Path to model directory or `.gguf` file. (Default: `data`)
        *   `[prompt]`: The text prompt to send to the model. (Default: "Hello, world!")
        *   `[steps]`: Maximum number of new tokens to generate. (Default: `64`)
        *   `[temperature]`: Sampling temperature. (Default: `0.7`)
    *   **Example (SafeTensors directory)**:
        ```bash
        ./build/tinyllama ./data "Who is the prime minister of Australia?" 32 0.1
        ```
    *   **Example (GGUF file)**:
        ```bash
        ./build/tinyllama ./models/my_model.Q4_K_M.gguf "Explain black holes." 128 0.5

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
*   `manage.sh`: Management script for common tasks (build, clean, run, etc.) on Linux/macOS.
*   `manage.ps1`: Management script providing equivalent functionality for Windows PowerShell.
*   `test_pybindings.py`: Python script for testing and demonstrating the Python bindings.
*   `.clang-format`: Configuration file for the `clang-format` C++ code formatter.
*   `Doxyfile`: Configuration file for generating API documentation with Doxygen.
*   Key C++, Header, and CUDA files (typically in the root or organized by CMake):
    *   `main_gguf.cpp`: Main entry point for the `tinyllama` command-line chat client (supports GGUF and SafeTensors models).
    *   `server.cpp`: Implements the `tinyllama_server` HTTP server and its main entry point for web UI interaction.
    *   `api.cpp`/`api.h`: Defines the `TinyLlamaSession` class, providing a high-level API for loading models and generating text.
    *   `bindings.cpp`: Implements Python bindings for `TinyLlamaSession` and `ModelConfig` using `pybind11`.
    *   `model.cpp`/`model.h`: Contains the core Transformer model architecture and logic (attention, feed-forward layers, etc.).
    *   `model_constants.h`: Defines various constants related to model architecture and parameters.
    *   `model_macros.h`: Provides utility macros, notably for `NOMINMAX` compatibility (e.g., `SAFE_MIN`, `SAFE_MAX`) and other compile-time helpers.
    *   `tokenizer.cpp`/`tokenizer.h`: Handles loading of `tokenizer.json`, BPE encoding/decoding, and chat template application.
    *   `safetensors_loader.cpp`/`safetensors_loader.h`: Logic for parsing metadata and loading tensor data from `.safetensors` files.
    *   `gguf_parser.cpp`/`gguf_parser.h`: Logic for parsing metadata and loading tensor data from `.gguf` files.
    *   `cuda_kernels.cu` (and potentially `cuda_utils.h`): Contains CUDA kernels for GPU-accelerated operations and supporting utility functions (compiled if `HAS_CUDA=ON`).
    *   `logger.cpp`/`logger.h`: A simple utility for logging messages to `debugging.log`.
    *   `quantization.h`: Defines data structures and functions for handling various GGUF quantization types (e.g., Q8_0, Q4_K).
*   `www/`: Directory containing static web assets (HTML, CSS, JavaScript) for the chat interface served by `tinyllama_server`.
*   `data/` (Example Directory): Conventionally used for placing SafeTensors model directories, which include `config.json`, `model.safetensors`, and `tokenizer.json`.
*   `models/` (Example Directory): Conventionally used for storing GGUF model files (e.g., `my_model.Q8_0.gguf`).

## Documentation

For detailed API documentation, including class references, function signatures, and implementation details, please visit my documentation website:

[https://johnnyteutonic.github.io/tinyllama.cpp/](https://johnnyteutonic.github.io/tinyllama.cpp/)

The documentation is automatically generated from the source code using Doxygen and includes:
* Complete API reference for all classes and functions
* Detailed explanations of the model architecture
* Implementation details for both CPU and CUDA backends
* Examples and usage patterns
* Configuration options and build parameters

You can also generate the documentation locally using the management scripts:
```bash
# Using manage.sh (Linux/macOS)
./manage.sh docs

# Using manage.ps1 (Windows)
.\manage.ps1 docs
```

### Python Bindings (`tinyllama_bindings`)

This project includes Python bindings built using `pybind11`, allowing you to interact with the TinyLlama inference engine directly from Python. The core component exposed is the `TinyLlamaSession` class, which simplifies model loading and text generation.

#### Building and Installing the Python Bindings

The Python bindings are built as part of the main C++ project when CMake is configured. There isn't a separate installation step like `pip install .` in the traditional Python sense; rather, the build process generates a Python module file (`.pyd` on Windows, `.so` on Linux) that can be imported if it's in your Python path or if your script is run from a location where Python can find it (e.g., the project root after building).

1.  **Prerequisites:**
    *   Ensure you have Python installed (the version used for `pybind11` development, typically Python 3.x).
    *   Make sure CMake can find your Python installation. If you encounter issues, you might need to set CMake variables like `Python_EXECUTABLE` or ensure Python is correctly added to your system's PATH.
    *   The `pybind11` library is fetched automatically by CMake.

2.  **Build the Project:**
    Follow the general build instructions in the "Build the C++ Application" section. For example:
    ```bash
    # In project root
    mkdir build
    cd build
    cmake .. 
    # On Linux/macOS:
    make -j$(nproc)
    # On Windows (e.g., from Developer Command Prompt):
    cmake --build . --config Release
    ```
    After a successful build, the Python module (e.g., `build/Release/tinyllama_bindings.pyd` or `build/tinyllama_bindings.so`) should be created. The exact location might vary slightly based on your CMake generator and build type.

#### Using the Python Bindings

Once built, you can import and use the `tinyllama_bindings` module in your Python scripts.

**Example (`test_pybindings.py` located in the project root provides a more complete example):**

```python
import tinyllama_bindings
import os

# Path to your model (directory containing safetensors/config.json or a .gguf file)
# Adjust the path as necessary.
# model_path = "data/TinyLlama-1.1B-Chat-v1.0" 
# or for GGUF:
model_path = "models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"

if not os.path.exists(model_path):
    print(f"Model path not found: {model_path}")
    print("Please download the model and update the 'model_path' variable.")
    exit()

try:
    # Redirect C++ stdout/stderr to Python (optional, for viewing C++ logs)
    with tinyllama_bindings.ostream_redirect(stdout=True, stderr=True):
        print(f"Initializing TinyLlamaSession with model: {model_path}")
        # 1. Initialize the session with the model path
        session = tinyllama_bindings.TinyLlamaSession(model_path)
        print("TinyLlamaSession initialized.")

        # 2. Get model configuration (optional)
        config = session.get_config()
        print(f"Model BOS token ID: {config.bos_token_id}")
        print(f"Model EOS token ID: {config.eos_token_id}")
        print(f"Loaded from GGUF: {config.is_gguf_file_loaded}")

        # 3. Generate text
        prompt = "What is the capital of France?"
        num_steps = 50
        temperature = 0.7
        top_k = 40
        top_p = 0.9
        # The 'stop_tokens_str' and 'apply_q_a_format' are positional in current bindings
        # For safetensors models, apply_q_a_format=True is often desired.
        # For GGUF, it might depend on the model's pre-prompting.
        
        print(f"Generating text for prompt: '{prompt}'")
        generated_text = session.generate(
            prompt,
            num_steps,
            temperature,
            top_k,
            top_p,
            "",  # stop_tokens_str (e.g., "<|user|>")
            True # apply_q_a_format 
        )
        
        print("\n--- Generated Text ---")
        print(f"Prompt: {prompt}")
        print(f"Output: {generated_text}")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

### Running the Command-Line Chat Client (`tinyllama`)

```bash
# Navigate back to the project root or ensure paths are correct
# Run the server, pointing it to your model data directory
./build/tinyllama_server ./data 

# Example if executable is directly in build:
# ./build/tinyllama_server ./data

# Example on Windows Release build:
# ./build/Release/tinyllama_server.exe ./data
```

*   Replace `./data` with the actual path to the directory containing your `config.json`, `tokenizer.json`, and `model.safetensors`.
*   The server will start, load the model, and listen on `http://localhost:8080` by default.
*   Open your web browser and navigate to `http://localhost:8080`.
*   You should see a basic chat interface where you can interact with the model.

## Acknowledgements

This project has drawn significant inspiration and architectural insights from the excellent `llama.cpp` project by Georgi Gerganov and its contributors. Many of the core concepts for GGUF parsing, quantization, and efficient inference are based on the pioneering work done in `llama.cpp`.

Find `llama.cpp` on GitHub: [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) 

## Known Limitations

*   **SafeTensors Model Formats**: While the SafeTensors loading mechanism is in place, only BF16 (BFloat16) weight types have been extensively tested for these models. Other float types (like FP16 or FP32) in SafeTensors files may load but are not as thoroughly validated in this specific C++ implementation.
*   **Windows Support**: Building and running on Windows is possible but is considered highly experimental. The primary development and testing focus has been on Linux. Users may encounter build issues or runtime instabilities on Windows that are not present on Linux.
*   **Quantization Support (GGUF Path)**: The GGUF versions of tinyllama and Llama v2 have been tested using Q8_0 quants, but there is nominal support for Q4_K and Q6_K (along with their 'M' suffixes).
