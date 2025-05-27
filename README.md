# TinyLlama.cpp - Cross-Format LLM Inference Engine

This codebase supports inference for Llama 2/3 architecture models using both GGUF and Safetensors formats, and additionally contains Python bindings.

The GGUF format support includes loading models with various tensor types such as BF16, FP16, FP32, and the quantised types: Q4_K_M, Q6_K, Q8_0, and Q8_K.

**Note for Older Llama Models (Llama 2/TinyLlama):** Some older GGUF files may not contain explicit BPE merge rules. The system automatically handles this by generating merge rules from vocabulary and token scores (similar to llama.cpp's approach), ensuring proper tokenization without requiring external files.

## Features

*   Pure C++ inference core (CPU-based).
*   Optional CUDA backend for GPU acceleration.
*   Support for both safetensors and GGUF formats (various quantizations like Q4_K_M, Q6_K, Q8_0, Q8_K for GGUF).
*   Python bindings
*   Built-in web server (`cpp-httplib`) for easy interaction via a web UI.
*   Minimal external dependencies managed via CMake.
*   Cross-platform (tested on Linux, Windows - requires C++17 compiler).

## Dependencies and Setup

This section outlines the necessary components to build and run the project, and how to obtain model files.

### 1. C++ Build Environment Dependencies

Core requirements to **build and run** the C++ application:

1.  **CMake (>= 3.11):** For building the project.
2.  **C++17 Compliant Compiler:** Such as g++, Clang, or MSVC.
3.  **Boost Libraries (Specifically Regex & Xpressive):** Needed for tokenizer functionalities, especially for Llama 3 tokenizers.
    *   **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install build-essential cmake libboost-all-dev libomp-dev`
        *   `libboost-all-dev` is simplest. For minimal, ensure `libboost-regex-dev` and Boost Xpressive headers are installed.
    *   **Linux (Fedora/RHEL):** `sudo dnf install gcc-c++ cmake boost-devel libgomp` (or `libomp-devel`).
    *   **macOS (Homebrew):** `brew install cmake boost llvm` (llvm for OpenMP; may need extra flags if clang isn't finding OpenMP).
    *   **Windows (vcpkg):** `vcpkg install boost-regex nlohmann-json cpp-httplib` (OpenMP usually included with MSVC).
        *   Ensure Xpressive headers are pulled, often via `boost-headers` or a full `boost` package if `boost-regex` alone is insufficient.
    *   **Windows (Chocolatey):** `choco install cmake visualstudio2022buildtools boost-msvc-14.3` (or similar for your VS and Boost versions).
4.  **nlohmann/json & cpp-httplib:** These are fetched automatically by CMake if not found system-wide (e.g., if not installed via vcpkg or a system package manager). Usually, no separate manual installation is needed.
5.  **OpenMP (Optional but Recommended):** For multi-threading CPU acceleration. Often included with the compiler. If missing, install (e.g., `libomp-dev` on Debian, `libgomp` on Fedora, from `llvm` on macOS, or part of MSVC). Performance will be lower without it.
6.  **CUDA Toolkit (Optional - For GPU Acceleration):**
    *   Required **only** if you want GPU-accelerated inference. You'll need a compatible NVIDIA GPU and drivers.
    *   **Installation:** Download from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and follow NVIDIA's official installation guide. Ensure `nvcc` is in your PATH.
    *   **Alternative (Linux - Simpler, May Be Older):** On Debian/Ubuntu, `sudo apt install nvidia-cuda-toolkit libcublas-dev` can be used after NVIDIA drivers are set up.
    *   CMake (`-DHAS_CUDA=ON`) will detect it. `nvcc` (compiler) and `cublas` (library) are key.

### 2. Model Files & Tokenizers

To run the model, you need both the model weights and tokenizer information. These should be placed in an accessible directory (e.g., `data/` or `models/` within your project structure).

#### Supported Formats:

*   **SafeTensors:** This format typically involves three files:
    *   `config.json`: Contains the model architecture, hyperparameters, and other metadata.
    *   `tokenizer.json`: Defines the vocabulary, merge rules, and other tokenizer configurations. **Required for SafeTensors format.**
    *   `model.safetensors`: The file containing the model weights.
    *   *Data Types:* The loader supports `F32`, `BF16`, and `F16` weight types from SafeTensors. `BF16` and `F16` are automatically converted to `F32` upon loading. Internal computation then proceeds in `F32`.

*   **GGUF (GPT-Generated Unified Format):** This format packages the model into a single self-contained file (`.gguf`).
    *   **Tokenizer Requirements:**
        *   **All GGUF models:** Tokenizer is embedded in the GGUF file. No external tokenizer files are required.
        *   **Llama 2/TinyLlama models:** Older GGUF files may lack explicit BPE merge rules, but the system automatically generates them from embedded vocabulary and scores.
        *   **Llama 3+ models:** Full tokenizer with merge rules is typically embedded.
        *   You can set `tokenizer_path` to the same path as the model file, or omit it entirely in Python bindings.
    *   *Quantizations:* Supports various tensor types including `FP32`, `FP16`, `BF16`, and common quantized types like `Q4_K_M`, `Q6_K`, `Q8_0`, `Q8_K`, etc., as supported by the underlying GGUF parsing library.

#### Example Model Sources:

It's recommended to download models from reputable sources like Hugging Face. Here are some examples that have been tested:

*   **TinyLlama 1.1B Chat v1.0:**
    *   SafeTensors: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) (download `config.json`, `tokenizer.json`, `model.safetensors`)
    *   GGUF (e.g., Q8_0): [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) (download the `.gguf` file)
*   **Llama-2 7B:**
    *   SafeTensors (HF format): [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
    *   GGUF (e.g., Q8_0): [TheBloke/Llama-2-7B-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)
*   **Meta-Llama-3 8B:**
    *   GGUF (e.g., Q4_K_M): [QuantFactory/Meta-Llama-3-8B-GGUF](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF)
    *   For Llama 3 SafeTensors, you would typically download from the official Meta Llama repository or a trusted conversion, ensuring you get the Llama 3 specific `tokenizer.json` and `config.json`.

### 3. Build the C++ Application

Use CMake to build the project. Navigate to the project root directory in your terminal.

```bash
# Create a build directory (if it doesn't exist)
mkdir -p build
cd build

# Configure with CMake and Build
# Option 1: Auto-detect CUDA (Recommended if CUDA is installed and intended for use)
# CMake will try to find CUDA. If found, HAS_CUDA will be ON.
cmake ..

# Option 2: Explicitly Enable CUDA (if auto-detection fails or to be certain)
# cmake .. -DHAS_CUDA=ON 

# Option 3: Explicitly Disable CUDA (if CUDA is installed but you want a CPU-only build)
# cmake .. -DHAS_CUDA=OFF

# (If CUDA is not installed, HAS_CUDA will automatically be OFF)

# Build the executables
# For Makefiles (Linux/macOS default):
make -j$(nproc) # $(nproc) gets the number of processing units

# For MSVC on Windows (from a Developer Command Prompt, typically): 
# cmake --build . --config Release # Or Debug
# Alternatively, open the .sln file generated in the build/ directory with Visual Studio.
```

This process will create executables in the `build/` directory (or a subdirectory like `build/Release/` on Windows with MSVC). Key executables include:

*   `tinyllama`: The main command-line interface for direct interaction (chat/prompt modes).
*   `tinyllama_server`: A web server for interacting with SafeTensors models via a UI (this may be merged or evolve depending on project direction).

**Running the `tinyllama` Executable:**

```bash
# Example path, adjust if your executable is in a subdirectory like build/Release
./build/tinyllama <model_path> <tokenizer_path> <num_threads> <prompt|chat> [options...]
```

**Key Command-Line Arguments for `tinyllama`:**

*   `<model_path>`: Path to model file (.gguf) or directory (SafeTensors).
*   `<tokenizer_path>`: Path to tokenizer file. 
    *   **Required for:** SafeTensors format (must point to `tokenizer.json`)
    *   **For GGUF models:** Use the same path as model_path (tokenizer is embedded)
*   `<num_threads>`: Number of CPU threads for generation.
*   `<prompt|chat>`: `prompt` for single generation, `chat` for interactive mode.
*   `--system-prompt "<text>"` (Optional): System-level instruction.
*   `initial_user_prompt` (Optional): First user message or main prompt text.
*   `--max-tokens <N>` (Optional): Max new tokens to generate (Default: 256).
*   `--n-gpu-layers <N>` (Optional): Layers to offload to GPU (-1 for all, 0 for none. Default: -1).
*   `--use-mmap <true|false>` (Optional): Memory-map GGUF files (Default: true).
*   `--temperature <F>` (Optional): Sampling temperature (e.g., 0.1. Default: 0.1).
*   `--top-k <N>` (Optional): Top-K sampling parameter (0 to disable). Default: 40.
*   `--top-p <F>` (Optional): Top-P/nucleus sampling parameter (0.0-1.0). Default: 0.9.
*   `--use-kv-quant <true|false>` (Optional): Use INT8 KVCache on GPU (Default: false).
*   `--use-batch-generation <true|false>` (Optional): Enable single-token batch generation (Default: false).
*   `--max-batch-size <N>` (Optional): Maximum number of sequences for multi-prompt batch processing (Default: 1).

**Note on Sampling Parameters**: The `tinyllama` executable supports `--temperature`, `--top-k`, and `--top-p` via command line for full control over text generation sampling.

**Example Invocation:**

```bash
# For GGUF models (tokenizer embedded in file)
./build/tinyllama ./models/model.Q4_K_M.gguf ./models/model.Q4_K_M.gguf 4 chat --system-prompt "You are a helpful AI." --n-gpu-layers -1 --use-kv-quant true --temperature 0.7 --top-k 50 --top-p 0.95 "Tell me a joke."

# For SafeTensors format (separate tokenizer required)
./build/tinyllama ./models/safetensors_directory ./models/tokenizer.json 4 chat --system-prompt "You are a helpful AI." --n-gpu-layers -1 --use-kv-quant true --temperature 0.7 --top-k 50 --top-p 0.95 "Tell me a joke."
```

For detailed operational logs, inspect `debugging.log` in the application's runtime directory.

### Python Package Installation

**Development Installation (CPU-only):**
```bash
git clone https://github.com/JohnnyTeutonic/tinyllama.cpp.git
cd tinyllama.cpp
# Install from the project directory
pip install .
```

**Development Installation with CUDA Support:**
```bash
git clone https://github.com/JohnnyTeutonic/tinyllama.cpp.git
cd tinyllama.cpp
# Set environment variable to enable CUDA build
export TINYLLAMA_CPP_BUILD_CUDA=1  # Linux/macOS
# or
set TINYLLAMA_CPP_BUILD_CUDA=1     # Windows CMD
# or
$env:TINYLLAMA_CPP_BUILD_CUDA=1    # Windows PowerShell

# Install from the project directory
pip install .
```

**Development Installation with PyTorch Dependencies:**
```bash
git clone https://github.com/JohnnyTeutonic/tinyllama.cpp.git
cd tinyllama.cpp
# Install from the project directory with PyTorch extras
pip install .[torch]
```

**Editable Development Installation:**
```bash
git clone https://github.com/JohnnyTeutonic/tinyllama.cpp.git
cd tinyllama.cpp
# Editable install from the project directory (CPU-only)
pip install -e .

# For CUDA support with editable install:
export TINYLLAMA_CPP_BUILD_CUDA=1   # Linux/macOS
set TINYLLAMA_CPP_BUILD_CUDA=1      # Windows
pip install -e .
```

**Prerequisites for CUDA Build:**
- NVIDIA CUDA Toolkit (11.0 or later) installed and in PATH
- Compatible NVIDIA GPU drivers
- CMake 3.18 or later
- C++17 compatible compiler

**Usage:**
```python
import tinyllama_cpp

# For SafeTensors format (tokenizer_path required)
session = tinyllama_cpp.TinyLlamaSession(
    model_path="path/to/safetensors/directory",
    tokenizer_path="path/to/tokenizer.json",
    threads=4,
    n_gpu_layers=-1  # Use GPU if available
)

# For GGUF models (tokenizer embedded, use same path for both)
session = tinyllama_cpp.TinyLlamaSession(
    model_path="path/to/model.gguf",
    tokenizer_path="path/to/model.gguf",  # Same as model_path
    threads=4,
    n_gpu_layers=-1
)

response = session.generate("What is AI?", steps=64)
print(response)
```

### Using the Management Scripts

For ease of use, comprehensive scripts are provided in the project root to automate common development and project tasks. These scripts simplify building, cleaning, running the applications, formatting code, generating documentation, and packaging releases.

#### `manage.sh` (for Linux/macOS)

**First, make the script executable:**

```bash
chmod +x manage.sh
```

**Key Command Options (refer to `./manage.sh help` for all options):**

*   `./manage.sh build [--build-type <Release|Debug>] [--cuda <ON|OFF>]`
*   `./manage.sh run-server [--model-dir <path>] [--tokenizer <path>] [--threads <num>] [--host <hostname>] [--port <num>] [--n-gpu-layers <num>] [--mmap <true|false>] [--no-log]`
*   `./manage.sh run-chat [--model-dir <path>] [--tokenizer <path>] [--threads <num>] [--system-prompt <text>] [--prompt <text>] [--steps <num>] [--n-gpu-layers <num>] [--mmap <true|false>]`
    *   (Note: `run-chat` specific sampling parameters like temperature, top-k, top-p are set to defaults in the C++ `main`.)
*   `./manage.sh run-prompt [--model-dir <path>] [--tokenizer <path>] [--prompt <text>] [--steps <num>] [--threads <num>] [--n-gpu-layers <num>] [--mmap <true|false>][--temperature <num>]`
    *   This command runs the model with a single provided prompt and then exits.
    *   If `--model-dir` is not provided, you can specify the model directory/GGUF file path as a single positional argument after `run-prompt`.
    *   Example: `./manage.sh run-prompt path/to/your/model --prompt "Translate to French: Hello"`
*   `./manage.sh install [--gpu|--cpu]`
    *   Installs the Python package in editable mode with optional GPU support.
    *   `--cpu` (default): CPU-only installation
    *   `--gpu`: Installation with CUDA support (requires CUDA toolkit)
    *   Example: `./manage.sh install --gpu` for GPU support or `./manage.sh install --cpu` for CPU-only

It is recommended to use this script for most routine operations. For detailed options for each command, please run `./manage.sh help`.

#### `manage.ps1` (for Windows PowerShell)

This script provides equivalent functionality to `manage.sh` for Windows users.

**Running the script (example):**

```powershell
.\\manage.ps1 build -BuildType Debug -Cuda OFF
.\\manage.ps1 run-chat -ModelDir .\\models\\my_model.gguf -TokenizerPath .\\models\\tokenizer.json -Threads 2 -SystemPrompt "You are a helpful assistant."
```

**Key Command Options (refer to `.\\manage.ps1 help` for all options):**

*   `.\\manage.ps1 build [-BuildType <Release|Debug>] [-Cuda <ON|OFF>]`
*   `.\\manage.ps1 run-server [-ModelDir <path>] [-TokenizerPath <path>] [-Threads <num>] [-Host <hostname>] [-Port <num>] [-NGpuLayers <num>] [-Mmap <$true|$false>] [-NoLog]`
*   `.\\manage.ps1 run-chat [-ModelDir <path>] [-TokenizerPath <path>] [-Threads <num>] [-SystemPrompt <text>] [-Prompt <text>] [-Steps <num>] [-NGpuLayers <num>] [-Mmap <$true|$false>]`
    *   (Note: `run-chat` specific sampling parameters like temperature, top-k, top-p are set to defaults in the C++ `main`.)
*   `.\\manage.ps1 run-prompt [-ModelDir <path>] [-TokenizerPath <path>] [-Prompt <text>] [-Steps <num>] [-Threads <num>] [-NGpuLayers <num>] [-Mmap <$true|$false>][-Temperature <num>]`
    *   This command runs the model with a single provided prompt and then exits.
    *   If `-ModelDir` is not provided, you can specify the model directory/GGUF file path as a single positional argument after `run-prompt`.
    *   Example: `.\\manage.ps1 run-prompt -ModelDir path\\to\\your\\model -Prompt "What is the capital of France?"`
*   `.\\manage.ps1 install [-Gpu|-Cpu]`
    *   Installs the Python package in editable mode with optional GPU support.
    *   `-Cpu` (default): CPU-only installation
    *   `-Gpu`: Installation with CUDA support (requires CUDA toolkit)
    *   Example: `.\\manage.ps1 install -Gpu` for GPU support or `.\\manage.ps1 install -Cpu` for CPU-only


For detailed options for each command, run `.\\manage.ps1 help`.

### 3. Run the Chat Server (Primary Example)

The main way to use this project is via the web server:

```bash
# Navigate back to the project root or ensure paths are correct
# Run the server, pointing it to your model data directory
./build/tinyllama_server ./data 

# Example on Windows Release build:
# ./build/Release/tinyllama_server.exe ./data
```

*   Replace `./data` with the actual path to the directory containing your `config.json`, `tokenizer.json`, and `model.safetensors`.
*   The server will start, load the model, and listen on `http://localhost:8080` by default.
*   Open your web browser and navigate to `http://localhost:8080`.
*   You should see a basic chat interface where you can interact with the model.

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

### Root Directory
*   `CMakeLists.txt`: Main build configuration defining dependencies, targets, and compilation options.
*   `pyproject.toml`: Modern Python packaging configuration with optional dependencies for GPU and PyTorch support.
*   `manage.sh`: Comprehensive management script for Linux/macOS (build, clean, run, format, docs, etc.).
*   `manage.ps1`: Windows PowerShell equivalent of the management script.
*   `.clang-format`: Code formatting configuration for consistent C++ style.
*   `Doxyfile`: Doxygen configuration for generating API documentation.
*   `README.md`: This comprehensive documentation file.

### Core C++ Implementation
*   **`main.cpp`**: Command-line interface entry point for `tinyllama` executable.
*   **`server.cpp`**: HTTP server implementation for web UI interaction (`tinyllama_server` executable).
*   **`api.cpp`/`api.h`**: High-level `TinyLlamaSession` API for model loading and text generation.
*   **`bindings.cpp`**: Python bindings using `pybind11` with comprehensive documentation for `help()` support.
*   **`model.cpp`/`model.h`**: Core Transformer architecture (attention, feed-forward, RoPE, etc.) with SIMD optimizations.
*   **`model_constants.h`**: Architecture constants and configuration parameters.
*   **`model_macros.h`**: Utility macros for cross-platform compatibility and safe operations.
*   **`gguf_structs.h`**: Data structures and type definitions for GGUF format parsing.
*   **`ggml_types.h`**: Type definitions compatible with GGML format specifications.

### Data Loading & Processing
*   **`tokenizer.cpp`/`tokenizer.h`**: BPE tokenization, chat template application, and multi-format tokenizer support.
*   **`safetensors_loader.cpp`/`safetensors_loader.h`**: SafeTensors format parsing and tensor loading.
*   **`gguf_parser.cpp`/`gguf_parser.h`**: GGUF format parsing with support for various quantizations.
*   **`quantization.cpp`/`quantization.h`**: Quantization utilities and dequantization routines.
*   **`utils.cpp`/`utils.h`**: General utility functions for string processing, file operations, and helper routines.

### GPU Acceleration (Optional)
*   **`cuda_kernels.cu`/`cuda_kernels.h`**: CUDA kernels for GPU-accelerated inference.
*   **`logger.cpp`/`logger.h`**: Logging utilities with GPU memory monitoring.

### Python Package Structure
*   **`tinyllama_cpp/`**: Python package directory
    *   `__init__.py`: Package initialization with dynamic versioning and error handling.
    *   `_version.py`: Auto-generated version file (created during build).

### PyTorch Reference Implementation
*   **`pytorch/`**: Pure PyTorch implementation for comparison and experimentation
    *   `run_inference.py`: Main PyTorch inference script.
    *   `tinyllama.py`: PyTorch model definition.
    *   `utils.py`: Utility functions for PyTorch implementation.
    *   `requirements.txt`: PyTorch-specific dependencies.
    *   `README.md`: PyTorch implementation documentation.

### Examples & Web Interface
*   **`examples/`**: Example scripts and usage demonstrations
*   **`www/`**: Web interface assets for the HTTP server
*   **`docs/`**: Generated documentation and additional guides

### Build & Development
*   **`build/`**: CMake build directory (created during compilation)
    *   Contains compiled executables: `tinyllama`, `tinyllama_server`
*   **`_skbuild/`**: Python build artifacts (created during `pip install`)
*   **`debugging.log`**: Runtime debugging output (created during execution)

### Key Features by Component
*   **Model Loading**: Supports both GGUF (single file) and SafeTensors (multi-file) formats
*   **Tokenization**: Handles Llama/Llama2 SentencePiece and Llama3 TikToken tokenizers
*   **Inference**: CPU with OpenMP + optional SIMD, GPU with CUDA acceleration
*   **Python Bindings**: Full-featured with comprehensive help documentation
*   **Batch Processing**: Efficient parallel processing of multiple prompts
*   **Memory Management**: KV cache with optional INT8 quantization, memory mapping support