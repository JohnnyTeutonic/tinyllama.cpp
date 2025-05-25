# TinyLlama.cpp - A C++ Chat Inference

This codebase supports inference for Llama 2/3 architecture models using Safetensors model format as well as GGUF format.

The GGUF format support includes loading models with various tensor types such as BF16, FP16, FP32, and the quantised types: Q4_K_M, Q6_K and Q8_0.

## Features

*   Pure C++ inference core (CPU-based).
*   Optional CUDA backend for GPU acceleration.
*   Support for both safetensors and GGUF formats (various quantizations like Q4_K_M, Q6_K, Q8_0 for GGUF).
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
    *   `tokenizer.json`: Defines the vocabulary, merge rules, and other tokenizer configurations.
    *   `model.safetensors`: The file containing the model weights.
    *   *Data Types:* The loader supports `F32`, `BF16`, and `F16` weight types from SafeTensors. `BF16` and `F16` are automatically converted to `F32` upon loading. Internal computation then proceeds in `F32`.

*   **GGUF (GPT-Generated Unified Format):** This format aims to package the model into a single file (`.gguf`).
    *   Ideally, the GGUF file embeds all necessary metadata, including tokenizer information.
    *   However, for some GGUF files (especially older ones or those converted with minimal metadata), or if the `tinyllama` executable's argument parsing requires it, a separate `tokenizer.json` compatible with the model (e.g., Llama 2/3 style SentencePiece) might still be needed. Place it in the same directory as the GGUF file or provide its path via the `<tokenizer_path>` argument.
    *   *Quantizations:* Supports various tensor types including `FP32`, `FP16`, `BF16`, and common quantized types like `Q4_K_M`, `Q6_K`, `Q8_0`, etc., as supported by the underlying GGUF parsing library.

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

*A Note on Python for Model Acquisition:*
While not required to build or run the C++ application itself, Python scripts using libraries like `torch`, Hugging Face `transformers`, `safetensors`, and `sentencepiece` are commonly used to download models from the Hugging Face Hub and save them into the required `config.json`, `tokenizer.json`, and `model.safetensors` (or `.gguf`) structure.

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
*   `<tokenizer_path>`: Path to `tokenizer.json` or `.model` file. (See "Model Files & Tokenizers" for GGUF notes).
*   `<num_threads>`: Number of CPU threads for generation.
*   `<prompt|chat>`: `prompt` for single generation, `chat` for interactive mode.
*   `--system-prompt "<text>"` (Optional): System-level instruction.
*   `initial_user_prompt` (Optional): First user message or main prompt text.
*   `--max-tokens <N>` (Optional): Max new tokens to generate (Default: 256).
*   `--n-gpu-layers <N>` (Optional): Layers to offload to GPU (-1 for all, 0 for none. Default: -1).
*   `--use-mmap <true|false>` (Optional): Memory-map GGUF files (Default: true).
*   `--temperature <F>` (Optional): Sampling temperature (e.g., 0.1. Default: 0.1).
*   `--use-kv-quant <true|false>` (Optional): Use INT8 KVCache on GPU (Default: false).*   `--use-batch-generation <true|false>` (Optional): Enable single-token batch generation (Default: false).*   `--max-batch-size <N>` (Optional): Maximum number of sequences for multi-prompt batch processing (Default: 1).

**Example Invocation:**

```bash
./build/tinyllama ./models/Llama-3-8B.Q4_K_M.gguf ./models/tokenizer.json 4 chat --system-prompt "You are a helpful AI." --n-gpu-layers -1 --use-kv-quant true "Tell me a joke."
```

For detailed operational logs, inspect `debugging.log` in the application's runtime directory.

### Python Package Installation

In addition to building and running the C++ executables, you can install `tinyllama.cpp` as a Python package. This allows you to use its core inference capabilities directly from Python scripts.

**Prerequisites:**

*   Ensure you have the C++ build dependencies installed as listed in the "C++ Runtime Dependencies" section above (CMake, C++17 compiler, OpenMP, and crucially **Boost.Regex**).
*   If you are using a Conda environment, it's highly recommended to install the C++ compilers and other dependencies from Conda channels to ensure ABI compatibility:
    ```bash
    conda install -c conda-forge cxx-compiler openmp 'boost<1.83' # Specify boost version if needed for compatibility
    # Note: 'boost' in conda-forge usually includes the regex component. 
    # If you encounter issues, you might need to be more specific or ensure the installed boost version is compatible.
    # For older boost versions, you might need 'conda install -c conda-forge boost-cpp'
    ```

**Installation Steps:**

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/JohnnyTeutonic/tinyllama.cpp.git # Or your fork
    cd tinyllama.cpp
    ```

2.  **Install the Python package using pip:**
    Navigate to the root of the cloned repository and run:
    ```bash
    pip install .
    ```

**Building with CUDA Support (Optional):**

By default, `pip install .` builds the CPU-only version of the package. To build the version with CUDA acceleration, you must have the NVIDIA CUDA Toolkit installed (see "CUDA Toolkit (Optional - For GPU Acceleration)" under C++ dependencies).

Then, set the `TINYLLAMA_CPP_BUILD_CUDA` environment variable to `1` before running pip:

```bash
# On Linux / macOS
TINYLLAMA_CPP_BUILD_CUDA=1 pip install .

# For an editable install with CUDA:
TINYLLAMA_CPP_BUILD_CUDA=1 pip install -e .
```

```powershell
# On Windows (PowerShell)
$env:TINYLLAMA_CPP_BUILD_CUDA="1"
pip install .

# For an editable install with CUDA:
$env:TINYLLAMA_CPP_BUILD_CUDA="1"
pip install -e .
```
If the `TINYLLAMA_CPP_BUILD_CUDA` variable is not set, or set to any other value than `1`, the CPU version will be built.

**Usage in Python:**

Once installed, you can import and use the package in your Python scripts:

```python
import tinyllama_cpp

# Example: (Ensure model paths are correct)
model_path = "path/to/your/model_or_gguf_file"
tokenizer_path = "path/to/your/tokenizer.json_or.model"

# Initialize the session with new enhanced API
# For GGUF, tokenizer_path can often be the same as model_path if tokenizer is embedded,
# or a separate tokenizer.model/tokenizer.json.
# For SafeTensors, model_path is the directory containing model.safetensors, config.json, tokenizer.json,
# and tokenizer_path should point to the tokenizer.json in that directory.
session = tinyllama_cpp.TinyLlamaSession(
    model_path=model_path,
    tokenizer_path=tokenizer_path,
    threads=4,                      # Number of CPU threads
    n_gpu_layers=-1,               # Use -1 for all GPU layers if CUDA enabled
    use_mmap=True,                 # Memory-map GGUF files for efficiency
    use_kv_quant=False,            # Enable INT8 KVCache quantization on GPU
    use_batch_generation=False,     # Enable single-token batch generation
    max_batch_size=1               # Maximum number of sequences for multi-prompt batch processing
)

# Define prompts
user_prompt = "What is the capital of France?"
system_prompt_optional = "You are a helpful geography expert." # Optional

# Single Generation
# The `apply_q_a_format` parameter defaults to `true`. This means Q:A style formatting 
# (e.g., "Q: [prompt]\\nA:") is applied by default if the loaded model is not Llama 3 
# and does not have an explicit GGUF chat template. This is often preferred for Llama 2 models.
# Set to `False` if you want to use a raw prompt for such models.
# If a GGUF or Llama 3 chat template is active, that template takes precedence.
response = session.generate(
    prompt=user_prompt, 
    steps=64, 
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    system_prompt=system_prompt_optional, # Pass the system prompt here
    apply_q_a_format=True # This is now the default
)

print(f"User: {user_prompt}")
if system_prompt_optional:
    print(f"System: {system_prompt_optional}")
print(f"AI: {response}")

# Batch Generation (NEW FEATURE!)
# Process multiple prompts efficiently in a single batch
if session.max_batch_size > 1:  # Only if batch processing is enabled
    batch_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about coding."
    ]
    
    print("\n--- Batch Generation ---")
    batch_results = session.generate_batch(
        prompts=batch_prompts,
        steps=64,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        system_prompt=system_prompt_optional,
        apply_q_a_format=True
    )
    
    for i, (prompt, result) in enumerate(zip(batch_prompts, batch_results)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Response {i+1}: {result}")

# Example for chat interaction (simplified loop)
print("\nEntering chat mode (type 'quit' to exit)...")
while True:
    current_user_input = input("You: ")
    if current_user_input.lower() == 'quit':
        break
    
    chat_response = session.generate(
        prompt=current_user_input,
        steps=128,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        system_prompt=system_prompt_optional, # Maintain system prompt across turns if desired
        apply_q_a_format=True # Default, applies Q:A if no GGUF/Llama3 template
    )
    print(f"AI: {chat_response}")
```

Refer to your Python bindings implementation (`bindings.cpp` and `tinyllama_cpp/__init__.py`) for the exact classes and methods available.

#### New Batch Processing Features

**TinyLlama.cpp** now supports efficient batch processing for multiple prompts:

**Constructor Parameters:**
- `use_batch_generation` (bool): Enables single-token batch generation for improved performance
- `max_batch_size` (int): Maximum number of sequences that can be processed in a single batch

**New Methods:**
- `generate_batch(prompts, steps, temperature, top_k, top_p, system_prompt, apply_q_a_format)`: Process multiple prompts in a single efficient batch operation

**Example for batch processing with enhanced constructor:**

```python
# Initialize with batch processing enabled
session = tinyllama_cpp.TinyLlamaSession(
    model_path="./models/model.gguf",
    tokenizer_path="./models/tokenizer.json",
    threads=4,
    n_gpu_layers=-1,
    use_mmap=True,
    use_kv_quant=False,
    use_batch_generation=True,      # Enable batch generation
    max_batch_size=5               # Support up to 5 prompts per batch
)

# Process multiple prompts efficiently
prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "How does AI work?"
]

batch_results = session.generate_batch(
    prompts=prompts,
    steps=100,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    system_prompt="You are an AI expert.",
    apply_q_a_format=True
)

for prompt, result in zip(prompts, batch_results):
    print(f"Q: {prompt}")
    print(f"A: {result}\n")
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

### 4. Other Executables / Command-Line Usage

Besides the web server, you can interact with the models directly via command-line executables. These are typically found in `./build/bin/` or `./build/`.

*   **`tinyllama`** (main executable, usually in `./build/bin/main` or `./build/tinyllama`):
    *   **Description**: Command-line interface for chat or single prompt generation. Can load models from a SafeTensors model directory (containing `config.json`, `model.safetensors`, `tokenizer.json`) OR by providing a direct path to a `.gguf` model file.
    *   **Usage**:
        ```
        ./build/bin/main <model_path> <tokenizer_path> <num_threads> <mode> [initial_prompt] [max_tokens] [n_gpu_layers] [use_mmap]
        ```
        *   `<model_path>`: Path to the model file (.gguf) or directory (SafeTensors).
        *   `<tokenizer_path>`: Path to the `tokenizer.json` file. For GGUF models that embed tokenizer info, this can often be an empty string `""` or a placeholder if `manage.sh` supplies it, but it's a required positional argument for the C++ executable.
        *   `<num_threads>`: Number of threads to use for generation (CPU computation).
        *   `<mode>`: Operation mode. Use `"chat"` for interactive chat or `"prompt"` for single prompt generation.
        *   `[initial_prompt]`: (Optional) The initial prompt string. For `chat` mode, this starts the conversation. For `prompt` mode, this is the text to complete. (Default: "Hello, world!")
        *   `[max_tokens]`: (Optional) Maximum number of new tokens to generate. (Default: `256`)
        *   `[n_gpu_layers]`: (Optional) Number of layers to offload to GPU. `-1` means all layers to GPU, `0` means all layers to CPU. A positive number specifies the exact number of layers for the GPU (remaining on CPU). (Default: `-1`).
        *   `[use_mmap]`: (Optional) Whether to use memory-mapping for GGUF file metadata reading (`true` or `false`). (Default: `true`). Note: For GGUF weight loading itself, mmap is currently always used internally by the model loader for efficiency; this flag primarily influences the initial metadata peek by `TinyLlamaSession`.
    *   **Note on Sampling Parameters**: The `tinyllama` executable currently uses default internal values for temperature, top-k, and top-p. To control these, modify them within `main.cpp` or extend `main.cpp` to parse them from the command line.
    *   **Example (SafeTensors directory, chat mode via `manage.sh` which constructs the correct call):**
        ```bash
        ./manage.sh run-chat --model-dir ./data/TinyLlama-1.1B-Chat-v1.0 --tokenizer ./data/TinyLlama-1.1B-Chat-v1.0/tokenizer.json --threads 4 --system-prompt "You are a helpful assistant."
        ```
    *   **Example (GGUF file, direct call, prompt mode):**
        ```bash
        ./build/bin/main ./models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf ./models/tokenizer.json 4 prompt "Explain black holes in simple terms" 128 0 true
        ```

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
    *   `server.cpp`: Implements the `tinyllama_server` HTTP server and its main entry point for web UI interaction.
    *   `api.cpp`/`api.h`: Defines the `TinyLlamaSession` class, providing a high-level API for loading models and generating text.
    *   `bindings.cpp`: Implements Python bindings for `TinyLlamaSession` and `ModelConfig` using `pybind11`.
    *   `model.cpp`/`model.h`: Contains the core Transformer model architecture and logic (attention, feed-forward layers, etc.).
    *   `model_constants.h`: Defines various constants related to model architecture and parameters.
    *   `model_macros.h`: Provides utility macros, notably for `NOMINMAX` compatibility (e.g., `SAFE_MIN`, `SAFE_MAX`) and other compile-time helpers.
    *   `tokenizer.cpp`/`tokenizer.h`: Handles loading of `tokenizer.json`, BPE encoding/decoding, and chat template application.
    *   `safetensors_loader.cpp`/`safetensors_loader.h`: Logic for parsing metadata and loading tensor data from `.safetensors` files.
    *   `gguf_parser.cpp`/`gguf_parser.h`: Logic for parsing metadata and loading tensor data from `.gguf` files.
    *   `cuda_kernels.cu`