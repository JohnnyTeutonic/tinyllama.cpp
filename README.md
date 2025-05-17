# TinyLlama.cpp - A C++ Chat Inference

This codebase supports inference for Llama 2/3 architecture models using Safetensors model format as well as GGUF format.

The GGUF format support includes loading models with various tensor types such as BF16, FP16, FP32, and the quantised types: Q4_K_M, Q6_K and Q8_0.

## Features

*   Pure C++ inference core (CPU-based).
*   Optional CUDA backend for GPU acceleration.
*   Support for both safetensors and GGUF formats.
*   Python bindings
*   Built-in web server (`cpp-httplib`) for easy interaction via a web UI.
*   Minimal external dependencies managed via CMake.
*   Cross-platform (tested on Linux, Windows - requires C++17 compiler).

## Dependencies

### C++ Runtime Dependencies

These are needed to **build and run** the C++ application:

1.  **CMake (>= 3.11):** Used for building the project.
2.  **C++17 Compliant Compiler:** Such as g++, Clang, or MSVC.
3.  **Boost Libraries (Regex, Xpressive, etc.):** For regular expression support. (Handled by `find_package` in CMake; system installation of appropriate Boost libraries needed)
    *   `libboost-all-dev` is recommended to ensure all necessary components,
        including header-only libraries like Boost.Xpressive (required for Llama 3
        tokenizer functionality), are installed.
        If you prefer a more minimal installation, you would need at least
        `libboost-regex-dev` and ensure `boost/xpressive/xpressive.hpp`
        is available from another Boost header package (e.g. `libboost-dev`
        for a specific version which includes all headers).
        However, `libboost-all-dev` is the most straightforward way to get everything.
4.  **nlohmann/json:** For parsing JSON configuration files. (Fetched automatically by CMake if not found system-wide).
5.  **cpp-httplib:** For the web server backend. (Fetched automatically by CMake).
6.  **OpenMP (Optional but Recommended):** For multi-threading CPU acceleration. CMake will try to find it; performance will be lower without it.
7.  **CUDA Toolkit (Optional):** Required **only** if you want GPU-accelerated inference. CMake will detect it if available and build the CUDA kernels. You'll need a compatible NVIDIA GPU and drivers.

#### Installing C++ Dependencies on Linux (Debian/Ubuntu Example)

This section provides example commands for installing the necessary C++ dependencies on Debian-based Linux distributions like Ubuntu. Package names and commands may vary for other distributions (e.g., Fedora, Arch Linux, CentOS).

```bash
# 1. Update package lists
sudo apt update

# 2. Essential build tools (includes g++ compiler) and CMake
sudo apt install build-essential cmake

# 3. Boost Libraries (Regex, Xpressive, etc.)
#    `libboost-all-dev` is recommended to ensure all necessary components,
#    including header-only libraries like Boost.Xpressive (required for Llama 3
#    tokenizer functionality), are installed.
#    If you prefer a more minimal installation, you would need at least
#    `libboost-regex-dev` and ensure `boost/xpressive/xpressive.hpp`
#    is available from another Boost header package (e.g. `libboost-dev`
#    for a specific version which includes all headers).
#    However, `libboost-all-dev` is the most straightforward way to get everything.
sudo apt install libboost-all-dev

# 4. OpenMP (for parallel processing)
#    Usually comes with modern g++, but can be installed explicitly if needed.
sudo apt install libomp-dev

```
*   **Boost.Regex on other systems:**
    *   **Fedora/RHEL:** `sudo dnf install boost-devel` (this typically includes all Boost headers and libraries, including Xpressive and Regex)
    *   **macOS (Homebrew):** `brew install boost` (this installs multiple Boost libraries, including Xpressive headers and Regex)
    *   **Windows:** If using Chocolatey, `boost-msvc-14.3` (or a similar versioned package like `boost-msvc-14.2`) aims to provide a comprehensive Boost installation. If using `vcpkg`, install `boost-regex` and ensure Xpressive headers are available (often through `boost-headers` or by installing `boost-xpressive` if available as a separate component, or by installing a full `boost` package). If building Boost from source, ensure the regex library is built and all headers are installed.
*   **nlohmann/json & cpp-httplib:** These libraries are fetched directly by CMake if not found system-wide, so no separate installation step is typically needed for them.
*   **Other Distributions:** For non-Debian/Ubuntu systems, please use your distribution's package manager to find the equivalent packages for `build-essential`, `cmake`, `libboost-all-dev` (or its equivalent like `boost-devel`), and `libomp-dev`.

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

#### Safetensors

For safetensors models, you need three files from a compatible TinyLlama model (e.g., from Hugging Face):

*   `config.json`
*   `tokenizer.json`
*   `model.safetensors`

Place these three files into a directory, for example, named `data/`.

***Supported SafeTensors Data Types:***

*This project's SafeTensors loader currently supports models with weights in the following formats:*
    *   **`F32` (Single-precision Floating Point):** Loaded as is.
    *   **`BF16` (BFloat16 Floating Point):** These tensors are automatically converted to `F32` upon loading.
    *   **`F16` (Half-precision Floating Point):** These tensors are also automatically converted to `F32` upon loading.

*The internal representation and computation for models loaded from SafeTensors will therefore use `F32` precision. Support for other data types (e.g., quantized integer types like `I8`) directly from SafeTensors is not yet implemented.*

#### GGUF

Simply download a llama v2 or tinyllama model from huggingface (TheBloke's repos are the best for this) and place the file into the appropriate directory.


### Where to Download Tested Model Weights

This project has been tested with a number of different weights. You can download some of them from the following locations:

*   **For SafeTensors (BF16 format expected):**
    *   **TinyLlama 1.1B Chat v1.0:** [TinyLlama/TinyLlama-1.1B-Chat-v1.0 on Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
    *   **Llama-2-7-b-hf:** [meta-llama/Llama-2-7b-hf on Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-hf/)
    *   Ensure you download `config.json`, `tokenizer.json`, and `model.safetensors`

*   **For GGUF:**
    *   **TinyLlama 1.1B Chat v1.0 (Q8_0 GGUF):** [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF - Q8_0.gguf](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf)
    *   **Llama 2 7B (Q8_0 GGUF):** [TheBloke/Llama-2-7B-GGUF - llama-2-7b.Q8_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q8_0.gguf)
    *   **Llama 3 8B (Q4_K_M GGUF):** [QuantFactory/Llama-3-8B-GGUF - llama-3-8b.Q4_K_M.gguf](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF/blob/main/Meta-Llama-3-8B.Q4_K_M.gguf)
    *   When using GGUF files, typically only the `.gguf` file itself is needed, as it should contain the necessary metadata. However, for some GGUF files or if using the `tinyllama` executable with a GGUF that doesn't fully embed tokenizer information, ensure a `tokenizer.json` (compatible with Llama 2) is present in the same directory or the model directory specified.


As a reminder, place the downloaded GGUF files in a directory (e.g., `models/`) or directly provide the path to the `tinyllama` executable.

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

The `tinyllama` executable is the main command-line interface for direct interaction:

```bash
./build/tinyllama <model_path> <tokenizer_path> <num_threads> <prompt|chat> \
                  [--system-prompt <system_prompt_string>] \
                  [initial_user_prompt] \
                  [max_tokens] \
                  [n_gpu_layers] \
                  [use_mmap] \
                  [temperature]
```

**Arguments for `tinyllama` executable:**

*   `<model_path>`: Path to the model file (.gguf) or directory (SafeTensors).
*   `<tokenizer_path>`: Path to the tokenizer file.
*   `<num_threads>`: Number of threads to use for generation.
*   `<prompt|chat>`: Mode of operation.
    *   `prompt`: Single prompt generation, then exits.
    *   `chat`: Interactive chat mode.
*   `--system-prompt <system_prompt_string>` (Optional): Provides a system-level instruction to the model. Enclose in quotes if it contains spaces. Default: Empty.
*   `initial_user_prompt` (Optional): The first user message in `chat` mode, or the main prompt in `prompt` mode. Default: "Hello, world!".
*   `max_tokens` (Optional): Maximum number of new tokens to generate. Default: 256.
*   `n_gpu_layers` (Optional): Number of layers to offload to GPU (-1 for all, 0 for none). Default: -1 (all layers on GPU if available).
*   `use_mmap` (Optional): Use memory-mapping for GGUF files ('true' or 'false'). Default: true.
*   `temperature` (Optional): Sampling temperature (e.g., 0.1). Lower is more deterministic. Default: 0.1.

**Example with System Prompt:**

```bash
./build/tinyllama ./models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./models/tokenizer.json 4 chat --system-prompt "You are a pirate." "Avast ye, what be the news?"
```

For detailed insight into the operations performed by the executables (e.g., `tinyllama`, `tinyllama_server`), you can inspect the `debugging.log` file generated in the application's working directory. This log provides a step-by-step account of model loading, tokenization, and generation processes.

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

# Initialize the session
# For GGUF, tokenizer_path can often be the same as model_path if tokenizer is embedded,
# or a separate tokenizer.model/tokenizer.json.
# For SafeTensors, model_path is the directory containing model.safetensors, config.json, tokenizer.json,
# and tokenizer_path should point to the tokenizer.json in that directory.
session = tinyllama_cpp.TinyLlamaSession(model_path, tokenizer_path, num_gpu_layers=-1) # Use -1 for all GPU layers if CUDA enabled

# Define prompts
user_prompt = "What is the capital of France?"
system_prompt_optional = "You are a helpful geography expert." # Optional

# Generate text
# The `apply_q_a_format` parameter defaults to `true`. This means Q:A style formatting 
# (e.g., "Q: [prompt]\\nA:") is applied by default if the loaded model is not Llama 3 
# and does not have an explicit GGUF chat template. This is often preferred for Llama 2 models.
# Set to `False` if you want to use a raw prompt for such models.
# If a GGUF or Llama 3 chat template is active, that template takes precedence.
response = session.generate(
    prompt=user_prompt, 
    steps=64, 
    temperature=0.7,
    system_prompt=system_prompt_optional # Pass the system prompt here
    # apply_q_a_format=True # This is now the default
)

print(f"User: {user_prompt}")
if system_prompt_optional:
    print(f"System: {system_prompt_optional}")
print(f"AI: {response}")

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
        system_prompt=system_prompt_optional, # Maintain system prompt across turns if desired
        # apply_q_a_format=True # Default, applies Q:A if no GGUF/Llama3 template
    )
    print(f"AI: {chat_response}")
```

Refer to your Python bindings implementation (`bindings.cpp` and `tinyllama_cpp/__init__.py`) for the exact classes and methods available.

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