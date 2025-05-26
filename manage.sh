#!/bin/bash

# Exit on error
set -e

# --- Configuration & Defaults ---
DEFAULT_BUILD_TYPE="Release"
DEFAULT_HAS_CUDA="ON"
DEFAULT_MODEL_DIR="data"
DEFAULT_SERVER_HOST="localhost"
DEFAULT_SERVER_PORT="8080"
DEFAULT_N_GPU_LAYERS="-1" # Default for N_GPU_LAYERS (-1 for auto/all)
DEFAULT_RELEASE_VERSION="1.0.14"
DEFAULT_TEMPERATURE="0.1"
DEFAULT_TOP_K="40"
DEFAULT_TOP_P="0.9"
DEFAULT_USE_KV_QUANT="false" # Default for KVCache Quantization
DEFAULT_USE_BATCH_GENERATION="false" # Default for Batch Generation
FORMAT_TOOL="clang-format"
DOXYGEN_CONFIG_FILE="Doxyfile"
PROJECT_ROOT_DIR=$(pwd) # Assuming script is run from project root

DEFAULT_MODEL_PATH=""
DEFAULT_TOKENIZER_PATH=""
DEFAULT_THREADS=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
DEFAULT_USE_MMAP="true"
DEFAULT_MODEL_DIR_CHAT="${DEFAULT_MODEL_DIR}"

CURRENT_INTERACTIVE_PROMPT=""
MAX_TOKENS_SERVER=1024

# --- Helper Functions ---
log() {
    echo "[INFO] $1"
}

error() {
    echo "[ERROR] $1" >&2
    exit 1
}

usage() {
    echo "TinyLlama.cpp Project Management Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build        Build the project."
    echo "               Options:"
    echo "                 --build-type <Release|Debug> (default: ${DEFAULT_BUILD_TYPE})"
    echo "                 --cuda <ON|OFF>             (default: ${DEFAULT_HAS_CUDA})"
    echo ""
    echo "  clean        Clean build artifacts and generated documentation."
    echo ""
    echo "  run-server   Run the chat server."
    echo "               Options:"
    echo "                 --model-dir <path>          (default: ${DEFAULT_MODEL_DIR})"
    echo "                 --host <hostname>           (default: ${DEFAULT_SERVER_HOST})"
    echo "                 --port <port_number>        (default: ${DEFAULT_SERVER_PORT})"
    echo "                 --n-gpu-layers <int>        (default: ${DEFAULT_N_GPU_LAYERS}, -1 for all on GPU)"
    echo "                 --mmap <true|false>          (default: ${DEFAULT_USE_MMAP})"
    echo "                 --no-log                    Disable logging to file for server mode (logs to console only)"
    echo ""
    echo "  run-chat     Run the command-line chat client."
    echo "               Options:"
    echo "                 --model-dir <path>          (default: ${DEFAULT_MODEL_DIR})"
    echo "                 --tokenizer <path>        (default: '${DEFAULT_TOKENIZER_PATH}' or auto-detect from model_dir)"
    echo "                 --threads <num>             (default: ${DEFAULT_THREADS})"
    echo "                 --system-prompt <text>    (Optional) System prompt to guide the model."
    echo "                 --temperature <float>        (default: ${DEFAULT_TEMPERATURE}) (Note: Currently uses C++ default)"
    echo "                 --top-k <int>               (default: ${DEFAULT_TOP_K}) (Note: Currently uses C++ default)"
    echo "                 --top-p <float>             (default: ${DEFAULT_TOP_P}) (Note: Currently uses C++ default)"
    echo "                 --prompt <text>             (default: interactive mode)"
    echo "                 --n-gpu-layers <int>        (default: ${DEFAULT_N_GPU_LAYERS}, -1 for all on GPU)"
    echo "                 --mmap <true|false>          (default: ${DEFAULT_USE_MMAP})"
    echo "                 --use-kv-quant <true|false> (default: ${DEFAULT_USE_KV_QUANT})"
    echo "                 --use-batch-gen <true|false> (default: ${DEFAULT_USE_BATCH_GENERATION})"
    echo ""
    echo "  run-prompt   Run the C++ model with a single prompt and exit."
    echo "               Options:"
    echo "                 --model-dir <path>          (default: ${DEFAULT_MODEL_DIR})"
    echo "                 --tokenizer <path>          (default: ${DEFAULT_TOKENIZER_PATH})"
    echo "                 --system-prompt <text>    (Optional) System prompt to guide the model."
    echo "                 --prompt <text>             (default: ${CURRENT_INTERACTIVE_PROMPT})"
    echo "                 --steps <num>               (default: ${MAX_TOKENS_SERVER})"
    echo "                 --threads <num>             (default: ${DEFAULT_THREADS})"
    echo "                 --temperature <float>       (default: ${DEFAULT_TEMPERATURE})"
    echo "                 --top-k <int>               (default: ${DEFAULT_TOP_K})"
    echo "                 --top-p <float>             (default: ${DEFAULT_TOP_P})"
    echo "                 --n-gpu-layers <int>        (default: ${DEFAULT_N_GPU_LAYERS}, -1 for all on GPU)"
    echo "                 --mmap <true|false>          (default: ${DEFAULT_USE_MMAP})"
    echo "                 --use-kv-quant <true|false> (default: ${DEFAULT_USE_KV_QUANT})"
    echo "                 --use-batch-gen <true|false> (default: ${DEFAULT_USE_BATCH_GENERATION})"
    echo ""
    echo "  run-batch    Run multiple prompts in parallel using batch processing."
    echo "               Options:"
    echo "                 --model-dir <path>          (default: ${DEFAULT_MODEL_DIR})"
    echo "                 --tokenizer <path>          (default: ${DEFAULT_TOKENIZER_PATH})"
    echo "                 --prompts <\"prompt1\" \"prompt2\" ...> (Required: Multiple prompts in quotes)"
    echo "                 --system-prompt <text>     (Optional) System prompt to guide the model."
    echo "                 --steps <num>               (default: ${MAX_TOKENS_SERVER})"
    echo "                 --threads <num>             (default: ${DEFAULT_THREADS})"
    echo "                 --temperature <float>       (default: ${DEFAULT_TEMPERATURE})"
    echo "                 --top-k <int>               (default: ${DEFAULT_TOP_K})"
    echo "                 --top-p <float>             (default: ${DEFAULT_TOP_P})"
    echo "                 --n-gpu-layers <int>        (default: ${DEFAULT_N_GPU_LAYERS}, -1 for all on GPU)"
    echo "                 --mmap <true|false>         (default: ${DEFAULT_USE_MMAP})"
    echo "                 --use-kv-quant <true|false> (default: ${DEFAULT_USE_KV_QUANT})"
    echo "                 --max-batch-size <int>      (default: 8)"
    echo ""
    echo "  format       Format C++/CUDA source code using ${FORMAT_TOOL}."
    echo "               (Assumes .clang-format file in project root)"
    echo ""
    echo "  docs         Generate documentation using Doxygen."
    echo "               (Assumes ${DOXYGEN_CONFIG_FILE} in project root)"
    echo ""
    echo "  docs-serve   Start a static server for viewing documentation."
    echo "               (Serves the docs/html directory on http://localhost:8000)"
    echo ""
    echo "  docs-clean   Remove generated documentation."
    echo ""
    echo "  package      Package a release tarball."
    echo "               Options:"
    echo "                 --version <semver>          (default: ${DEFAULT_RELEASE_VERSION})"
    echo "                 --build-type <Release|Debug> (default: Release, for packaging)"
    echo ""
    echo "  install      Install the Python package."
    echo "               Options:"
    echo "                 --gpu                       Enable GPU support (CUDA)"
    echo "                 --cpu                       CPU-only mode (default)"
    echo ""
    echo "  help         Show this help message."
    echo ""
    echo "  --n-gpu-layers <num> Number of layers to offload to GPU (-1 for all, 0 for none, default: ${DEFAULT_N_GPU_LAYERS})"
    echo "  --mmap <true|false>  Use mmap for GGUF files (default: ${DEFAULT_USE_MMAP})"
    echo "  --no-log             Disable logging to file for server mode (logs to console only)"
    echo ""
    exit 0
}

# --- Task Functions ---

do_build() {
    local build_type="${DEFAULT_BUILD_TYPE}"
    local has_cuda="${DEFAULT_HAS_CUDA}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --build-type) build_type="$2"; shift 2 ;;
            --cuda) has_cuda="$2"; shift 2 ;;
            *) error "Unknown option for build: $1"; usage ;;
        esac
    done

    log "Starting build process..."
    log "Build type: ${build_type}"
    log "CUDA enabled: ${has_cuda}"

    mkdir -p "${PROJECT_ROOT_DIR}/build"
    cd "${PROJECT_ROOT_DIR}/build" || error "Failed to cd into build directory"

    cmake_args=("-DCMAKE_BUILD_TYPE=${build_type}" "-DHAS_CUDA=${has_cuda}")

    log "Configuring CMake with: cmake .. ${cmake_args[*]}"
    cmake .. "${cmake_args[@]}"
    if [ $? -ne 0 ]; then error "CMake configuration failed."; fi

    log "Building project..."
    num_procs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
    make -j"${num_procs}"
    if [ $? -ne 0 ]; then error "Build failed."; fi

    log "Build successful. Executables should be in ./build/"
    cd "${PROJECT_ROOT_DIR}" || error "Failed to cd back to project root"
}

do_clean() {
    log "Cleaning project..."
    if [ -d "${PROJECT_ROOT_DIR}/build" ]; then
        log "Removing build directory..."
        rm -rf "${PROJECT_ROOT_DIR}/build"
    else
        log "Build directory not found. Nothing to clean from there."
    fi

    if [ -d "${PROJECT_ROOT_DIR}/docs/html" ]; then
        log "Removing docs/html directory..."
        rm -rf "${PROJECT_ROOT_DIR}/docs/html"
    fi
    log "Clean complete."
}

do_run_server() {
    local model_dir="${DEFAULT_MODEL_DIR}"
    local server_host="${DEFAULT_SERVER_HOST}"
    local server_port="${DEFAULT_SERVER_PORT}"
    local n_gpu_layers="${DEFAULT_N_GPU_LAYERS}"
    local use_mmap="${DEFAULT_USE_MMAP}"
    local no_log="false"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-dir) model_dir="$2"; shift 2 ;;
            --host) server_host="$2"; shift 2 ;;
            --port) server_port="$2"; shift 2 ;;
            --n-gpu-layers) n_gpu_layers="$2"; shift 2 ;;
            --mmap) use_mmap="$2"; shift 2 ;;
            --no-log) no_log="true"; shift 1 ;;
            *) error "Unknown option for run-server: $1"; usage ;;
        esac
    done

    local executable_path="${PROJECT_ROOT_DIR}/build/tinyllama_server"
    if [ ! -f "$executable_path" ]; then
        error "Server executable not found at $executable_path. Please build the project first."
    fi
    log "Starting server from $executable_path..."
    log "Model directory: $model_dir"
    log "Host: $server_host"
    log "Port: $server_port"
    log "N GPU Layers: $n_gpu_layers"
    log "Use Mmap: $use_mmap"
    log "No Log: $no_log"

    local n_gpu_layers_arg="${n_gpu_layers}"
    local use_mmap_arg="${use_mmap}"
    local no_log_flag="${no_log}"

    echo "Starting server with: Executable=${executable_path}, Model=${model_dir}, Host=${server_host}, Port=${server_port}, N_GPU_Layers=${n_gpu_layers_arg}, Use_Mmap=${use_mmap_arg}, No_Log=${no_log_flag}"
    LD_LIBRARY_PATH=./build/lib "${executable_path}" "${model_dir}" "${server_port}" "${server_host}" "${n_gpu_layers_arg}" "${use_mmap_arg}" "${no_log_flag}" > "${SERVER_LOG_FILE}" 2>&1 &
    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"
}

do_run_chat() {
    local model_dir_arg="${DEFAULT_MODEL_DIR}"
    local tokenizer_path_arg="${DEFAULT_TOKENIZER_PATH}"
    local threads_arg="${DEFAULT_THREADS}"
    local temperature_arg="${DEFAULT_TEMPERATURE}"
    local top_k_arg="${DEFAULT_TOP_K}"
    local top_p_arg="${DEFAULT_TOP_P}"
    local prompt_arg=""
    local system_prompt_arg=""
    local steps_arg="64" # Corresponds to max_tokens in main.cpp
    local n_gpu_layers_arg="${DEFAULT_N_GPU_LAYERS}"
    local use_mmap_arg="${DEFAULT_USE_MMAP}"
    local use_kv_quant_arg="${DEFAULT_USE_KV_QUANT}"
    local use_batch_generation_arg="${DEFAULT_USE_BATCH_GENERATION}"
    local pass_through_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-dir)
            model_dir_arg="$2"
            shift; shift;;
            --tokenizer)
            tokenizer_path_arg="$2"
            shift; shift;;
            --threads)
            threads_arg="$2"
            shift; shift;;
            --system-prompt)
            system_prompt_arg="$2"
            shift; shift;;
            --temperature)
            temperature_arg="$2"
            shift; shift;;
            --top-k)
            top_k_arg="$2"
            shift; shift;;
            --top-p)
            top_p_arg="$2"
            shift; shift;;
            --prompt)
            prompt_arg="$2"
            shift; shift;;
            --n-gpu-layers)
            n_gpu_layers_arg="$2"
            shift; shift;;
            --mmap)
            use_mmap_arg="$2"
            shift; shift;;
            --use-kv-quant)
            use_kv_quant_arg="$2"
            shift; shift;;
            --use-batch-gen)
            use_batch_generation_arg="$2"
            shift; shift;;
            *)
            error "Unknown option for run-chat: $1"; usage ;;
        esac
    done

    local executable_path="${PROJECT_ROOT_DIR}/build/main"
    if [ ! -f "$executable_path" ]; then
        executable_path="${PROJECT_ROOT_DIR}/build/tinyllama"
        if [ ! -f "$executable_path" ]; then
            error "Main executable not found at ${PROJECT_ROOT_DIR}/build/main or ${PROJECT_ROOT_DIR}/build/tinyllama. Please build the project first."
        fi
    fi

    log "Starting chat client from $executable_path..."
    log "  Model Path: $model_dir_arg"
    log "  Tokenizer Path: $tokenizer_path_arg"
    log "  Threads: $threads_arg"
    log "  N GPU Layers: $n_gpu_layers_arg"
    log "  Use Mmap: $use_mmap_arg"
    log "  Temperature: $temperature_arg"
    log "  Top-K: $top_k_arg"
    log "  Top-P: $top_p_arg"
    log "  Prompt: ${prompt_arg:-'(interactive)'}"
    log "  Max Tokens (steps): $steps_arg"

    local mode_for_main="chat"
    
    pass_through_args+=("${model_dir_arg}")
    pass_through_args+=("${tokenizer_path_arg}")
    pass_through_args+=("${threads_arg}")
    pass_through_args+=("${mode_for_main}")

    if [ -n "$system_prompt_arg" ]; then
        pass_through_args+=("--system-prompt" "${system_prompt_arg}")
    fi
    if [ -n "$prompt_arg" ]; then
        pass_through_args+=("${prompt_arg}")
    fi
    
    pass_through_args+=("--max-tokens" "${steps_arg}")
    pass_through_args+=("--n-gpu-layers" "${n_gpu_layers_arg}")
    pass_through_args+=("--use-mmap" "${use_mmap_arg}")
    pass_through_args+=("--use-kv-quant" "${use_kv_quant_arg}")
    pass_through_args+=("--use-batch-generation" "${use_batch_generation_arg}")
    pass_through_args+=("--temperature" "${temperature_arg}")
    pass_through_args+=("--top-k" "${top_k_arg}")
    pass_through_args+=("--top-p" "${top_p_arg}")

    echo "Invoking C++ main: $executable_path ${pass_through_args[*]}"
    LD_LIBRARY_PATH=./build/lib "$executable_path" "${pass_through_args[@]}"
}

do_run_prompt() {
    local model_dir_arg="${DEFAULT_MODEL_DIR}"
    local tokenizer_path_arg="${DEFAULT_TOKENIZER_PATH}"
    local prompt_arg="${CURRENT_INTERACTIVE_PROMPT}"
    local system_prompt_arg=""
    local steps_arg="${MAX_TOKENS_SERVER}"
    local threads_arg="${DEFAULT_THREADS}"
    local temperature_arg="${DEFAULT_TEMPERATURE}"
    local top_k_arg="${DEFAULT_TOP_K}"
    local top_p_arg="${DEFAULT_TOP_P}"
    local n_gpu_layers_arg="${DEFAULT_N_GPU_LAYERS}"
    local use_mmap_arg="${DEFAULT_USE_MMAP}"
    local use_kv_quant_arg="${DEFAULT_USE_KV_QUANT}"
    local use_batch_generation_arg="${DEFAULT_USE_BATCH_GENERATION}"
    local pass_through_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-dir)
            model_dir_arg="$2"
            shift; shift;;
            --tokenizer)
            tokenizer_path_arg="$2"
            shift; shift;;
            --system-prompt)
            system_prompt_arg="$2"
            shift; shift;;
            --prompt)
            prompt_arg="$2"
            shift; shift;;
            --steps)
            steps_arg="$2"
            shift; shift;;
            --threads)
            threads_arg="$2"
            shift; shift;;
            --temperature)
            temperature_arg="$2"
            shift; shift;;
            --top-k)
            top_k_arg="$2"
            shift; shift;;
            --top-p)
            top_p_arg="$2"
            shift; shift;;
            --n-gpu-layers)
            n_gpu_layers_arg="$2"
            shift; shift;;
            --mmap)
            use_mmap_arg="$2"
            shift; shift;;
            --use-kv-quant)
            use_kv_quant_arg="$2"
            shift; shift;;
            --use-batch-gen)
            use_batch_generation_arg="$2"
            shift; shift;;
            *)
            error "Unknown option for run-prompt: $1"; usage ;;
        esac
    done

    local executable_path="${PROJECT_ROOT_DIR}/build/main"
    if [ ! -f "$executable_path" ]; then
        executable_path="${PROJECT_ROOT_DIR}/build/tinyllama"
        if [ ! -f "$executable_path" ]; then
            error "Main executable not found at ${PROJECT_ROOT_DIR}/build/main or ${PROJECT_ROOT_DIR}/build/tinyllama. Please build the project first."
        fi
    fi
    
    log "Starting prompt mode..."
    log "  Model Path: $model_dir_arg"
    log "  Tokenizer Path: $tokenizer_path_arg"
    log "  Threads: $threads_arg"
    log "  Temperature: $temperature_arg"
    log "  Top-K: $top_k_arg"
    log "  Top-P: $top_p_arg"
    log "  Prompt: $prompt_arg"
    log "  Max Tokens (steps): $steps_arg"

    pass_through_args+=("${model_dir_arg}")
    pass_through_args+=("${tokenizer_path_arg}")
    pass_through_args+=("${threads_arg}")
    pass_through_args+=("prompt")

    if [ -n "$system_prompt_arg" ]; then
        pass_through_args+=("--system-prompt" "${system_prompt_arg}")
    fi
    
    pass_through_args+=("${prompt_arg}")
    
    pass_through_args+=("--max-tokens" "${steps_arg}")
    pass_through_args+=("--n-gpu-layers" "${n_gpu_layers_arg}")
    pass_through_args+=("--use-mmap" "${use_mmap_arg}")
    pass_through_args+=("--use-kv-quant" "${use_kv_quant_arg}")
    pass_through_args+=("--use-batch-generation" "${use_batch_generation_arg}")
    pass_through_args+=("--temperature" "${temperature_arg}")
    pass_through_args+=("--top-k" "${top_k_arg}")
    pass_through_args+=("--top-p" "${top_p_arg}")

    echo "N GPU Layers: $n_gpu_layers_arg"
    echo "Use Mmap: $use_mmap_arg"

    echo "Executing: $executable_path ${pass_through_args[*]}"
    LD_LIBRARY_PATH=./build/lib "$executable_path" "${pass_through_args[@]}"
}

do_run_batch() {
    local model_dir_arg="${DEFAULT_MODEL_DIR}"
    local tokenizer_path_arg="${DEFAULT_TOKENIZER_PATH}"
    local system_prompt_arg=""
    local prompts_array=()
    local steps_arg="128"
    local threads_arg="${DEFAULT_THREADS}"
    local temperature_arg="${DEFAULT_TEMPERATURE}"
    local top_k_arg="${DEFAULT_TOP_K}"
    local top_p_arg="${DEFAULT_TOP_P}"
    local n_gpu_layers_arg="${DEFAULT_N_GPU_LAYERS}"
    local use_mmap_arg="${DEFAULT_USE_MMAP}"
    local use_kv_quant_arg="${DEFAULT_USE_KV_QUANT}"
    local max_batch_size_arg="8"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-dir) model_dir_arg="$2"; shift 2 ;;
            --tokenizer) tokenizer_path_arg="$2"; shift 2 ;;
            --prompts) 
                shift
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    prompts_array+=("$1")
                    shift
                done
                ;;
            --system-prompt) system_prompt_arg="$2"; shift 2 ;;
            --steps) steps_arg="$2"; shift 2 ;;
            --threads) threads_arg="$2"; shift 2 ;;
            --temperature) temperature_arg="$2"; shift 2 ;;
            --top-k) top_k_arg="$2"; shift 2 ;;
            --top-p) top_p_arg="$2"; shift 2 ;;
            --n-gpu-layers) n_gpu_layers_arg="$2"; shift 2 ;;
            --mmap) use_mmap_arg="$2"; shift 2 ;;
            --use-kv-quant) use_kv_quant_arg="$2"; shift 2 ;;
            --max-batch-size) max_batch_size_arg="$2"; shift 2 ;;
            *) error "Unknown option for run-batch: $1"; usage ;;
        esac
    done

    if [ ${#prompts_array[@]} -eq 0 ]; then
        error "No prompts provided. Use --prompts \"prompt1\" \"prompt2\" ..."
    fi

    if [ ${#prompts_array[@]} -gt "$max_batch_size_arg" ]; then
        error "Number of prompts (${#prompts_array[@]}) exceeds max batch size ($max_batch_size_arg)"
    fi

    # Auto-detect tokenizer if not specified
    if [ -z "$tokenizer_path_arg" ] || [ "$tokenizer_path_arg" = "$DEFAULT_TOKENIZER_PATH" ]; then
        tokenizer_path_arg="$model_dir_arg"
    fi

    log "Starting native C++ batch processing for ${#prompts_array[@]} prompts..."
    log "  Model Path: $model_dir_arg"
    log "  Tokenizer Path: $tokenizer_path_arg"
    log "  Threads: $threads_arg"
    log "  Steps: $steps_arg"
    log "  Temperature: $temperature_arg"
    log "  Top-K: $top_k_arg"
    log "  Top-P: $top_p_arg"
    log "  N GPU Layers: $n_gpu_layers_arg"
    log "  Use KV Quant: $use_kv_quant_arg"
    log "  Max Batch Size: $max_batch_size_arg"

    local executable_path="${PROJECT_ROOT_DIR}/build/main"
    if [ ! -f "$executable_path" ]; then
        executable_path="${PROJECT_ROOT_DIR}/build/tinyllama"
        if [ ! -f "$executable_path" ]; then
            error "Main executable not found at ${PROJECT_ROOT_DIR}/build/main or ${PROJECT_ROOT_DIR}/build/tinyllama. Please build the project first."
        fi
    fi

    # Build command line arguments for C++ batch mode
    local pass_through_args=()
    pass_through_args+=("${model_dir_arg}")
    pass_through_args+=("${tokenizer_path_arg}")
    pass_through_args+=("${threads_arg}")
    pass_through_args+=("batch")  # Mode

    if [ -n "$system_prompt_arg" ]; then
        pass_through_args+=("--system-prompt" "${system_prompt_arg}")
    fi

    # Add batch-specific arguments
    pass_through_args+=("--batch-prompts")
    for prompt in "${prompts_array[@]}"; do
        pass_through_args+=("${prompt}")
    done

    pass_through_args+=("--max-tokens" "${steps_arg}")
    pass_through_args+=("--n-gpu-layers" "${n_gpu_layers_arg}")
    pass_through_args+=("--use-mmap" "${use_mmap_arg}")
    pass_through_args+=("--use-kv-quant" "${use_kv_quant_arg}")
    pass_through_args+=("--temperature" "${temperature_arg}")
    pass_through_args+=("--top-k" "${top_k_arg}")
    pass_through_args+=("--top-p" "${top_p_arg}")
    pass_through_args+=("--max-batch-size" "${max_batch_size_arg}")

    log "Executing C++ batch processing..."
    echo "Command: $executable_path ${pass_through_args[*]}"
    
    cd "${PROJECT_ROOT_DIR}" || error "Failed to cd to project root"
    LD_LIBRARY_PATH=./build/lib "$executable_path" "${pass_through_args[@]}"
    
    if [ $? -ne 0 ]; then
        error "Batch processing failed"
    fi
    
    log "Batch processing completed successfully"
}

do_format() {
    if ! command -v ${FORMAT_TOOL} &> /dev/null; then
        error "${FORMAT_TOOL} could not be found. Please install it and ensure it's in your PATH."
    fi
    if [ ! -f "${PROJECT_ROOT_DIR}/.clang-format" ]; then
        log "Warning: .clang-format file not found at project root. ${FORMAT_TOOL} will use its default style."
    fi
    log "Formatting C++/CUDA source files using ${FORMAT_TOOL}..."
    find "${PROJECT_ROOT_DIR}" \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) \
        -not \( -path "${PROJECT_ROOT_DIR}/build/*" -o \
               -path "${PROJECT_ROOT_DIR}/docs/*" -o \
               -path "${PROJECT_ROOT_DIR}/.git/*" -o \
               -path "*/_deps/*" \) \
        -exec ${FORMAT_TOOL} -i {} +
    log "Formatting complete."
}

do_generate_docs() {
    if ! command -v doxygen &> /dev/null; then
        error "Doxygen could not be found. Please install it and ensure it's in your PATH."
    fi
    if [ ! -f "${PROJECT_ROOT_DIR}/${DOXYGEN_CONFIG_FILE}" ]; then
        error "Doxygen configuration file ($DOXYGEN_CONFIG_FILE) not found at project root."
    fi
    log "Generating documentation using Doxygen..."
    cd "${PROJECT_ROOT_DIR}" # Ensure Doxygen runs from the root where Doxyfile is
    doxygen "${DOXYGEN_CONFIG_FILE}"
    if [ $? -ne 0 ]; then error "Doxygen generation failed."; fi
    log "Documentation generated successfully (check Doxyfile for OUTPUT_DIRECTORY)."
}

do_docs_serve() {
    local docs_dir="${PROJECT_ROOT_DIR}/docs/html"
    
    if [ ! -d "$docs_dir" ]; then
        error "Documentation not found at $docs_dir. Please generate docs first using: $0 docs"
    fi

    # Check if directory is empty
    if [ -z "$(ls -A $docs_dir)" ]; then
        error "Documentation directory is empty. Please generate docs first using: $0 docs"
    fi

    # Check if Python is available
    if command -v python3 &> /dev/null; then
        log "Starting documentation server on http://localhost:8000"
        log "Press Ctrl+C to stop the server"
        cd "$docs_dir" && python3 -m http.server 8000 --bind 0.0.0.0
    elif command -v python &> /dev/null; then
        log "Starting documentation server on http://localhost:8000"
        log "Press Ctrl+C to stop the server"
        cd "$docs_dir" && python -m http.server 8000 --bind 0.0.0.0
    else
        error "Python not found. Please install Python to use the docs-serve feature."
    fi
}

do_docs_clean() {
    local docs_dir="${PROJECT_ROOT_DIR}/docs"
    
    if [ -d "$docs_dir" ]; then
        log "Removing documentation directory..."
        rm -rf "$docs_dir"
        log "Documentation cleaned successfully."
    else
        log "No documentation directory found at $docs_dir. Nothing to clean."
    fi
}

do_package_release() {
    local release_version="${DEFAULT_RELEASE_VERSION}"
    local package_build_type="Release" # Default to Release for packaging

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version) release_version="$2"; shift 2 ;;
            --build-type) package_build_type="$2"; shift 2 ;; # Allow override for packaging build type
            *) error "Unknown option for package: $1"; usage ;;
        esac
    done

    log "Packaging release version: ${release_version} (Build type for package: ${package_build_type})"

    # 1. Ensure project is built
    log "Ensuring project is built in '${package_build_type}' mode..."
    # Call build function with specific arguments for packaging
    do_build --build-type "${package_build_type}" --cuda "${DEFAULT_HAS_CUDA}" # Use default CUDA for packaging build

    # 2. Create staging directory
    local staging_dir_name="tinyllama_cpp_${release_version}"
    local staging_dir_path="${PROJECT_ROOT_DIR}/${staging_dir_name}"
    local archive_name="${staging_dir_name}.tar.gz"
    local archive_path="${PROJECT_ROOT_DIR}/${archive_name}"

    rm -rf "${staging_dir_path}" "${archive_path}" # Clean previous attempts
    mkdir -p "${staging_dir_path}"

    log "Copying artifacts to staging directory: ${staging_dir_path}"

    # Copy executables
    cp "${PROJECT_ROOT_DIR}/build/tinyllama" "${staging_dir_path}/" || error "Failed to copy tinyllama executable."
    cp "${PROJECT_ROOT_DIR}/build/tinyllama_server" "${staging_dir_path}/" || error "Failed to copy tinyllama_server executable."

    # Copy essential files
    cp "${PROJECT_ROOT_DIR}/README.md" "${staging_dir_path}/" || error "Failed to copy README.md"
    if [ -f "${PROJECT_ROOT_DIR}/LICENSE" ]; then
        cp "${PROJECT_ROOT_DIR}/LICENSE" "${staging_dir_path}/" || log "LICENSE file not found, skipping."
    fi

    local doxy_output_dir="docs/html"
    if [ -d "${PROJECT_ROOT_DIR}/${doxy_output_dir}" ]; then
        mkdir -p "${staging_dir_path}/docs"
        cp -r "${PROJECT_ROOT_DIR}/${doxy_output_dir}" "${staging_dir_path}/docs/" || error "Failed to copy documentation."
    else
        log "Documentation directory (${PROJECT_ROOT_DIR}/${doxy_output_dir}) not found, skipping."
    fi

    # 3. Create archive
    log "Creating release archive: ${archive_path}"
    cd "${PROJECT_ROOT_DIR}" || error "Failed to cd to project root for tar"
    tar -czvf "${archive_name}" "${staging_dir_name}"
    if [ $? -ne 0 ]; then error "Failed to create release archive."; fi

    # 4. Cleanup staging directory
    rm -rf "${staging_dir_path}"

    log "Release packaged successfully: ${archive_path}"
    cd "${PROJECT_ROOT_DIR}" || error "Failed to cd back to project root"
}

do_install() {
    local use_gpu="false"
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gpu) use_gpu="true"; shift ;;
            --cpu) use_gpu="false"; shift ;;
            *) error "Unknown option for install: $1"; usage ;;
        esac
    done

    log "Installing Python package..."
    if [ "$use_gpu" = "true" ]; then
        log "Installing with GPU (CUDA) support..."
        export TINYLLAMA_CPP_BUILD_CUDA=ON
    else
        log "Installing with CPU-only support..."
        export TINYLLAMA_CPP_BUILD_CUDA=OFF
    fi

    cd "${PROJECT_ROOT_DIR}" || error "Failed to cd to project root"
    
    if ! command -v pip &> /dev/null; then
        error "pip could not be found. Please install pip and ensure it's in your PATH."
    fi

    log "Running pip install in editable mode..."
    pip install -e . --verbose
    
    if [ $? -ne 0 ]; then
        error "Python package installation failed"
    fi
    
    log "Python package installed successfully"
}

# --- Main Script Logic ---
if [ $# -eq 0 ]; then
    usage
fi

COMMAND=$1
shift

case $COMMAND in
    build) do_build "$@" ;;
    clean) do_clean ;;
    run-server) do_run_server "$@" ;;
    run-chat) do_run_chat "$@" ;;
    run-prompt) do_run_prompt "$@" ;;
    run-batch) do_run_batch "$@" ;;
    format) do_format ;;
    docs) do_generate_docs ;;
    docs-serve) do_docs_serve ;;
    docs-clean) do_docs_clean ;;
    package) do_package_release "$@" ;;
    install) do_install "$@" ;;
    help|--help|-h) usage ;;
    *)
        echo "[ERROR] Unknown command: $COMMAND"
        usage
        ;;
esac

exit 0 