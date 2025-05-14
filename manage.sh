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
    echo "                 --temperature <float>        (default: ${DEFAULT_TEMPERATURE}) (Note: Currently uses C++ default)"
    echo "                 --top-k <int>               (default: ${DEFAULT_TOP_K}) (Note: Currently uses C++ default)"
    echo "                 --top-p <float>             (default: ${DEFAULT_TOP_P}) (Note: Currently uses C++ default)"
    echo "                 --prompt <text>             (default: interactive mode)"
    echo "                 --n-gpu-layers <int>        (default: ${DEFAULT_N_GPU_LAYERS}, -1 for all on GPU)"
    echo "                 --mmap <true|false>          (default: ${DEFAULT_USE_MMAP})"
    echo ""
    echo "  run-prompt   Run the C++ model with a single prompt and exit."
    echo "               Options:"
    echo "                 --model-dir <path>          (default: ${DEFAULT_MODEL_DIR})"
    echo "                 --tokenizer <path>          (default: ${DEFAULT_TOKENIZER_PATH})"
    echo "                 --prompt <text>             (default: ${CURRENT_INTERACTIVE_PROMPT})"
    echo "                 --steps <num>               (default: ${MAX_TOKENS_SERVER})"
    echo "                 --threads <num>             (default: ${DEFAULT_THREADS})"
    echo "                 --temperature <float>       (default: ${DEFAULT_TEMPERATURE})"
    echo "                 --n-gpu-layers <int>        (default: ${DEFAULT_N_GPU_LAYERS}, -1 for all on GPU)"
    echo "                 --mmap <true|false>          (default: ${DEFAULT_USE_MMAP})"
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

    echo "Starting server with: Model=${model_dir}, Host=${server_host}, Port=${server_port}, N_GPU_Layers=${n_gpu_layers_arg}, Use_Mmap=${use_mmap_arg}, No_Log=${no_log_flag}"
    LD_LIBRARY_PATH=./build/lib ./build/bin/main "${model_dir}" "${server_port}" "${server_host}" "${n_gpu_layers_arg}" "${use_mmap_arg}" "${no_log_flag}" > "${SERVER_LOG_FILE}" 2>&1 &
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
    local steps_arg="64" # Corresponds to max_tokens in main.cpp
    local n_gpu_layers_arg="${DEFAULT_N_GPU_LAYERS}"
    local use_mmap_arg="${DEFAULT_USE_MMAP}"

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
            *)
            error "Unknown option for run-chat: $1"; usage ;;
        esac
    done

    local executable_path="${PROJECT_ROOT_DIR}/build/bin/main" # Adjusted to use the main executable
    if [ ! -f "$executable_path" ]; then
        # Fallback for older build structures or different executable name if needed
        executable_path="${PROJECT_ROOT_DIR}/build/tinyllama"
        if [ ! -f "$executable_path" ]; then
            error "Chat client executable not found at ./build/bin/main or ./build/tinyllama. Please build the project first."
        fi
    fi

    log "Starting chat client from $executable_path..."
    log "  Model Path: $model_dir_arg"
    log "  Tokenizer Path: $tokenizer_path_arg"
    log "  Threads: $threads_arg"
    log "  N GPU Layers: $n_gpu_layers_arg"
    log "  Use Mmap: $use_mmap_arg"
    log "  Prompt: ${prompt_arg:-'(interactive)'}"
    log "  Max Tokens (steps): $steps_arg"
    log "  (Note: Temperature, Top-K, Top-P from manage.sh are not currently passed to C++ main)"

    local mode_for_main="chat"
    
    # Construct arguments for main.cpp
    # main.cpp expects: <model_path> <tokenizer_path> <num_threads> <mode> <initial_prompt> <max_tokens> <n_gpu_layers> <use_mmap>
    local exec_args_for_cpp=(
        "$executable_path"
        "$model_dir_arg"
        "$tokenizer_path_arg"
        "$threads_arg"
        "$mode_for_main"
        "$prompt_arg" # main.cpp's chat loop handles if this is empty for interactive
        "$steps_arg"
        "$n_gpu_layers_arg"
        "$use_mmap_arg"
    )

    echo "Invoking C++ main: ${exec_args_for_cpp[*]}"
    LD_LIBRARY_PATH=./build/lib "${exec_args_for_cpp[@]}"
}

do_run_prompt() {
    local model_path_arg="" # New variable for the model path
    local tokenizer_path_arg_base="$DEFAULT_TOKENIZER_PATH"
    local prompt_arg="$DEFAULT_PROMPT"
    local steps_arg="$DEFAULT_STEPS"
    local threads_arg="$DEFAULT_THREADS"
    local temperature_arg="$DEFAULT_TEMPERATURE"
    local n_gpu_layers_arg="$DEFAULT_N_GPU_LAYERS"
    local use_mmap_arg="$DEFAULT_USE_MMAP"
    # local model_dir_provided=false # Flag to track if --model-dir was used (optional if we simplify --model-dir handling)

    shift # Get past "run-prompt"

    # Check if the first argument is a model path (GGUF file or directory)
    if [[ "$#" -gt 0 && "$1" != -* ]]; then
        model_path_arg="$1"
        shift # Consume the model path argument
    fi

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --model-dir) # This option is primarily for providing a directory
                if [ -z "$2" ]; then error "Missing value for --model-dir"; fi
                if [ -n "$model_path_arg" ]; then # If model_path_arg was set positionally
                    error "Cannot specify both a positional model path ('$model_path_arg') and --model-dir ('$2'). Please provide only one."
                fi
                model_path_arg="$2" # Treat --model-dir as setting the model_path_arg
                # model_dir_provided=true # (optional flag)
                shift # Consume --model-dir
                shift # Consume its value
                ;;
            --tokenizer)
                if [ -z "$2" ]; then error "Missing value for --tokenizer"; fi
                tokenizer_path_arg_base="$2"
                shift # Consume --tokenizer
                shift # Consume its value
                ;;
            --prompt)
                if [ -z "$2" ]; then error "Missing value for --prompt"; fi
                prompt_arg="$2"
                shift # Consume --prompt
                shift # Consume its value
                ;;
            --steps)
                if [ -z "$2" ]; then error "Missing value for --steps"; fi
                steps_arg="$2"
                shift # Consume --steps
                shift # Consume its value
                ;;
            --threads)
                if [ -z "$2" ]; then error "Missing value for --threads"; fi
                threads_arg="$2"
                shift # Consume --threads
                shift # Consume its value
                ;;
            --temperature)
                if [ -z "$2" ]; then error "Missing value for --temperature"; fi
                temperature_arg="$2"
                shift # Consume --temperature
                shift # Consume its value
                ;;
            --n-gpu-layers)
                if [ -z "$2" ]; then error "Missing value for --n-gpu-layers"; fi
                n_gpu_layers_arg="$2"
                shift # Consume --n-gpu-layers
                shift # Consume its value
                ;;
            --mmap)
                if [ -z "$2" ]; then error "Missing value for --mmap"; fi
                use_mmap_arg="$2"
                shift # Consume --mmap
                shift # Consume its value
                ;;
            -*)
                # Handle unknown options that start with -
                echo "Unknown option for run-prompt: $1"
                usage
                exit 1
                ;;
            *)
                # This case should ideally not be hit if the model path is consumed before the loop
                # and all other arguments are options.
                echo "Unexpected argument for run-prompt: $1. Model path should be the first argument or specified with --model-dir."
                usage
                exit 1
                ;;
        esac
    done

    # If no model path was provided positionally or via --model-dir, use default from DEFAULT_MODEL_DIR_CHAT
    if [ -z "$model_path_arg" ]; then
        log "No model path specified by user, defaulting to model directory: $DEFAULT_MODEL_DIR_CHAT"
        model_path_arg="$DEFAULT_MODEL_DIR_CHAT" # Default to directory if nothing else provided
    fi

    local executable_path="${PROJECT_ROOT_DIR}/build/bin/main"
    if [ ! -f "$executable_path" ]; then
        executable_path="${PROJECT_ROOT_DIR}/build/tinyllama" # Fallback for older structure
        if [ ! -f "$executable_path" ]; then
            error "Main executable not found at ./build/bin/main or ./build/tinyllama. Please build the project first."
        fi
    fi
    log "Using executable: $executable_path"

    local tokenizer_path_arg
    # Determine tokenizer_path_arg based on model_path_arg
    if [ -d "$model_path_arg" ]; then # If model_path_arg is a directory
        # If user specified a base for tokenizer (e.g. --tokenizer tokenizer.bin) use it relative to model_path_arg
        # Otherwise, if tokenizer_path_arg_base is the default (empty), form path like "data/tokenizer.model"
        if [ -n "$tokenizer_path_arg_base" ] && [ "$tokenizer_path_arg_base" != "$DEFAULT_TOKENIZER_PATH" ]; then
             tokenizer_path_arg="${model_path_arg}/${tokenizer_path_arg_base}"
        else # Default tokenizer name with model directory
             tokenizer_path_arg="${model_path_arg}/tokenizer.model"
        fi
    elif [ -f "$model_path_arg" ]; then # If model_path_arg is a file (e.g., GGUF)
        if [ "$tokenizer_path_arg_base" = "$DEFAULT_TOKENIZER_PATH" ] || [ -z "$tokenizer_path_arg_base" ]; then
            # If no explicit tokenizer path is given, or it's the default (empty string),
            # pass "tokenizer.model". The C++ app will look for a real file with this name
            # next to the GGUF, or use the GGUF's internal tokenizer.
            tokenizer_path_arg="tokenizer.model"
        else
            # User explicitly provided a tokenizer path, use that.
            tokenizer_path_arg="$tokenizer_path_arg_base"
        fi
    else
        # If model_path_arg is not a directory and not a file, but was specified (not default)
        if [ "$model_path_arg" != "$DEFAULT_MODEL_DIR_CHAT" ]; then
             error "Model path '$model_path_arg' is not a valid file or directory."
        else # It's the default directory which might not exist yet, which is fine for --model-dir
             log "Default model directory '$model_path_arg' may not exist yet. This is acceptable if the C++ application creates it or expects it."
             # For default directory, construct tokenizer path as before
             tokenizer_path_arg="${model_path_arg}/tokenizer.model"
        fi
    fi
    
    echo "Running prompt mode..."
    echo "Model: $model_path_arg"
    echo "Tokenizer: $tokenizer_path_arg"
    echo "Threads: $threads_arg"
    echo "Prompt: $prompt_arg"
    echo "Steps: $steps_arg"
    echo "Temperature: $temperature_arg"
    echo "N GPU Layers: $n_gpu_layers_arg"
    echo "Use Mmap: $use_mmap_arg"

    # Order: model_path, tokenizer_path, num_threads, mode, initial_prompt_string, max_tokens, n_gpu_layers, use_mmap, temperature
    local exec_args_for_cpp=("$executable_path" "$model_path_arg" "$tokenizer_path_arg" "$threads_arg" "prompt" "$prompt_arg" "$steps_arg" "$n_gpu_layers_arg" "$use_mmap_arg" "$temperature_arg")
    
    echo "Executing: ${exec_args_for_cpp[@]}"
    LD_LIBRARY_PATH=./build/lib "${exec_args_for_cpp[@]}"
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
    format) do_format ;;
    docs) do_generate_docs ;;
    docs-serve) do_docs_serve ;;
    docs-clean) do_docs_clean ;;
    package) do_package_release "$@" ;;
    help|--help|-h) usage ;;
    *)
        echo "[ERROR] Unknown command: $COMMAND"
        usage
        ;;
esac

exit 0 