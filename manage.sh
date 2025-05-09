#!/bin/bash

# Exit on error
set -e

# --- Configuration & Defaults ---
DEFAULT_BUILD_TYPE="Release"
DEFAULT_HAS_CUDA="ON"
DEFAULT_MODEL_DIR="data"
DEFAULT_SERVER_HOST="localhost"
DEFAULT_SERVER_PORT="8080"
DEFAULT_RELEASE_VERSION="0.1.0"
FORMAT_TOOL="clang-format"
DOXYGEN_CONFIG_FILE="Doxyfile"
PROJECT_ROOT_DIR=$(pwd) # Assuming script is run from project root

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
    echo ""
    echo "  run-chat     Run the command-line chat client."
    echo "               Options:"
    echo "                 --model-dir <path>          (default: ${DEFAULT_MODEL_DIR})"
    echo ""
    echo "  format       Format C++/CUDA source code using ${FORMAT_TOOL}."
    echo "               (Assumes .clang-format file in project root)"
    echo ""
    echo "  docs         Generate documentation using Doxygen."
    echo "               (Assumes ${DOXYGEN_CONFIG_FILE} in project root)"
    echo ""
    echo "  package      Package a release tarball."
    echo "               Options:"
    echo "                 --version <semver>          (default: ${DEFAULT_RELEASE_VERSION})"
    echo "                 --build-type <Release|Debug> (default: Release, for packaging)"
    echo ""
    echo "  help         Show this help message."
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

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-dir) model_dir="$2"; shift 2 ;;
            --host) server_host="$2"; shift 2 ;;
            --port) server_port="$2"; shift 2 ;;
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
    "$executable_path" "$model_dir" "$server_port" "$server_host"
}

do_run_chat() {
    local model_dir="${DEFAULT_MODEL_DIR}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-dir) model_dir="$2"; shift 2 ;;
            *) error "Unknown option for run-chat: $1"; usage ;;
        esac
    done

    local executable_path="${PROJECT_ROOT_DIR}/build/tinyllama"
    if [ ! -f "$executable_path" ]; then
        error "Chat client executable not found at $executable_path. Please build the project first."
    fi
    log "Starting chat client from $executable_path..."
    log "Model directory/path: $model_dir"
    "$executable_path" "$model_dir"
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

    local doxy_output_dir="docs/html" # Common default, adjust if your Doxyfile is different
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
shift # Remove command from arguments list, rest are options for the command

case $COMMAND in
    build) do_build "$@" ;;
    clean) do_clean "$@" ;;
    run-server) do_run_server "$@" ;;
    run-chat) do_run_chat "$@" ;;
    format) do_format "$@" ;;
    docs) do_generate_docs "$@" ;;
    package) do_package_release "$@" ;;
    help|--help|-h) usage ;;
    *)
        echo "[ERROR] Unknown command: $COMMAND"
        usage
        ;;
esac

exit 0 