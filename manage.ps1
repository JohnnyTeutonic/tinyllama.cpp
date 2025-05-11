#Requires -Version 5.1
<#
.SYNOPSIS
    TinyLlama.cpp Project Management Script (PowerShell)
.DESCRIPTION
    Provides commands to build, clean, run, test, format, and manage the TinyLlama.cpp project.
.NOTES
    Author: Jonathan Reich
    Version: 1.0
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory=$false, Position=0)]
    [ValidateSet("build", "clean", "run-server", "run-chat", "format", "docs", "docs-serve", "docs-clean", "package", "help")]
    [string]$Command,

    [Parameter(Mandatory=$false, Position=1, ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

# --- Configuration & Defaults ---
$DefaultBuildType = "Release"
$DefaultHasCuda = "ON" # Set to "OFF" if CUDA is not intended to be used by default
$DefaultModelDir = "data"
$DefaultServerHost = "localhost"
$DefaultServerPort = "8080"
$DefaultNGpuLayers = -1 # Default for N_GPU_LAYERS (-1 for auto/all)
$DefaultReleaseVersion = "0.1.0"
$DefaultTemperature = "0.1"
$DefaultTopK = 40
$DefaultTopP = 0.9
$FormatTool = "clang-format.exe" # Ensure clang-format is in PATH or provide full path
$DoxygenConfigFile = "Doxyfile"
$ProjectRootDir = $PSScriptRoot # Assumes script is in the project root
$DefaultModelPath = ""
$DefaultTokenizerPath = ""
$DefaultThreads = (Get-WmiObject Win32_Processor).NumberOfLogicalProcessors # Or a sensible default if query fails
if (-not $DefaultThreads) { $DefaultThreads = 4 }
$DefaultUseMmap = $true

$CurrentInteractivePrompt = ""

# --- Helper Functions ---
function Log-Message {
    param ([string]$Message)
    Write-Host "[INFO] $Message"
}

function Write-ErrorAndExit {
    param ([string]$Message, [int]$ExitCode = 1)
    Write-Error "[ERROR] $Message"
    exit $ExitCode
}

function Show-Usage {
    Write-Host "TinyLlama.cpp Project Management Script (PowerShell)"
    Write-Host ""
    Write-Host "Usage: .\manage.ps1 <command> [options]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build        Build the project."
    Write-Host "               Options:"
    Write-Host "                 -BuildType <Release|Debug> (default: ${DefaultBuildType})"
    Write-Host "                 -Cuda <ON|OFF>             (default: ${DefaultHasCuda})"
    Write-Host ""
    Write-Host "  clean        Clean build artifacts and generated documentation."
    Write-Host ""
    Write-Host "  run-server   Run the chat server."
    Write-Host "               Options:"
    Write-Host "                 -ModelDir <path>          (default: ${DefaultModelDir})"
    Write-Host "                 -Host <hostname>           (default: ${DefaultServerHost})"
    Write-Host "                 -Port <port_number>        (default: ${DefaultServerPort})"
    Write-Host "                 -NGpuLayers <int>         (default: ${DefaultNGpuLayers}, -1 for all on GPU)"
    Write-Host "                 -Mmap <true|false>        (default: ${DefaultUseMmap})"
    Write-Host "                 -NoLog                  : Disable logging to file for server mode (logs to console only)."
    Write-Host ""
    Write-Host "  run-chat     Run the command-line chat client."
    Write-Host "               Options:"
    Write-Host "                 -ModelDir <path>          (default: ${DefaultModelDir})"
    Write-Host "                 -Temperature <float>        (default: ${DefaultTemperature})"
    Write-Host "                 -TopK <int>               (default: ${DefaultTopK})"
    Write-Host "                 -TopP <float>             (default: ${DefaultTopP})"
    Write-Host "                 -Prompt <text>             (default: interactive mode, uses default prompt)"
    Write-Host "                 -NGpuLayers <int>         (default: ${DefaultNGpuLayers}, -1 for all on GPU)"
    Write-Host "                 -Mmap <true|false>        (default: ${DefaultUseMmap})"
    Write-Host ""
    Write-Host "  format       Format C++/CUDA source code using ${FormatTool}."
    Write-Host "               (Assumes .clang-format file in project root)"
    Write-Host ""
    Write-Host "  docs         Generate documentation using Doxygen."
    Write-Host "               (Assumes ${DoxygenConfigFile} in project root)"
    Write-Host ""
    Write-Host "  docs-serve   Start a static server for viewing documentation (requires Python)."
    Write-Host "               (Serves the docs/html directory on http://localhost:8000)"
    Write-Host ""
    Write-Host "  docs-clean   Remove generated documentation."
    Write-Host ""
    Write-Host "  package      Package a release archive (primarily for Windows - creates a ZIP)."
    Write-Host "               Options:"
    Write-Host "                 -Version <semver>          (default: ${DefaultReleaseVersion})"
    Write-Host "                 -BuildType <Release|Debug> (default: Release, for packaging)"
    Write-Host ""
    Write-Host "  help         Show this help message."
    Write-Host ""
    exit 0
}

# --- Task Functions ---

function Invoke-Build {
    param (
        [string]$BuildType = $DefaultBuildType,
        [string]$HasCuda = $DefaultHasCuda
    )

    # Argument parsing for -BuildType, -Cuda
    $Params = $script:Arguments | ConvertFrom-StringData -Delimiter ' '
    if ($Params.BuildType) { $BuildType = $Params.BuildType }
    if ($Params.Cuda) { $HasCuda = $Params.Cuda }


    Log-Message "Starting build process..."
    Log-Message "Build type: ${BuildType}"
    Log-Message "CUDA enabled: ${HasCuda}"

    $BuildDir = Join-Path -Path $ProjectRootDir -ChildPath "build"
    if (-not (Test-Path $BuildDir)) {
        New-Item -ItemType Directory -Path $BuildDir | Out-Null
    }
    Set-Location $BuildDir

    $CmakeArgs = @("-DCMAKE_BUILD_TYPE=${BuildType}", "-DHAS_CUDA=${HasCuda}")

    Log-Message "Configuring CMake with: cmake .. $CmakeArgs"
    cmake .. $CmakeArgs
    if ($LASTEXITCODE -ne 0) { Write-ErrorAndExit "CMake configuration failed." }

    Log-Message "Building project (using --config ${BuildType})..."
    cmake --build . --config ${BuildType} --parallel $([System.Environment]::ProcessorCount)
    if ($LASTEXITCODE -ne 0) { Write-ErrorAndExit "Build failed." }

    Log-Message "Build successful. Executables should be in ./build/${BuildType}/ (or ./build/ for single-config generators)"
    Set-Location $ProjectRootDir
}

function Invoke-Clean {
    Log-Message "Cleaning project..."
    $BuildDir = Join-Path -Path $ProjectRootDir -ChildPath "build"
    if (Test-Path $BuildDir) {
        Log-Message "Removing build directory: $BuildDir"
        Remove-Item -Recurse -Force $BuildDir
    } else {
        Log-Message "Build directory not found. Nothing to clean from there."
    }

    $DocsHtmlDir = Join-Path -Path $ProjectRootDir -ChildPath "docs/html"
    if (Test-Path $DocsHtmlDir) {
        Log-Message "Removing docs/html directory: $DocsHtmlDir"
        Remove-Item -Recurse -Force $DocsHtmlDir
    }
    Log-Message "Clean complete."
}

function Invoke-RunServer {
    param (
        [string]$ModelDir = $DefaultModelDir,
        [string]$ServerHost = $DefaultServerHost,
        [string]$ServerPort = $DefaultServerPort,
        [string]$BuildType = $DefaultBuildType, # To find the executable
        [int]$NGpuLayers = $DefaultNGpuLayers,
        [string]$TokenizerPath = $DefaultTokenizerPath,
        [int]$Threads = $DefaultThreads,
        [bool]$UseMmap = $DefaultUseMmap,
        [switch]$NoLog
    )
    $Params = $script:Arguments | ConvertFrom-StringData -Delimiter ' '
    if ($Params.ModelDir) { $ModelDir = $Params.ModelDir }
    if ($Params.Host) { $ServerHost = $Params.Host }
    if ($Params.Port) { $ServerPort = $Params.Port }
    if ($Params.NGpuLayers) { $NGpuLayers = [int]$Params.NGpuLayers }
    if ($Params.TokenizerPath) { $TokenizerPath = $Params.TokenizerPath }
    if ($Params.Threads) { $Threads = [int]$Params.Threads }
    if ($Params.UseMmap) { $UseMmap = [bool]$Params.UseMmap }
    if ($Params.NoLog) { $NoLog = $Params.NoLog }
    
    # Try to determine executable path based on common CMake single/multi-config generator outputs
    $ExecutablePath = Join-Path -Path $ProjectRootDir -ChildPath "build/tinyllama_server.exe" # Common for single-config
    if (-not (Test-Path $ExecutablePath)) {
      $ExecutablePath = Join-Path -Path $ProjectRootDir -ChildPath "build/${BuildType}/tinyllama_server.exe" # Common for multi-config (e.g. MSVC)
    }
     if (-not (Test-Path $ExecutablePath)) {
      $ExecutablePath = Join-Path -Path $ProjectRootDir -ChildPath "build/Release/tinyllama_server.exe" # MSVC default
    }
     if (-not (Test-Path $ExecutablePath)) {
      $ExecutablePath = Join-Path -Path $ProjectRootDir -ChildPath "build/Debug/tinyllama_server.exe" # MSVC default
    }


    if (-not (Test-Path $ExecutablePath)) {
        Write-ErrorAndExit "Server executable not found at common paths (e.g., $ProjectRootDir\build\tinyllama_server.exe or $ProjectRootDir\build\$BuildType\tinyllama_server.exe). Please build the project first."
    }

    Log-Message "Starting server from $ExecutablePath..."
    Log-Message "Model directory: $ModelDir"
    Log-Message "Host: $ServerHost"
    Log-Message "Port: $ServerPort"
    Log-Message "N GPU Layers: $NGpuLayers"
    Log-Message "Tokenizer Path: $TokenizerPath"
    Log-Message "Threads: $Threads"
    Log-Message "Use Mmap: $UseMmap"
    Log-Message "No Log: $NoLog"
    
    $ScriptArgs = @($ModelDir, $ServerPort, $ServerHost, $NGpuLayers, [string]$UseMmap)
    if ($NoLog) {
        $ScriptArgs += "--no-log"
    }
    
    & $ExecutablePath $ScriptArgs
    if ($LASTEXITCODE -ne 0) { Write-ErrorAndExit "Server execution failed."}
}

function Invoke-RunChat {
    param (
        [string]$ModelDir = $DefaultModelDir,
        [string]$Temperature = $DefaultTemperature,
        [string]$TopK = $DefaultTopK,
        [string]$TopP = $DefaultTopP,
        [string]$Prompt = "", # Empty means interactive
        [string]$BuildType = $DefaultBuildType, # To find the executable
        [string]$Steps = "64", # Default steps for non-interactive
        [int]$NGpuLayers = $DefaultNGpuLayers,
        [string]$TokenizerPath = $DefaultTokenizerPath,
        [int]$Threads = $DefaultThreads,
        [bool]$UseMmap = $DefaultUseMmap,
        [switch]$NoLog
    )
    $Params = $script:Arguments | ConvertFrom-StringData -Delimiter ' '
    if ($Params.ModelDir) { $ModelDir = $Params.ModelDir }
    if ($Params.Temperature) { $Temperature = $Params.Temperature }
    if ($Params.TopK) { $TopK = $Params.TopK }
    if ($Params.TopP) { $TopP = $Params.TopP }
    if ($Params.Prompt) { $Prompt = $Params.Prompt }
    if ($Params.NGpuLayers) { $NGpuLayers = [int]$Params.NGpuLayers }
    if ($Params.TokenizerPath) { $TokenizerPath = $Params.TokenizerPath }
    if ($Params.Threads) { $Threads = [int]$Params.Threads }
    if ($Params.UseMmap) { $UseMmap = [bool]$Params.UseMmap }
    if ($Params.NoLog) { $NoLog = $Params.NoLog }

    $ExecutablePath = Join-Path -Path $ProjectRootDir -ChildPath "build/tinyllama.exe"
     if (-not (Test-Path $ExecutablePath)) {
      $ExecutablePath = Join-Path -Path $ProjectRootDir -ChildPath "build/${BuildType}/tinyllama.exe"
    }
    if (-not (Test-Path $ExecutablePath)) {
      $ExecutablePath = Join-Path -Path $ProjectRootDir -ChildPath "build/Release/tinyllama.exe" # MSVC default
    }
     if (-not (Test-Path $ExecutablePath)) {
      $ExecutablePath = Join-Path -Path $ProjectRootDir -ChildPath "build/Debug/tinyllama.exe" # MSVC default
    }


    if (-not (Test-Path $ExecutablePath)) {
        Write-ErrorAndExit "Chat client executable not found at common paths (e.g., $ProjectRootDir\build\tinyllama.exe or $ProjectRootDir\build\$BuildType\tinyllama.exe). Please build the project first."
    }

    Log-Message "Starting chat client from $ExecutablePath..."
    Log-Message "Model directory/path: $ModelDir"
    Log-Message "Temperature: $Temperature"
    Log-Message "Top-K: $TopK"
    Log-Message "Top-P: $TopP"
    Log-Message "N GPU Layers: $NGpuLayers"
    Log-Message "Tokenizer Path: $TokenizerPath"
    Log-Message "Threads: $Threads"
    Log-Message "Use Mmap: $UseMmap"
    Log-Message "No Log: $NoLog"
    
    $ScriptArgs = @($ModelDir, $TokenizerPath, $Threads, $NGpuLayers, [string]$UseMmap)
    if ([string]::IsNullOrEmpty($Prompt)) {
        Log-Message "Mode: Interactive"
        # For interactive, tinyllama.exe might not need a prompt, or a default one. Assuming it handles interactive mode if no prompt.
        # The C++ tinyllama expects: model_path, prompt, steps, temperature, top_k, top_p
        # For interactive mode, let's pass a default prompt and let the C++ app handle it.
        $ScriptArgs += "What is the capital of France?" # Default placeholder prompt for the C++ app if it expects one
        $ScriptArgs += $Steps
    } else {
        Log-Message "Prompt: $Prompt"
        $ScriptArgs += $Prompt
        $ScriptArgs += $Steps # Use default steps if prompt is provided
    }
    $ScriptArgs += @($Temperature, $TopK, $TopP)
    
    if ($NoLog) {
        $ScriptArgs += "--no-log"
    }
    
    Log-Message "Executing: $ExecutablePath $ScriptArgs"
    & $ExecutablePath $ScriptArgs
    if ($LASTEXITCODE -ne 0) { Write-ErrorAndExit "Chat client execution failed."}
}

function Invoke-FormatCode {
    Log-Message "Formatting code with $FormatTool..."
    if (-not (Get-Command $FormatTool -ErrorAction SilentlyContinue)) {
        Write-ErrorAndExit "${FormatTool} could not be found. Please install it and ensure it's in your PATH."
    }
    $ClangFormatFile = Join-Path -Path $ProjectRootDir -ChildPath ".clang-format"
    if (-not (Test-Path $ClangFormatFile)) {
        Log-Message "Warning: .clang-format file not found at $ClangFormatFile. ${FormatTool} will use its default style."
    }

    Get-ChildItem -Path $ProjectRootDir -Recurse -Include "*.cpp", "*.h", "*.cu", "*.cuh" | ForEach-Object {
        Log-Message "Formatting $($_.FullName)"
        & $FormatTool -i $_.FullName
    }
    Log-Message "Formatting complete."
}

function Invoke-Docs {
    Log-Message "Generating Doxygen documentation..."
    $DoxygenExe = "doxygen.exe" # Assuming doxygen is in PATH
    if (-not (Get-Command $DoxygenExe -ErrorAction SilentlyContinue)) {
        Write-ErrorAndExit "$DoxygenExe could not be found. Please install Doxygen and ensure it's in your PATH."
    }
    $DoxyfilePath = Join-Path -Path $ProjectRootDir -ChildPath $DoxygenConfigFile
    if (-not (Test-Path $DoxyfilePath)) {
        Write-ErrorAndExit "Doxygen config file not found at $DoxyfilePath."
    }
    
    Set-Location $ProjectRootDir
    & $DoxygenExe $DoxyfilePath
    if ($LASTEXITCODE -ne 0) { Write-ErrorAndExit "Doxygen documentation generation failed."}
    Log-Message "Documentation generated in docs/html."
}

function Invoke-DocsServe {
    Log-Message "Serving documentation from docs/html on http://localhost:8000..."
    $DocsHtmlDir = Join-Path -Path $ProjectRootDir -ChildPath "docs/html"
    if (-not (Test-Path $DocsHtmlDir)) {
        Write-ErrorAndExit "Documentation directory $DocsHtmlDir not found. Generate docs first."
    }
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
         Write-ErrorAndExit "Python not found in PATH. Cannot start HTTP server."
    }
    
    Set-Location $DocsHtmlDir
    Log-Message "Starting Python HTTP server. Press Ctrl+C to stop."
    python -m http.server 8000
    Set-Location $ProjectRootDir
}

function Invoke-DocsClean {
    Log-Message "Cleaning documentation..."
    $DocsHtmlDir = Join-Path -Path $ProjectRootDir -ChildPath "docs/html"
    if (Test-Path $DocsHtmlDir) {
        Log-Message "Removing docs/html directory..."
        Remove-Item -Recurse -Force $DocsHtmlDir
    } else {
        Log-Message "docs/html directory not found. Nothing to clean."
    }
    Log-Message "Documentation clean complete."
}

function Invoke-Package {
    param(
        [string]$Version = $DefaultReleaseVersion,
        [string]$BuildType = "Release" # Packaging should almost always use Release
    )
    $Params = $script:Arguments | ConvertFrom-StringData -Delimiter ' '
    if ($Params.Version) { $Version = $Params.Version }
    if ($Params.BuildType) { $BuildType = $Params.BuildType }

    Log-Message "Packaging project version $Version (BuildType: $BuildType)..."

    # 1. Ensure project is built
    Log-Message "Ensuring project is built in $BuildType mode..."
    Invoke-Build -BuildType $BuildType -HasCuda $DefaultHasCuda # Or pass current $HasCuda if it's a param for package too

    $PackageName = "tinyllama_cpp_v${Version}_win_x64"
    $PackageDir = Join-Path -Path $ProjectRootDir -ChildPath "package"
    $TempPackagePath = Join-Path -Path $PackageDir -ChildPath $PackageName
    $ZipFilePath = Join-Path -Path $PackageDir -ChildPath "${PackageName}.zip"

    if (Test-Path $TempPackagePath) {
        Log-Message "Removing existing temporary package directory: $TempPackagePath"
        Remove-Item -Recurse -Force $TempPackagePath
    }
    if (Test-Path $ZipFilePath) {
        Log-Message "Removing existing package ZIP file: $ZipFilePath"
        Remove-Item -Force $ZipFilePath
    }
    New-Item -ItemType Directory -Path $TempPackagePath -Force | Out-Null

    Log-Message "Copying executables..."
    $SourceExeDir = Join-Path $ProjectRootDir "build/${BuildType}"
    if (-not (Test-Path $SourceExeDir)) { # Fallback for single-config generators
        $SourceExeDir = Join-Path $ProjectRootDir "build"
    }
    
    Copy-Item -Path (Join-Path $SourceExeDir "tinyllama.exe") -Destination $TempPackagePath -ErrorAction Stop
    Copy-Item -Path (Join-Path $SourceExeDir "tinyllama_server.exe") -Destination $TempPackagePath -ErrorAction Stop
    # Add Python bindings .pyd file if it exists and is desired in package
    $PydPath = Join-Path $SourceExeDir "tinyllama_bindings.pyd" 
     if (-not (Test-Path $PydPath)) { # Try another common location
        $PydPath = Join-Path $ProjectRootDir "build/$(($DefaultBuildType).ToLower())/tinyllama_bindings.pyd"
    }
    if (-not (Test-Path $PydPath)) { 
        $PydPath = Join-Path $ProjectRootDir "build/Release/tinyllama_bindings.pyd"
    }
     if (-not (Test-Path $PydPath)) { 
        $PydPath = Join-Path $ProjectRootDir "build/Debug/tinyllama_bindings.pyd"
    }

    if (Test-Path $PydPath) {
        Copy-Item -Path $PydPath -Destination $TempPackagePath
        Log-Message "Copied Python bindings: $PydPath"
    }


    Log-Message "Copying README and LICENSE..."
    Copy-Item -Path (Join-Path $ProjectRootDir "README.md") -Destination $TempPackagePath
    Copy-Item -Path (Join-Path $ProjectRootDir "LICENSE") -Destination $TempPackagePath -ErrorAction SilentlyContinue # License might not exist

    # Add other files as needed (e.g., example configs, model download scripts)
    # New-Item -ItemType Directory -Path (Join-Path $TempPackagePath "data_examples") | Out-Null
    # Copy-Item -Path (Join-Path $ProjectRootDir "config_example.json") -Destination (Join-Path $TempPackagePath "data_examples")

    Log-Message "Creating ZIP archive: $ZipFilePath"
    Compress-Archive -Path (Join-Path $TempPackagePath "*") -DestinationPath $ZipFilePath -Force
    
    Log-Message "Removing temporary package directory: $TempPackagePath"
    Remove-Item -Recurse -Force $TempPackagePath

    Log-Message "Package created successfully: $ZipFilePath"
}


# --- Main Script Logic ---
if (-not $Command -or $Command -eq "help") {
    Show-Usage
}

# A simple way to parse named arguments for PowerShell functions from a flat string array
# This is a basic approach. For more complex needs, consider advanced parameter handling per function.
# The $Arguments array will contain all args after the command. Example: .\manage.ps1 build -BuildType Debug -Cuda OFF
# Each function will need to parse its own relevant arguments from $script:Arguments

Log-Message "Executing command: $Command with arguments: $($Arguments -join ' ')"

switch ($Command) {
    "build"       { Invoke-Build }
    "clean"       { Invoke-Clean }
    "run-server"  { Invoke-RunServer }
    "run-chat"    { Invoke-RunChat }
    "format"      { Invoke-FormatCode }
    "docs"        { Invoke-Docs }
    "docs-serve"  { Invoke-DocsServe }
    "docs-clean"  { Invoke-DocsClean }
    "package"     { Invoke-Package }
    default {
        Write-ErrorAndExit "Unknown command: $Command"
        Show-Usage
    }
}

Log-Message "Script finished." 