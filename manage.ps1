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
    [ValidateSet("build", "clean", "run-server", "run-chat", "run-prompt", "format", "docs", "docs-serve", "docs-clean", "package", "help")]
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
    Write-Host "                 -TokenizerPath <path>      (default: auto-detect from ModelDir or '${DefaultTokenizerPath}')"
    Write-Host "                 -Threads <num>             (default: ${DefaultThreads})"
    Write-Host "                 -SystemPrompt <text>      (Optional) System prompt to guide the model."
    Write-Host "                 -Temperature <float>        (default: ${DefaultTemperature})"
    Write-Host "                 -TopK <int>               (default: ${DefaultTopK})"
    Write-Host "                 -TopP <float>             (default: ${DefaultTopP})"
    Write-Host "                 -Prompt <text>             (default: ${DefaultPrompt})"
    Write-Host "                 -NGpuLayers <int>         (default: ${DefaultNGpuLayers}, -1 for all on GPU)"
    Write-Host "                 -Mmap <true|false>        (default: ${DefaultUseMmap})"
    Write-Host ""
    Write-Host "  run-prompt   Run the C++ model with a single prompt and exit."
    Write-Host "               Options:"
    Write-Host "                 -ModelDir <path>          (default: ${DefaultModelDir})"
    Write-Host "                 -TokenizerPath <path>      (default: auto-detect from ModelDir or '${DefaultTokenizerPath}')"
    Write-Host "                 -SystemPrompt <text>      (Optional) System prompt to guide the model."
    Write-Host "                 -Prompt <text>             (default: ${DefaultPrompt})"
    Write-Host "                 -Steps <num>               (default: ${DefaultSteps})"
    Write-Host "                 -Temperature <float>       (default: ${DefaultTemperature})"
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
        [string]$SystemPrompt = "", # New parameter
        [switch]$NoLog
    )
    $LocalArgs = $PSBoundParameters # Capture explicitly passed parameters

    # Parse remaining -Arguments as a hashtable for options not explicitly defined as params
    # This is a bit simplistic; a more robust parser would be better for general -key value pairs.
    $RemainingArgs = @{}
    for ($i = 0; $i -lt $Arguments.Length; $i += 2) {
        if ($Arguments[$i] -match '^-([\w-]+)$') {
            $Key = $Matches[1]
            if (($i + 1) -lt $Arguments.Length) {
                $RemainingArgs[$Key] = $Arguments[$i+1]
            } else {
                $RemainingArgs[$Key] = $true # For switch-like behavior if value is missing
            }
        }
    }

    # Override with $RemainingArgs if they exist
    if ($RemainingArgs.ContainsKey('ModelDir')) { $ModelDir = $RemainingArgs['ModelDir'] }
    if ($RemainingArgs.ContainsKey('Temperature')) { $Temperature = $RemainingArgs['Temperature'] }
    if ($RemainingArgs.ContainsKey('TopK')) { $TopK = $RemainingArgs['TopK'] }
    if ($RemainingArgs.ContainsKey('TopP')) { $TopP = $RemainingArgs['TopP'] }
    if ($RemainingArgs.ContainsKey('Prompt')) { $Prompt = $RemainingArgs['Prompt'] }
    if ($RemainingArgs.ContainsKey('BuildType')) { $BuildType = $RemainingArgs['BuildType'] }
    if ($RemainingArgs.ContainsKey('Steps')) { $Steps = $RemainingArgs['Steps'] }
    if ($RemainingArgs.ContainsKey('NGpuLayers')) { $NGpuLayers = [int]$RemainingArgs['NGpuLayers'] }
    if ($RemainingArgs.ContainsKey('TokenizerPath')) { $TokenizerPath = $RemainingArgs['TokenizerPath'] }
    if ($RemainingArgs.ContainsKey('Threads')) { $Threads = [int]$RemainingArgs['Threads'] }
    if ($RemainingArgs.ContainsKey('UseMmap')) { $UseMmap = [bool]::Parse($RemainingArgs['UseMmap']) }
    if ($RemainingArgs.ContainsKey('SystemPrompt')) { $SystemPrompt = $RemainingArgs['SystemPrompt'] }
    if ($RemainingArgs.ContainsKey('NoLog')) { $NoLog = $true }
    
    $ExecutablePath = Find-Executable -Name "main" -BuildType $BuildType
    if (-not $ExecutablePath) {
        Write-ErrorAndExit "Main executable (main.exe or main) not found. Please build the project first."
    }

    Log-Message "Starting chat client from $ExecutablePath..."

    $ActualModelPath = Resolve-ModelPath -ModelDir $ModelDir
    $ActualTokenizerPath = Resolve-TokenizerPath -ModelDir $ModelDir -TokenizerPath $TokenizerPath -ModelPathToCheck $ActualModelPath

    $ExecArgs = @(
        $ActualModelPath,
        $ActualTokenizerPath,
        [string]$Threads,
        "chat" # Mode
    )

    if (-not [string]::IsNullOrEmpty($SystemPrompt)) {
        $ExecArgs += "--system-prompt", $SystemPrompt
    }
    if (-not [string]::IsNullOrEmpty($Prompt)) {
        $ExecArgs += $Prompt # This is initial_user_prompt
    }
    
    $ExecArgs += "--max-tokens", $Steps
    $ExecArgs += "--n-gpu-layers", ([string]$NGpuLayers)
    $ExecArgs += "--use-mmap", ([string]$UseMmap).ToLower()
    $ExecArgs += "--temperature", $Temperature
    # TopK and TopP are not directly passed to main.cpp in this version

    Log-Message "Running chat with: $ExecutablePath $($ExecArgs -join ' ' )"
    & $ExecutablePath $ExecArgs
    if ($LASTEXITCODE -ne 0) { Write-ErrorAndExit "Chat client execution failed."}
}

function Invoke-RunPrompt {
    param (
        [string]$ModelDir = $DefaultModelDir,
        [string]$TokenizerPath = $DefaultTokenizerPath,
        [string]$Prompt = "Hello, world!",
        [string]$Steps = "128",
        [string]$BuildType = $DefaultBuildType, 
        [string]$Temperature = $DefaultTemperature,
        [int]$NGpuLayers = $DefaultNGpuLayers,
        [int]$Threads = $DefaultThreads,
        [bool]$UseMmap = $DefaultUseMmap,
        [string]$SystemPrompt = "" # New parameter
    )
    $LocalArgs = $PSBoundParameters
    $RemainingArgs = @{}
    for ($i = 0; $i -lt $Arguments.Length; $i += 2) {
        if ($Arguments[$i] -match '^-([\w-]+)$') {
            $Key = $Matches[1]
            if (($i + 1) -lt $Arguments.Length) { $RemainingArgs[$Key] = $Arguments[$i+1] }
            else { $RemainingArgs[$Key] = $true }
        }
    }

    if ($RemainingArgs.ContainsKey('ModelDir')) { $ModelDir = $RemainingArgs['ModelDir'] }
    if ($RemainingArgs.ContainsKey('TokenizerPath')) { $TokenizerPath = $RemainingArgs['TokenizerPath'] }
    if ($RemainingArgs.ContainsKey('Prompt')) { $Prompt = $RemainingArgs['Prompt'] }
    if ($RemainingArgs.ContainsKey('Steps')) { $Steps = $RemainingArgs['Steps'] }
    if ($RemainingArgs.ContainsKey('BuildType')) { $BuildType = $RemainingArgs['BuildType'] }
    if ($RemainingArgs.ContainsKey('Temperature')) { $Temperature = $RemainingArgs['Temperature'] }
    if ($RemainingArgs.ContainsKey('NGpuLayers')) { $NGpuLayers = [int]$RemainingArgs['NGpuLayers'] }
    if ($RemainingArgs.ContainsKey('Threads')) { $Threads = [int]$RemainingArgs['Threads'] }
    if ($RemainingArgs.ContainsKey('UseMmap')) { $UseMmap = [bool]::Parse($RemainingArgs['UseMmap']) }
    if ($RemainingArgs.ContainsKey('SystemPrompt')) { $SystemPrompt = $RemainingArgs['SystemPrompt'] }

    if ([string]::IsNullOrEmpty($Prompt)) {
        Write-ErrorAndExit "Prompt text cannot be empty for run-prompt. Use -Prompt <text>."
    }
    
    $ExecutablePath = Find-Executable -Name "main" -BuildType $BuildType
    if (-not $ExecutablePath) {
        Write-ErrorAndExit "Main executable (main.exe or main) not found. Please build the project first."
    }

    $ActualModelPath = Resolve-ModelPath -ModelDir $ModelDir
    $ActualTokenizerPath = Resolve-TokenizerPath -ModelDir $ModelDir -TokenizerPath $TokenizerPath -ModelPathToCheck $ActualModelPath

    $ExecArgs = @(
        $ActualModelPath,
        $ActualTokenizerPath,
        [string]$Threads,
        "prompt" # Mode
    )
    if (-not [string]::IsNullOrEmpty($SystemPrompt)) {
        $ExecArgs += "--system-prompt", $SystemPrompt
    }
    $ExecArgs += $Prompt # This is initial_user_prompt or the main prompt
    $ExecArgs += "--max-tokens", $Steps
    $ExecArgs += "--n-gpu-layers", ([string]$NGpuLayers)
    $ExecArgs += "--use-mmap", ([string]$UseMmap).ToLower()
    $ExecArgs += "--temperature", $Temperature

    Log-Message "Running prompt with: $ExecutablePath $($ExecArgs -join ' ' )"
    & $ExecutablePath $ExecArgs
    if ($LASTEXITCODE -ne 0) { Write-ErrorAndExit "Prompt execution failed."}
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
    "run-prompt"  { Invoke-RunPrompt }
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