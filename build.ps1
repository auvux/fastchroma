# Build the extension in place: python\fastchroma\_fastchroma.pyd
#
#   .\build.ps1                       # uses python from PATH
#   $env:PYTHON = 'C:\path\python.exe'; .\build.ps1
#
# Needs pybind11 and numpy importable from that Python, and MSVC Build Tools.
# CPU FFT is PFFFT (vendored). The GPU backend is CUDA when a toolkit is
# found (newest under "NVIDIA GPU Computing Toolkit\CUDA", or set
# $env:FASTCHROMA_CUDA_HOME); otherwise the build is CPU-only.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$pyInfo = & $python -c @'
import sys, sysconfig
import pybind11, numpy
print(pybind11.get_include())
print(numpy.get_include())
print(sysconfig.get_paths()['include'])
print(sys.prefix)
'@
if ($LASTEXITCODE) { throw "failed to query $python for pybind11/numpy includes" }
$pbInc, $npInc, $pyInc, $pyPrefix = $pyInfo
$pyLibs = Join-Path $pyPrefix 'libs'

# Import the MSVC x64 environment (cl on PATH, INCLUDE/LIB set).
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw 'vswhere.exe not found - install Visual Studio Build Tools' }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'no Visual Studio installation with C++ tools found' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
foreach ($line in (cmd /d /c "call `"$vcvars`" >nul 2>&1 && set")) {
    if ($line -match '^([^=]+)=(.*)$') { Set-Item "env:$($Matches[1])" $Matches[2] }
}

# CUDA toolkit (optional): newest installed, unless FASTCHROMA_CUDA_HOME is set.
$cudaRoot = $env:FASTCHROMA_CUDA_HOME
if (-not $cudaRoot) {
    $base = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $base) {
        $cudaRoot = Get-ChildItem $base -Directory |
            Sort-Object { [version]($_.Name -replace '^v', '') } |
            Select-Object -Last 1 -ExpandProperty FullName
    }
}
$nvcc = if ($cudaRoot) { Join-Path $cudaRoot 'bin\nvcc.exe' } else { $null }

New-Item -ItemType Directory -Force build | Out-Null
$out = 'python\fastchroma\_fastchroma.pyd'
# pybind11 auto-links pythonXY.lib via #pragma comment(lib), so only the
# library path is needed.

if ($nvcc -and (Test-Path $nvcc)) {
    Write-Host "CUDA toolkit: $cudaRoot"
    & $nvcc -O3 -std=c++17 -shared -arch=native `
        -DFASTCHROMA_CUDA=1 `
        -Xcompiler /O2,/fp:fast,/EHsc,/bigobj,/MD `
        -Isrc -Isrc\third_party -I"$pbInc" -I"$npInc" -I"$pyInc" `
        src\fastchroma.cpp src\third_party\pffft.c src\bindings.cpp src\cuda_backend.cu `
        -L"$pyLibs" -o $out
    if ($LASTEXITCODE) { throw 'nvcc build failed' }
} else {
    Write-Host 'no CUDA toolkit found - building CPU-only'
    cl /nologo /O2 /fp:fast /EHsc /bigobj /std:c++17 /MD /LD `
        /Isrc /Isrc\third_party /I"$pbInc" /I"$npInc" /I"$pyInc" `
        src\fastchroma.cpp src\third_party\pffft.c src\bindings.cpp `
        /Fo:build\ /Fe:$out /link /LIBPATH:"$pyLibs"
    if ($LASTEXITCODE) { throw 'cl build failed' }
}

Write-Host "built $out"
