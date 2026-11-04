<#
.SYNOPSIS
    Create the project virtual environment (.venv) on Windows, with CUDA PyTorch.

.DESCRIPTION
    Python 3.13.9 - the Jetson Orin's interpreter - is installed by uv, so the
    venv depends on neither Anaconda nor a system Python.  PyTorch comes from
    the PyTorch CUDA index (the PyPI wheel is CPU-only on Windows); everything
    else comes from requirements.txt.  Safe to re-run: only missing pieces are
    installed.

.EXAMPLE
    .\scripts\setup_env.ps1

.EXAMPLE
    .\scripts\setup_env.ps1 -Cuda cu130 -Recreate
#>
param(
    [string]$Python = "3.13.9",   # same interpreter as the Jetson Orin run
    [string]$Torch = "2.14.1",
    [string]$Cuda = "cu126",      # CUDA 12.6 = JetPack 6 on the Jetson; cu130 / cu132 also work (with -Recreate)
    [switch]$Recreate             # delete .venv and build it from scratch
)
# Failures are detected through $LASTEXITCODE, not $ErrorActionPreference = "Stop":
# Windows PowerShell 5.1 turns uv's progress messages on stderr into errors.
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "Installing uv (Python + package manager) with winget..."
        winget install --id astral-sh.uv --exact --scope user --accept-package-agreements --accept-source-agreements
        $env:Path = "$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:Path"
    }

    if ($Recreate -and (Test-Path .venv)) { Remove-Item -Recurse -Force .venv -ErrorAction Stop }
    if (-not (Test-Path .venv\Scripts\python.exe)) {
        # A .venv without an interpreter is the remains of an interrupted run.
        if (Test-Path .venv) { Remove-Item -Recurse -Force .venv -ErrorAction Stop }
        # A standalone CPython managed by uv - never Anaconda's or the Store's.
        # On some machines `uv python install` reports an error creating its
        # "3.13" alias link although the interpreter installed fine, so the exit
        # code is not trusted; the exact executable is looked up instead.
        uv python install $Python
        $interp = uv python find $Python --system --python-preference only-managed
        if ($LASTEXITCODE -or -not $interp) { throw "installing Python $Python failed" }
        uv venv .venv --python $interp
        if ($LASTEXITCODE) { throw "creating .venv failed" }
    }
    $py = ".venv\Scripts\python.exe"

    Write-Host "`nInstalling PyTorch $Torch+$Cuda ..."
    uv pip install --python $py "torch==$Torch" --index-url "https://download.pytorch.org/whl/$Cuda"
    if ($LASTEXITCODE) { throw "PyTorch install failed" }

    Write-Host "`nInstalling requirements.txt ..."
    uv pip install --python $py -r requirements.txt
    if ($LASTEXITCODE) { throw "requirements install failed" }

    Write-Host ""
    & $py scripts\check_env.py --require-gpu
    if ($LASTEXITCODE) { throw "environment check failed" }
    Write-Host "`nReady. Activate with:  .venv\Scripts\Activate.ps1"
}
finally {
    Pop-Location
}
