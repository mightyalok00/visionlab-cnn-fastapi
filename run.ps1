param (
    [Parameter(Position = 0)]
    [ValidateSet("setup", "test", "train", "serve", "extract-samples", "help")]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host "VisionLab 2.0 CLI Helper" -ForegroundColor Cyan
    Write-Host "Usage: .\run.ps1 [command]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  setup           Create Python 3.11 virtualenv & install dependencies"
    Write-Host "  test            Run pytest suite with verbose output"
    Write-Host "  train           Run scratch CNN training benchmark across all variants"
    Write-Host "  serve           Start FastAPI server with hot-reload on port 8000"
    Write-Host "  extract-samples Extract sample images from Fashion-MNIST test set"
    Write-Host "  help            Show this help message"
}

$VenvPython = ".\.venv\Scripts\python.exe"
$VenvPytest = ".\.venv\Scripts\pytest.exe"
$VenvUvicorn = ".\.venv\Scripts\uvicorn.exe"

switch ($Command) {
    "setup" {
        Write-Host "Setting up Python 3.11 environment..." -ForegroundColor Green
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            uv venv --python 3.11
            uv pip install -r requirements.txt
        } else {
            py -3.11 -m venv .venv
            & $VenvPython -m pip install --upgrade pip
            & $VenvPython -m pip install -r requirements.txt
        }
        Write-Host "Setup complete!" -ForegroundColor Green
    }
    "test" {
        Write-Host "Running tests..." -ForegroundColor Green
        & $VenvPytest -v
    }
    "train" {
        Write-Host "Starting model training..." -ForegroundColor Green
        & $VenvPython -m scripts.train --epochs 8 --batch-size 128
    }
    "serve" {
        Write-Host "Launching FastAPI server at http://localhost:8000..." -ForegroundColor Green
        & $VenvUvicorn app.main:app --reload --port 8000
    }
    "extract-samples" {
        Write-Host "Extracting sample test images..." -ForegroundColor Green
        & $VenvPython -m scripts.extract_samples
    }
    default {
        Show-Help
    }
}
