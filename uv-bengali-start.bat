@echo off
REM UV Quick Start for Bengali Training
REM Handles UV-specific issues and provides fallbacks

echo ========================================
echo UV Bengali Training Quick Start
echo ========================================

REM Check if UV environment exists
if not exist ".venv-uv" (
    echo Creating UV virtual environment...
    uv venv .venv-uv
)

echo Activating UV environment...
call .venv-uv\Scripts\activate

REM Set UV environment variables
set UV_SYSTEM_PYTHON=1
set UV_CACHE_DIR=%TEMP%\uv-cache

echo.
echo Checking PyTorch installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>nul

if %ERRORLEVEL% neq 0 (
    echo Installing PyTorch with UV...
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)

echo.
echo Checking Unsloth installation...
python -c "import unsloth; print('Unsloth OK')" 2>nul

if %ERRORLEVEL% neq 0 (
    echo Installing Unsloth...
    echo Trying UV first...
    uv pip install git+https://github.com/unslothai/unsloth.git
    
    if %ERRORLEVEL% neq 0 (
        echo UV failed, using pip fallback...
        pip install git+https://github.com/unslothai/unsloth.git
    )
)

echo.
echo Installing remaining dependencies...
uv pip install transformers datasets accelerate peft trl python-dotenv
uv pip install "bitsandbytes>=0.45.5"
uv pip install pandas numpy tqdm colorama rich safetensors

echo.
echo Setting up Bengali training...
python main_bengali.py --create-env

echo.
echo Starting Bengali sample training...
python main_bengali.py --sample --eval

if %ERRORLEVEL% neq 0 (
    echo.
    echo ==========================================
    echo Sample training failed!
    echo ==========================================
    echo This might be due to:
    echo 1. GPU detection issues
    echo 2. Missing dependencies
    echo 3. UV package conflicts
    echo.
    echo Try the full setup: setup-uv.bat
    echo Or check UV_GUIDE.md for troubleshooting
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo SUCCESS! UV Bengali Training is working
echo ==========================================
echo.
echo Environment: .venv-uv
echo.
echo Next commands:
echo   python main_bengali.py --eval          # Full training
echo   python main_bengali.py --max-examples 200 --eval  # Limited training
echo   python main_bengali.py --eval --push   # Train and upload
echo.
echo To deactivate: deactivate
echo.
pause
