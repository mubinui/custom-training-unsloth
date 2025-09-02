@echo off
REM UV Package Manager Setup for Bengali Training on Windows
REM This script handles UV-specific installation issues

echo ===============================================
echo UV Package Manager - Bengali Training Setup
echo ===============================================

REM Check if UV is installed
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: UV is not installed or not in PATH
    echo Please install UV first: https://github.com/astral-sh/uv
    pause
    exit /b 1
)

echo ✓ UV package manager detected

REM Set UV environment variables for better compatibility
set UV_SYSTEM_PYTHON=1
set UV_CACHE_DIR=%TEMP%\uv-cache

echo.
echo Step 1: Creating UV virtual environment...
uv venv .venv-uv

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create UV virtual environment
    pause
    exit /b 1
)

echo ✓ UV virtual environment created

echo.
echo Step 2: Activating virtual environment...
call .venv-uv\Scripts\activate

echo.
echo Step 3: Installing PyTorch with CUDA support using UV...
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install PyTorch with UV
    echo Trying alternative method...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)

echo.
echo Step 4: Installing Unsloth (Git dependency)...
REM UV handles Git dependencies differently, so we install it separately
uv pip install git+https://github.com/unslothai/unsloth.git

if %ERRORLEVEL% neq 0 (
    echo WARNING: UV failed to install Unsloth, trying with pip...
    pip install git+https://github.com/unslothai/unsloth.git
)

echo.
echo Step 5: Installing other dependencies...
uv pip install -r requirements-uv.txt

if %ERRORLEVEL% neq 0 (
    echo WARNING: Some packages failed with UV, installing individually...
    
    uv pip install transformers datasets accelerate
    uv pip install "bitsandbytes>=0.45.5"
    uv pip install peft trl python-dotenv
    uv pip install pandas numpy tqdm colorama rich
    uv pip install safetensors protobuf packaging typing-extensions
    
    REM Install Windows-specific packages
    uv pip install triton xformers
)

echo.
echo Step 6: Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

if %ERRORLEVEL% neq 0 (
    echo ERROR: PyTorch verification failed
    pause
    exit /b 1
)

python -c "import unsloth; print('✓ Unsloth imported successfully')"

if %ERRORLEVEL% neq 0 (
    echo WARNING: Unsloth import failed - this might be a GPU detection issue
    echo You can still proceed with training
)

echo.
echo Step 7: Setting up Bengali training environment...
python main_bengali.py --create-env

echo.
echo ===============================================
echo UV Setup Complete!
echo ===============================================
echo.
echo Virtual Environment: .venv-uv
echo Activate with: .venv-uv\Scripts\activate
echo.
echo To start Bengali training:
echo   python main_bengali.py --sample --eval
echo.
echo To deactivate when done:
echo   deactivate
echo.
pause
