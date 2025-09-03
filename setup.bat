@echo off
echo ================================================
echo Unsloth Bengali Training - Windows RTX 5080 Setup
echo Using REAL Windows Triton (not mocked)
echo ================================================

:: Clean any existing environments
echo Cleaning previous environments...
if exist venv rmdir /s /q venv
if exist .venv rmdir /s /q .venv

:: Clear environment variables
set UV_SYSTEM_PYTHON=
set UV_CACHE_DIR=
set VIRTUAL_ENV=

:: Create fresh virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python 3.8+ is installed and in PATH
    pause
    exit /b 1
)

:: Activate environment
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: CRITICAL: Uninstall any existing triton first
echo Removing any existing triton installations...
pip uninstall -y triton

:: Install PyTorch stable with CUDA 12.4 for RTX 5080
echo Installing PyTorch stable for RTX 5080 (CUDA 12.4)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

if %errorlevel% neq 0 (
    echo CUDA 12.4 failed, trying nightly...
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
    if %errorlevel% neq 0 (
        echo ERROR: PyTorch installation failed
        pause
        exit /b 1
    )
)

:: CRITICAL: Install Windows Triton fork (REAL triton for Windows RTX 5080)
echo Installing Windows Triton fork for RTX 5080...
pip install -U "triton-windows<3.5"

if %errorlevel% neq 0 (
    echo ERROR: Windows Triton installation failed
    echo This is required for RTX 5080 support
    pause
    exit /b 1
)

:: Install core dependencies
echo Installing core dependencies...
pip install -r requirements.txt

:: Install Unsloth (should work with real triton now)
echo Installing Unsloth...
pip install unsloth

if %errorlevel% neq 0 (
    echo Trying Unsloth from source...
    pip install git+https://github.com/unslothai/unsloth.git
)

:: Verify installation
echo Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"No GPU\"}')"
python -c "import triton; print(f'Triton: {triton.__version__}')"

echo.
echo ================================================
echo Setup complete with REAL Windows Triton!
echo.
echo To activate environment: call activate.bat
echo To start training: python main_bengali.py --sample --eval
echo ================================================
pause
