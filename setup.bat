@echo off
echo ================================================
echo RTX 5080 Blackwell + Unsloth Setup (FINAL VERSION)
echo ================================================

:: Clean any existing environments
echo Cleaning previous environments...
if exist venv rmdir /s /q venv
if exist .venv rmdir /s /q .venv

:: Create fresh virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

:: Activate environment
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: CRITICAL: Install PyTorch with CUDA 12.8 for RTX 5080 Blackwell
echo Installing PyTorch with CUDA 12.8 for RTX 5080...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

if %errorlevel% neq 0 (
    echo CUDA 12.8 failed, trying nightly...
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    if %errorlevel% neq 0 (
        echo ERROR: PyTorch CUDA installation failed
        pause
        exit /b 1
    )
)

:: Install Triton 3.3.1+ for Blackwell
echo Installing Triton 3.3.1+ for RTX 5080...
pip install -U "triton>=3.3.1"

:: Install other dependencies
echo Installing dependencies...
pip install datasets transformers accelerate peft trl bitsandbytes

:: Install Unsloth from source (latest)
echo Installing Unsloth from source...
pip install git+https://github.com/unslothai/unsloth.git

:: Set RTX 5080 environment variables
echo Setting RTX 5080 environment variables...
set CUDA_VISIBLE_DEVICES=0
set TORCH_CUDA_ARCH_LIST=12.0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

:: CRITICAL: Test CUDA detection
echo ================================================
echo Testing RTX 5080 CUDA detection...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO GPU DETECTED!\"}')"

echo ================================================
echo Setup Complete! 
echo If you see "CUDA Available: True" above, run:
echo   activate.bat
echo   python main_bengali.py --sample --eval
echo ================================================
pause
