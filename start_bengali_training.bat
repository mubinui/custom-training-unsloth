@echo off
REM Quick Bengali Training Launcher for Windows
REM This script bypasses common setup issues and starts training directly

echo ============================================
echo Bengali Training Quick Start for Windows
echo ============================================

REM Set GPU environment variables to help with detection
set CUDA_VISIBLE_DEVICES=0
set TORCH_CUDA_ARCH_LIST=7.0+PTX

echo Step 1: Checking CUDA availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU detected\"}')"

if %ERRORLEVEL% neq 0 (
    echo ERROR: PyTorch check failed
    echo Please ensure PyTorch is installed with CUDA support
    pause
    exit /b 1
)

echo.
echo Step 2: Creating environment configuration...
python main_bengali.py --create-env

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create environment configuration
    pause
    exit /b 1
)

echo.
echo Step 3: Starting Bengali training with sample data...
echo This will test the system with 50 Bengali examples

python main_bengali.py --sample --eval

if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: Sample training failed
    echo This might be due to GPU detection issues
    echo.
    echo Troubleshooting steps:
    echo 1. Ensure you have an NVIDIA GPU
    echo 2. Update NVIDIA drivers
    echo 3. Reinstall PyTorch with CUDA:
    echo    pip uninstall torch torchvision torchaudio
    echo    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo SUCCESS! Sample training completed
echo ============================================
echo.
echo Next steps:
echo 1. For full training: python main_bengali.py --eval
echo 2. For limited training: python main_bengali.py --max-examples 200 --eval
echo 3. To upload to hub: python main_bengali.py --eval --push
echo.
echo Your model will be saved in: .\outputs\
echo Training summary: .\outputs\bengali_training_summary.json
echo.
pause
