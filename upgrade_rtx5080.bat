@echo off
echo =============================================
echo RTX 5080 PyTorch Upgrade Script
echo =============================================

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Current PyTorch version:
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'None')"

echo.
echo Upgrading to RTX 5080 compatible PyTorch (CUDA 12.4)...

echo Step 1: Uninstalling current PyTorch...
pip uninstall torch torchvision torchaudio -y

echo Step 2: Installing RTX 5080 compatible PyTorch (CUDA 12.4 nightly)...
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

if %errorlevel% neq 0 (
    echo RTX 5080 nightly failed, trying CUDA 12.4 stable...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    
    if %errorlevel% neq 0 (
        echo CUDA 12.4 failed, falling back to CUDA 12.1...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        echo WARNING: Using CUDA 12.1 - RTX 5080 may have compatibility warnings
    )
)

echo.
echo Updated PyTorch version:
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'None'); print('GPU capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'No GPU')"

echo.
echo RTX 5080 PyTorch upgrade complete!
pause
