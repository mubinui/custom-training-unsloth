@echo off
echo ================================================
echo Bengali Training Setup for Windows RTX 5080
echo ================================================

echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch for RTX 5080 (CUDA 12.4)...
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

echo Installing core ML libraries...
pip install transformers datasets accelerate peft trl python-dotenv
pip install pandas numpy tqdm colorama rich safetensors protobuf packaging
pip install "bitsandbytes>=0.45.5"

echo Installing Windows-compatible Unsloth...
pip install --no-deps git+https://github.com/unslothai/unsloth.git

echo Testing PyTorch and CUDA...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo Creating environment configuration...
python main_bengali.py --create-env

echo ================================================
echo Setup Complete!
echo ================================================
echo.
echo To start training:
echo   venv\Scripts\activate.bat
echo   python main_bengali.py --sample --eval
echo.
echo NOTE: RTX 5080 warnings are normal with CUDA 12.1
echo For full RTX 5080 support, PyTorch nightly builds are recommended
echo.
pause
echo.
echo To start training with your data:
echo   python main.py --data your_data.json --eval
echo.

pause
