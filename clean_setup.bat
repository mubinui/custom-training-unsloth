@echo off
echo ===============================================
echo Clean Setup - Remove UV and Use Standard Venv
echo ===============================================

echo Step 1: Removing any existing virtual environments...
if exist venv rmdir /s /q venv
if exist .venv rmdir /s /q .venv
if exist .venv-uv rmdir /s /q .venv-uv

echo Step 2: Clearing UV environment variables...
set UV_SYSTEM_PYTHON=
set UV_CACHE_DIR=
set VIRTUAL_ENV=

echo Step 3: Creating fresh Python virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python is installed and accessible
    pause
    exit /b 1
)

echo Step 4: Activating virtual environment...
call venv\Scripts\activate.bat

echo Step 5: Upgrading pip...
python -m pip install --upgrade pip

echo Step 6: Installing PyTorch for RTX 5080...
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

if %errorlevel% neq 0 (
    echo RTX 5080 nightly failed, trying CUDA 12.4 stable...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    
    if %errorlevel% neq 0 (
        echo CUDA 12.4 failed, using CUDA 12.1...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    )
)

echo Step 7: Installing ML libraries...
pip install transformers datasets accelerate peft trl python-dotenv
pip install pandas numpy tqdm colorama rich safetensors protobuf packaging
pip install "bitsandbytes>=0.45.5"

echo Step 8: Installing Unsloth...
pip install --no-deps git+https://github.com/unslothai/unsloth.git

echo Step 9: Testing installation...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo Step 10: Creating environment configuration...
python main_bengali.py --create-env

echo ===============================================
echo Clean Setup Complete!
echo ===============================================
echo.
echo Virtual environment: venv (standard Python venv)
echo No UV dependencies - pure pip installation
echo.
echo To activate in future:
echo   venv\Scripts\activate.bat
echo.
echo To start training:
echo   python main_bengali.py --sample --eval
echo.
pause
