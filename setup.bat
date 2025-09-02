@echo off
echo Setting up Unsloth Custom Training Environment...

REM Check CUDA availability
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" 2>nul
if %errorlevel% neq 0 (
    echo Python or PyTorch not found, proceeding with installation...
)

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

echo Installing PyTorch with RTX 5080 support (CUDA 12.4)...
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

if %errorlevel% neq 0 (
    echo RTX 5080 nightly failed, trying CUDA 12.4 stable...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    
    if %errorlevel% neq 0 (
        echo CUDA 12.4 failed, falling back to CUDA 12.1...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    )
)

echo Installing requirements...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo Requirements installation failed, trying individual packages...
    pip install transformers datasets accelerate peft trl python-dotenv
    pip install pandas numpy tqdm colorama rich safetensors protobuf packaging
    pip install "bitsandbytes>=0.45.5"
    pip install git+https://github.com/unslothai/unsloth.git
)

echo Testing installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'None')"

python -c "try: import unsloth; print('Unsloth imported successfully'); except ImportError as e: print('Unsloth import failed:', e)"

echo.
echo Setup complete! 
echo.
echo To activate the environment in future sessions, run:
echo   venv\Scripts\activate.bat
echo.
echo To start training with sample data:
echo   python main_bengali.py --sample --eval
echo.
echo To start training with your data:
echo   python main_bengali.py --eval
echo.
pause
echo.
echo To start training with your data:
echo   python main.py --data your_data.json --eval
echo.

pause
