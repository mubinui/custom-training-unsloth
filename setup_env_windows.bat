@echo off
REM Windows setup script for Unsloth custom training
REM This script handles the dependency conflicts and sets up the environment properly

echo ==================================================
echo Unsloth Custom Training - Windows Setup
echo ==================================================

REM Check if we're in a virtual environment
if "%VIRTUAL_ENV%"=="" (
    echo WARNING: No virtual environment detected
    echo It's recommended to use a virtual environment
    echo.
    echo To create one:
    echo python -m venv venv
    echo venv\Scripts\activate
    echo.
    pause
)

echo Step 1: Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo Step 2: Installing Unsloth and dependencies...

REM Try the main requirements first
echo Attempting to install from requirements.txt...
pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo WARNING: Main requirements failed, trying Windows-specific requirements...
    pip install -r requirements-windows.txt
    
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to install dependencies
        echo.
        echo Trying manual installation...
        
        REM Manual installation of core packages
        pip install git+https://github.com/unslothai/unsloth.git
        pip install transformers>=4.44.0
        pip install datasets>=2.21.0
        pip install accelerate>=0.33.0
        pip install "bitsandbytes>=0.45.5"
        pip install peft>=0.12.0
        pip install trl>=0.10.0
        pip install python-dotenv>=1.0.0
        pip install pandas numpy tqdm colorama
        pip install safetensors protobuf packaging typing-extensions
    )
)

echo.
echo Step 3: Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

if %ERRORLEVEL% neq 0 (
    echo ERROR: PyTorch verification failed
    pause
    exit /b 1
)

python -c "import unsloth; print('Unsloth imported successfully')"

if %ERRORLEVEL% neq 0 (
    echo ERROR: Unsloth verification failed
    pause
    exit /b 1
)

echo.
echo Step 4: Setting up environment configuration...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env"
        echo Created .env file from template
    ) else (
        echo WARNING: No .env.example found
    )
)

echo.
echo ==================================================
echo Setup completed successfully!
echo ==================================================
echo.
echo To start training:
echo   train_env.bat
echo.
echo For Bengali training:
echo   git checkout bengali-training
echo   python setup_bengali.py
echo   python main_bengali.py --sample --eval
echo.
echo For help:
echo   python main_env.py --help
echo.
pause
