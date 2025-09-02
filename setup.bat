@echo off
REM Windows batch script to set up the training environment

echo Setting up Unsloth Custom Training Environment...

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8-3.11 from https://python.org
    pause
    exit /b 1
)

REM Check if CUDA is available
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: PyTorch not installed yet. Will install with CUDA support.
) else (
    python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support first
echo Installing PyTorch with CUDA support...
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

REM Install requirements (try exact versions first, fallback to flexible)
echo Installing requirements...
pip install -r requirements-exact.txt
if %ERRORLEVEL% neq 0 (
    echo Exact versions failed, trying flexible requirements...
    pip install -r requirements.txt
)

REM Test installation
echo Testing installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else None}')"

REM Test unsloth import
python -c "try: import unsloth; print('Unsloth imported successfully'); except ImportError: print('Unsloth import failed')"

echo.
echo Setup complete! 
echo.
echo To activate the environment in future sessions, run:
echo   venv\Scripts\activate.bat
echo.
echo To start training with sample data:
echo   python main.py --sample --eval
echo.
echo To start training with your data:
echo   python main.py --data your_data.json --eval
echo.

pause
