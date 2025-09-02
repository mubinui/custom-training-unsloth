@echo off
echo =======================================================
echo            Unsloth Custom Training Setup
echo            Environment Configuration System
echo =======================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10, 3.11, or 3.12 from https://python.org
    pause
    exit /b 1
)

echo ✓ Python is installed

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
echo This may take several minutes...

REM Try exact versions first for maximum compatibility
python -m pip install -r requirements-exact.txt
if errorlevel 1 (
    echo WARNING: Exact version installation failed, trying flexible versions...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo Please check your internet connection and try again
        pause
        exit /b 1
    )
)

echo ✓ Dependencies installed successfully

REM Copy environment template if .env doesn't exist
if not exist ".env" (
    echo Creating .env configuration file...
    copy ".env.example" ".env" >nul 2>&1
    if errorlevel 1 (
        echo Warning: Could not copy .env.example to .env
        echo You may need to create .env manually
    ) else (
        echo ✓ Created .env file from template
    )
)

REM Create necessary directories
if not exist "outputs" mkdir outputs
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "cache" mkdir cache

echo ✓ Created necessary directories

echo.
echo =======================================================
echo                    Setup Complete!
echo =======================================================
echo.
echo IMPORTANT: Edit the .env file to configure your training:
echo   - Set MODEL_NAME to your preferred model
echo   - Set DATA_PATH to your training data file
echo   - Adjust batch size if you have GPU memory issues
echo   - Set HF_TOKEN if you want to push to Hugging Face
echo.
echo Quick start commands:
echo   1. Edit .env file with your settings
echo   2. Run: python main_env.py --sample    (test with sample data)
echo   3. Run: python main_env.py             (train with your data)
echo.
echo For help: python main_env.py --help
echo For documentation: see README.md
echo.
pause
