@echo off
REM Quick training script with environment configuration
REM Usage: train_env.bat [--sample] [--eval]

echo Starting Unsloth training with environment configuration...

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found. Run setup_env.bat first.
    pause
    exit /b 1
)

REM Check if .env exists
if not exist ".env" (
    echo ERROR: .env file not found. 
    echo Please run setup_env.bat or copy .env.example to .env and edit it.
    pause
    exit /b 1
)

REM Run training with environment configuration
python main_env.py %*

if errorlevel 1 (
    echo ERROR: Training failed
    pause
    exit /b 1
)

echo Training completed successfully!
pause
