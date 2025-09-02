@echo off
REM Quick training script for testing

echo Starting Unsloth Training with Sample Data...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run training with sample data
python main.py --sample --eval

echo.
echo Training completed! Check the outputs/ folder for results.
echo Check training.log for detailed logs.

pause
