@echo off
echo ================================
echo Manual Unsloth Installation
echo ================================

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Unsloth from GitHub...
pip install git+https://github.com/unslothai/unsloth.git

echo.
echo Testing Unsloth installation...
python -c "import unsloth; print('SUCCESS: Unsloth imported successfully')"

echo.
echo Manual Unsloth installation complete!
pause
