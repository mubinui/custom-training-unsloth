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
python -c "try: import unsloth; print('SUCCESS: Unsloth imported successfully'); print('Unsloth version:', unsloth.__version__ if hasattr(unsloth, '__version__') else 'Version unknown'); except ImportError as e: print('FAILED: Unsloth import failed -', e); except Exception as e: print('ERROR:', e)"

echo.
echo Manual Unsloth installation complete!
pause
