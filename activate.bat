@echo off
echo Activating Bengali Training Environment...
echo.

if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat or clean_setup.bat first
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Virtual environment activated!
echo Python: %VIRTUAL_ENV%
echo.
echo Ready for Bengali training. Available commands:
echo   python main_bengali.py --sample --eval    (test with sample data)
echo   python main_bengali.py --eval             (full training)
echo   python main_bengali.py --max-examples 200 --eval  (limited training)
echo.
echo Type 'deactivate' to exit the virtual environment
echo.

cmd /k
