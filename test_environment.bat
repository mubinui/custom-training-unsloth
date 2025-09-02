@echo off
echo =====================================
echo Testing Bengali Training Environment
echo =====================================

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Testing Python and PyTorch...
python -c "import sys; print('Python version:', sys.version)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'None')"

echo.
echo Testing GPU...
python -c "import torch; print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
python -c "import torch; print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

echo.
echo Testing core libraries...
python -c "import transformers; print('Transformers version:', transformers.__version__)"
python -c "import datasets; print('Datasets version:', datasets.__version__)"
python -c "import accelerate; print('Accelerate version:', accelerate.__version__)"
python -c "import unsloth; print('Unsloth version:', unsloth.__version__ if hasattr(unsloth, '__version__') else 'Available')"

echo.
echo Testing Bengali training modules...
python -c "from env_config import load_config; print('env_config module: OK')"
python -c "from bengali_data_processor import BengaliDataProcessor; print('bengali_data_processor module: OK')"
python -c "from trainer import UnslothTrainer; print('trainer module: OK')"

echo.
echo All tests completed!
echo.
echo Ready to start Bengali training with:
echo   python main_bengali.py --sample --eval
echo.
pause
