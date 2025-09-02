#!/bin/bash
# Linux/MacOS setup script (for reference, though you'll use Windows)

echo "Setting up Unsloth Custom Training Environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8-3.11"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install requirements (try exact versions first, fallback to flexible)
echo "Installing requirements..."
if pip install -r requirements-exact.txt; then
    echo "Exact versions installed successfully"
else
    echo "Exact versions failed, trying flexible requirements..."
    pip install -r requirements.txt
fi

# Test installation
echo "Testing installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else None}')"

# Test unsloth import
python -c "try: import unsloth; print('Unsloth imported successfully'); except ImportError: print('Unsloth import failed - this is expected on non-CUDA systems')"

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment in future sessions, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start training with sample data:"
echo "  python main.py --sample --eval"
echo ""
echo "To start training with your data:"
echo "  python main.py --data your_data.json --eval"
echo ""
