#!/bin/bash

echo "======================================================="
echo "           Unsloth Custom Training Setup"
echo "           Environment Configuration System"
echo "======================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.10, 3.11, or 3.12"
    exit 1
fi

echo "✓ Python is installed"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
echo "This may take several minutes..."

# Try exact versions first for maximum compatibility
python -m pip install -r requirements-exact.txt
if [ $? -ne 0 ]; then
    echo "WARNING: Exact version installation failed, trying flexible versions..."
    python -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        echo "Please check your internet connection and try again"
        exit 1
    fi
fi

echo "✓ Dependencies installed successfully"

# Copy environment template if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env configuration file..."
    if [ -f ".env.example" ]; then
        cp ".env.example" ".env"
        echo "✓ Created .env file from template"
    else
        echo "Warning: .env.example not found"
        echo "You may need to create .env manually"
    fi
fi

# Create necessary directories
mkdir -p outputs data logs cache
echo "✓ Created necessary directories"

echo ""
echo "======================================================="
echo "                   Setup Complete!"
echo "======================================================="
echo ""
echo "IMPORTANT: Edit the .env file to configure your training:"
echo "  - Set MODEL_NAME to your preferred model"
echo "  - Set DATA_PATH to your training data file"
echo "  - Adjust batch size if you have GPU memory issues"
echo "  - Set HF_TOKEN if you want to push to Hugging Face"
echo ""
echo "Quick start commands:"
echo "  1. Edit .env file with your settings"
echo "  2. Run: python main_env.py --sample    (test with sample data)"
echo "  3. Run: python main_env.py             (train with your data)"
echo ""
echo "For help: python main_env.py --help"
echo "For documentation: see README.md"
echo ""
