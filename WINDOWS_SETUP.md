# Windows Installation Instructions for Bengali Training

## Prerequisites
1. **Python 3.10, 3.11, or 3.12** (NOT 3.13 - not supported by Unsloth)
2. **NVIDIA GPU** with CUDA Capability 7.0+ (RTX 20 series or newer)
3. **NVIDIA Drivers** (latest version)
4. **Visual Studio C++** with Windows SDK
5. **CUDA Toolkit 12.1** (recommended)

## Installation Steps

### Step 1: Install PyTorch with CUDA Support FIRST
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Verify CUDA Installation
```cmd
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

### Step 3: Install Unsloth and Dependencies
```cmd
pip install -r requirements.txt
```

If you encounter the bitsandbytes version conflict, run:
```cmd
pip install "bitsandbytes>=0.45.5" --force-reinstall
```

### Step 4: Set Windows Environment Variables
```cmd
set CUDA_VISIBLE_DEVICES=0
set TORCH_CUDA_ARCH_LIST=7.0+PTX
```

## Quick Start Bengali Training

### Step 1: Create Environment Configuration
```cmd
python main_bengali.py --create-env
```

### Step 2: Test with Sample Data
```cmd
python main_bengali.py --sample --eval
```

### Step 3: Full Training
```cmd
python main_bengali.py --eval
```

## Alternative: Use the Quick Start Script
```cmd
start_bengali_training.bat
```

## Troubleshooting

### If GPU is Not Detected:
1. Update NVIDIA drivers
2. Reinstall PyTorch with correct CUDA version
3. Check Windows environment variables

### If Dependencies Fail:
1. Install packages individually:
   ```cmd
   pip install transformers datasets accelerate peft trl
   pip install "bitsandbytes>=0.45.5"
   pip install python-dotenv
   ```

### For Windows-Specific Issues:
- Ensure Visual Studio C++ is installed with Windows SDK
- Use Command Prompt (not PowerShell) for pip installations
- Run as Administrator if permission errors occur

## Success Indicators
- ✅ CUDA Available: True
- ✅ GPU detected and named
- ✅ Sample training completes successfully
- ✅ Bengali model generates responses

Your Bengali training system is now ready for Windows deployment! 🚀
