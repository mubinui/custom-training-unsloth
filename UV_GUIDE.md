# UV Package Manager - Bengali Training Guide

## UV-Specific Issues and Solutions

UV package manager can have unique challenges with ML packages. Here are the solutions:

## Quick Start with UV

### Option 1: Use the UV Setup Script (Recommended)
```cmd
setup-uv.bat
```

### Option 2: Manual UV Installation

#### Step 1: Create UV Virtual Environment
```cmd
uv venv .venv-uv
.venv-uv\Scripts\activate
```

#### Step 2: Install PyTorch First (Critical!)
```cmd
# UV may not handle PyTorch index URLs well, so be explicit
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Install Unsloth (Git Dependency)
```cmd
# UV handles Git deps differently - install separately
uv pip install git+https://github.com/unslothai/unsloth.git
```

#### Step 4: Install Other Dependencies
```cmd
uv pip install -r requirements-uv.txt
```

#### Step 5: Start Bengali Training
```cmd
python main_bengali.py --create-env
python main_bengali.py --sample --eval
```

## Common UV Issues and Fixes

### Issue 1: UV Git Dependency Problems
**Problem**: UV may fail to install Git dependencies like Unsloth

**Solution**: Install Git dependencies separately:
```cmd
uv pip install git+https://github.com/unslothai/unsloth.git
```

If that fails, fallback to pip for Git dependencies:
```cmd
pip install git+https://github.com/unslothai/unsloth.git
```

### Issue 2: UV Index URL Issues
**Problem**: UV may not properly handle PyTorch's CUDA index URLs

**Solution**: Be explicit with the index URL:
```cmd
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 3: UV Version Conflicts
**Problem**: UV's resolver might be stricter than pip

**Solution**: Install packages in specific order:
```cmd
# 1. PyTorch first
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Core ML packages
uv pip install transformers datasets accelerate

# 3. Quantization
uv pip install "bitsandbytes>=0.45.5"

# 4. Fine-tuning
uv pip install peft trl

# 5. Unsloth last
uv pip install git+https://github.com/unslothai/unsloth.git
```

### Issue 4: UV Cache Problems
**Problem**: UV cache might become corrupted

**Solution**: Clear UV cache:
```cmd
uv cache clean
```

Or set custom cache directory:
```cmd
set UV_CACHE_DIR=%TEMP%\uv-cache
```

### Issue 5: UV Virtual Environment Issues
**Problem**: UV virtual environments might not activate properly

**Solution**: Use UV's built-in activation:
```cmd
uv venv .venv-uv
.venv-uv\Scripts\activate
```

## UV Environment Variables

Set these for better UV compatibility:
```cmd
set UV_SYSTEM_PYTHON=1
set UV_CACHE_DIR=%TEMP%\uv-cache
set UV_RESOLUTION=lowest-direct
```

## Fallback to Pip

If UV continues to cause issues, you can fallback to pip within the UV environment:
```cmd
# Activate UV environment
.venv-uv\Scripts\activate

# Use pip for problematic packages
pip install git+https://github.com/unslothai/unsloth.git
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Continue with UV for other packages
uv pip install transformers datasets accelerate peft trl python-dotenv
```

## UV + Bengali Training Commands

Once setup is complete:

```cmd
# Activate UV environment
.venv-uv\Scripts\activate

# Create Bengali configuration
python main_bengali.py --create-env

# Test with sample data
python main_bengali.py --sample --eval

# Full Bengali training
python main_bengali.py --eval

# Deactivate when done
deactivate
```

## Troubleshooting UV

### Check UV Installation
```cmd
uv --version
where uv
```

### Check UV Environment
```cmd
uv pip list
uv pip show torch
uv pip show unsloth
```

### UV Debug Mode
```cmd
uv pip install --verbose git+https://github.com/unslothai/unsloth.git
```

## Mixed UV + Pip Approach (Recommended)

For maximum compatibility, use UV for standard packages and pip for Git dependencies:

```cmd
# Create UV environment
uv venv .venv-uv
.venv-uv\Scripts\activate

# Use pip for PyTorch and Unsloth (complex dependencies)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/unslothai/unsloth.git

# Use UV for standard packages (faster)
uv pip install transformers datasets accelerate peft trl
uv pip install python-dotenv pandas numpy tqdm colorama rich
uv pip install "bitsandbytes>=0.45.5"
```

This hybrid approach gives you UV's speed for standard packages while using pip's reliability for complex dependencies.

## UV Performance Benefits

Once working, UV provides:
- ⚡ **10-100x faster** package installation
- 🔒 **Better dependency resolution**
- 💾 **Efficient caching**
- 🛡️ **More reliable virtual environments**

Perfect for rapid iteration in ML development! 🚀
