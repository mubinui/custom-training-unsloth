# Windows Dependency Troubleshooting Guide

## Common Dependency Conflicts on Windows

### Issue 1: Bitsandbytes Version Conflict

**Error:**
```
No solution found when resolving dependencies:
unsloth[colab-new]==2025.8.10 depends on bitsandbytes>=0.45.5
but you require bitsandbytes>=0.43.0,<0.44.0
```

**Solutions:**

#### Option 1: Use Windows-Specific Requirements
```cmd
pip install -r requirements-windows.txt
```

#### Option 2: Manual Installation
```cmd
# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth
pip install git+https://github.com/unslothai/unsloth.git

# Install compatible bitsandbytes
pip install "bitsandbytes>=0.45.5"

# Install other dependencies
pip install transformers datasets accelerate peft trl python-dotenv
```

#### Option 3: Use UV Package Manager
```cmd
# If using UV, try without strict version constraints
uv pip install git+https://github.com/unslothai/unsloth.git
uv pip install transformers datasets accelerate peft trl python-dotenv
uv pip install "bitsandbytes>=0.45.5"
```

### Issue 2: CUDA Compatibility

**Symptoms:**
- CUDA not detected
- GPU not available for training

**Solutions:**

1. **Verify CUDA Installation:**
```cmd
nvidia-smi
nvcc --version
```

2. **Install Correct PyTorch Version:**
```cmd
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Test CUDA Availability:**
```cmd
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### Issue 3: Memory Issues During Installation

**Symptoms:**
- Installation fails with memory errors
- Large dependency downloads fail

**Solutions:**

1. **Use Pip Cache:**
```cmd
pip install --cache-dir C:\pip-cache -r requirements.txt
```

2. **Install in Smaller Batches:**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets
pip install accelerate bitsandbytes
pip install git+https://github.com/unslothai/unsloth.git
```

3. **Increase Virtual Memory:**
- Go to System Properties > Advanced > Performance Settings > Advanced > Virtual Memory
- Set custom size (e.g., 8000-16000 MB)

### Issue 4: Git Authentication

**Error:**
```
git+https://github.com/unslothai/unsloth.git requires authentication
```

**Solutions:**

1. **Use GitHub Token:**
```cmd
pip install git+https://<your-token>@github.com/unslothai/unsloth.git
```

2. **Use SSH (if configured):**
```cmd
pip install git+ssh://git@github.com/unslothai/unsloth.git
```

3. **Download and Install Locally:**
```cmd
git clone https://github.com/unslothai/unsloth.git
cd unsloth
pip install -e .
```

## Environment-Specific Solutions

### For Conda Environments
```cmd
conda create -n unsloth python=3.11
conda activate unsloth
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install git+https://github.com/unslothai/unsloth.git
pip install transformers datasets accelerate peft trl python-dotenv
```

### For UV Package Manager
```cmd
uv venv unsloth
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install git+https://github.com/unslothai/unsloth.git
uv pip install transformers datasets accelerate peft trl python-dotenv bitsandbytes
```

### For Poetry
```cmd
poetry install --no-dev
poetry add torch torchvision torchaudio --source pytorch-cu121
poetry add git+https://github.com/unslothai/unsloth.git
```

## Quick Fix Commands

### Complete Fresh Install
```cmd
# Remove existing installation
pip uninstall unsloth transformers datasets accelerate bitsandbytes peft trl -y

# Fresh install with correct versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/unslothai/unsloth.git
pip install transformers datasets accelerate "bitsandbytes>=0.45.5" peft trl python-dotenv
```

### Verify Installation
```cmd
python -c "import unsloth; import torch; print('All imports successful')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Bengali Training Specific

### Quick Bengali Setup on Windows
```cmd
git checkout bengali-training
python setup_bengali.py
python main_bengali.py --sample --eval
```

### If Bengali Setup Fails
```cmd
# Manual Bengali dependencies
pip install datasets>=2.21.0
python -c "from datasets import load_dataset; ds = load_dataset('spedrox-sac/bengali_chat_conv', split='train[:5]'); print('Dataset accessible')"
```

## Getting Help

1. **Check Python Version:**
```cmd
python --version
# Should be 3.10, 3.11, or 3.12
```

2. **Check GPU:**
```cmd
nvidia-smi
```

3. **Check Package Versions:**
```cmd
pip list | findstr "torch transformers unsloth"
```

4. **Run Setup Script:**
```cmd
setup_env_windows.bat
```

If all else fails, use the manual installation approach and create a minimal environment with just the essential packages.
