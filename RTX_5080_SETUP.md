# RTX 5080 Setup Guide for Bengali Training

## 🚨 **RTX 5080 CUDA Compatibility Issue**

The RTX 5080 has CUDA capability `sm_120` which requires **PyTorch 2.5+** and **CUDA 12.4+**.

## 📋 **Quick Fix Commands**

### **For UV Package Manager (Recommended):**

```bash
# Step 1: Install PyTorch with RTX 5080 support
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Step 2: Install other dependencies
uv pip install -r requirements-uv.txt

# Step 3: Install Unsloth separately (UV Git dependency fix)
uv pip install git+https://github.com/unslothai/unsloth.git
```

### **For Regular Pip:**

```bash
# Step 1: Install PyTorch with RTX 5080 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Step 2: Install other dependencies
pip install -r requirements.txt
```

## 🔧 **Manual Installation Steps**

### **1. PyTorch Installation:**

```bash
# Try CUDA 12.4 stable first
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# If that fails, use nightly build (recommended for RTX 5080)
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

### **2. Core Dependencies:**

```bash
uv pip install transformers>=4.44.0 datasets>=2.20.0
uv pip install accelerate>=0.33.0 "bitsandbytes>=0.45.5"
uv pip install peft>=0.12.0 trl>=0.10.0
uv pip install python-dotenv pandas numpy tqdm colorama rich
```

### **3. Unsloth Installation:**

```bash
# UV handles Git dependencies differently
uv pip install git+https://github.com/unslothai/unsloth.git
```

## ✅ **Verification Commands**

```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Capability: {torch.cuda.get_device_capability(0)}")
```

## 🐛 **Common Issues & Solutions**

### **Issue 1: "No module named 'transformers'"**
```bash
uv pip install transformers datasets accelerate
```

### **Issue 2: "CUDA capability sm_120 is not compatible"**
```bash
# Install nightly PyTorch build
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

### **Issue 3: "bitsandbytes not found"**
```bash
uv pip install "bitsandbytes>=0.45.5"
```

### **Issue 4: "UV Git dependency failed"**
```bash
# Install Unsloth manually
uv pip install git+https://github.com/unslothai/unsloth.git

# If still fails, use pip for this package
pip install git+https://github.com/unslothai/unsloth.git
```

## 🎯 **Quick Start for Bengali Training**

After installation, start Bengali training:

```bash
# Windows
start_bengali_training.bat

# Linux/Mac
python main_bengali.py --model_name "unsloth/llama-3.1-8b-bnb-4bit" --dataset_name "spedrox-sac/bengali_chat_conv"
```

## 📝 **Environment Variables (.env.bengali)**

Make sure your `.env.bengali` file has:

```env
# Model Configuration
MODEL_NAME=unsloth/llama-3.1-8b-bnb-4bit
DATASET_NAME=spedrox-sac/bengali_chat_conv
OUTPUT_DIR=./bengali-model-output

# Training Parameters
MAX_SEQ_LENGTH=2048
DTYPE=float16
LOAD_IN_4BIT=true

# Training Settings
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
WARMUP_STEPS=5
MAX_STEPS=60
LEARNING_RATE=2e-4
FP16=true

# Evaluation
RUN_EVALUATION=true
EVAL_STEPS=20
SAVE_STEPS=20

# Hardware
USE_GPU=true
GPU_MEMORY_FRACTION=0.8
```

## 🔄 **Alternative: Docker Setup**

If installation issues persist, consider using Docker:

```dockerfile
FROM pytorch/pytorch:2.5.0-devel-cuda12.4-cudnn9-ubuntu22.04

# Install UV
RUN pip install uv

# Copy requirements and install
COPY requirements-uv.txt .
RUN uv pip install -r requirements-uv.txt
RUN uv pip install git+https://github.com/unslothai/unsloth.git

# Copy Bengali training files
COPY . /workspace
WORKDIR /workspace
```

## 📞 **Support**

If you continue having issues:

1. Check GPU compatibility: `nvidia-smi`
2. Verify CUDA version: `nvcc --version`
3. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
4. Update GPU drivers if needed

**Last Updated:** September 2025 for RTX 5080 compatibility
