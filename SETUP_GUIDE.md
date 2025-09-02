# System Requirements and Setup Guide

## Hardware Requirements

### Minimum Requirements

#### For Gemma 2B Model
- **GPU**: NVIDIA RTX 3060 (12GB) or better
- **RAM**: 16GB system RAM  
- **Storage**: 50GB free space (SSD recommended)
- **CUDA**: Version 11.8 or 12.1+

#### For Gemma 9B Model
- **GPU**: NVIDIA RTX 4080 (16GB) or RTX 3090 (24GB)
- **RAM**: 32GB system RAM
- **Storage**: 100GB free space (SSD recommended)
- **CUDA**: Version 11.8 or 12.1+

### Recommended Requirements
- **GPU**: NVIDIA RTX 4090 (24GB) or A100
- **RAM**: 64GB system RAM
- **Storage**: 200GB NVMe SSD
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9
- **CUDA**: Latest version

### GPU Memory Usage

| Model Size | 4-bit Quantization | LoRA Training | Batch Size 2 |
|------------|-------------------|---------------|--------------|
| Gemma 2B   | ~3GB             | ~6GB          | ~8GB         |
| Gemma 9B   | ~6GB             | ~12GB         | ~16GB        |
| Llama 8B   | ~5GB             | ~10GB         | ~14GB        |

### Performance Estimates

#### Training Speed (steps/minute)
- **RTX 3060 12GB**: ~2-3 steps/min (Gemma 2B)
- **RTX 4080 16GB**: ~5-7 steps/min (Gemma 9B)
- **RTX 4090 24GB**: ~8-12 steps/min (Gemma 9B)

## Software Dependencies

### Python Environment
- **Python**: 3.8-3.11 (3.10 recommended)
- **CUDA**: 11.8 or 12.1
- **PyTorch**: 2.4.1 with CUDA support

### Core Dependencies
- **torch**: 2.4.1 (CUDA 12.1 compatible)
- **transformers**: 4.44.x (supports latest Gemma models)
- **unsloth**: Latest from GitHub
- **datasets**: 2.21.x (compatible with transformers)
- **peft**: 0.12.x (latest LoRA implementations)

### Installation Strategy

#### Option 1: Automated Setup (Recommended)
```bash
# Windows
setup.bat

# Linux/Mac  
./setup.sh
```

#### Option 2: Manual Installation
```bash
# 1. Create environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install PyTorch with CUDA
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install other dependencies
pip install -r requirements.txt
```

### Dependency Files
- **requirements.txt**: Flexible version ranges (recommended)
- **requirements-exact.txt**: Exact pinned versions (if conflicts occur)

## Troubleshooting Common Issues

### CUDA Out of Memory
**Solutions**:
- Reduce batch size to 1
- Increase gradient accumulation steps
- Use smaller model variant
- Reduce max_seq_length

### Installation Problems
**"No module named 'torch'"**:
```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

**"Microsoft Visual C++ required"** (Windows):
- Install Visual Studio Build Tools
- Or use conda: `conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia`

### Performance Issues
**Slow Training**:
- Check GPU utilization: `nvidia-smi`
- Ensure CUDA drivers are updated
- Close other GPU-intensive applications
- Use SSD for data storage

**Memory Warnings**:
- Monitor with `nvidia-smi -l 5`
- Reduce batch size if >90% memory usage
- Use gradient checkpointing for large sequences

## Environment Verification

### Quick Tests
```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test Unsloth import
python -c "import unsloth; print('Unsloth ready!')"

# Check GPU memory
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')"
```

### System Info Script
```python
import torch
from transformers import __version__ as transformers_version

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers_version}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
```

## Tips for Lower-End Hardware

### 8GB GPU Configuration
```json
{
  "model_name": "unsloth/gemma-2-2b-it-bnb-4bit",
  "max_seq_length": 1024,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "r": 8
}
```

### 12GB GPU Configuration  
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_seq_length": 2048,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "r": 16
}
```

### 16GB+ GPU Configuration
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit", 
  "max_seq_length": 2048,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "r": 24
}
```

This guide covers all essential system requirements and setup procedures for successful fine-tuning with the Unsloth training system.
