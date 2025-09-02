# Custom Training with Unsloth - Bengali Model

Professional fine-tuning system for Bengali language models using Unsloth and the spedrox-sac/bengali_chat_conv dataset.

## Quick Start for Windows RTX 5080

### Prerequisites
- Python 3.10-3.12
- NVIDIA RTX 5080 GPU with 16GB+ VRAM
- Git
- 20GB+ free disk space

### Setup (One-time)

1. **Clone the repository:**
```cmd
git clone https://github.com/mubinui/custom-training-unsloth.git
cd custom-training-unsloth
```

2. **Run setup (automatically cleans UV environments):**
```cmd
setup.bat
```

3. **If you had UV issues, use clean setup:**
```cmd
clean_setup.bat
```

4. **Activate environment for future use:**
```cmd
venv\Scripts\activate.bat
```

### Manual Setup (if automated setup fails)

1. **Create virtual environment:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

2. **Install PyTorch for RTX 5080:**
```cmd
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

3. **Install other dependencies:**
```cmd
pip install transformers datasets accelerate peft trl python-dotenv
pip install pandas numpy tqdm colorama rich safetensors protobuf packaging
pip install "bitsandbytes>=0.45.5"
```

4. **Install Unsloth:**
```cmd
pip install git+https://github.com/unslothai/unsloth.git
```

### Training Commands

**Test with sample data:**
```cmd
python main_bengali.py --sample --eval
```

**Train with limited data (recommended first):**
```cmd
python main_bengali.py --max-examples 200 --eval
```

**Full training:**
```cmd
python main_bengali.py --eval
```

**Train and push to Hugging Face Hub:**
```cmd
python main_bengali.py --eval --push
```

### Environment Configuration

Create or edit `.env` file:
```env
MODEL_NAME=unsloth/llama-3.1-8b-bnb-4bit
DATASET_NAME=spedrox-sac/bengali_chat_conv
OUTPUT_DIR=./bengali-model-output
MAX_SEQ_LENGTH=2048
DTYPE=float16
LOAD_IN_4BIT=true
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
WARMUP_STEPS=5
MAX_STEPS=60
LEARNING_RATE=2e-4
FP16=true
RUN_EVALUATION=true
EVAL_STEPS=20
SAVE_STEPS=20
USE_GPU=true
GPU_MEMORY_FRACTION=0.8
```

### Troubleshooting

**Issue: "No module named 'unsloth'" or triton errors**
This is normal on Windows. The system will use fallback methods.

**Issue: "CUDA capability sm_120 is not compatible"**
This warning is normal with CUDA 12.1. Training will still work but may be slower.

**Issue: "No module named 'transformers'"**
```cmd
venv\Scripts\activate.bat
pip install transformers datasets accelerate
```

**Issue: "bitsandbytes not found"**
```cmd
pip install "bitsandbytes>=0.45.5"
```

**Issue: "Import error: triton"**
This is normal on Windows. Triton is not supported on Windows and is not required.

**Issue: "Flash attention compilation failed"**
This is normal on Windows. Flash attention will be skipped automatically.

### Verification Commands

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

### Project Structure

```
custom-training-unsloth/
├── main_bengali.py          # Main Bengali training script
├── bengali_data_processor.py # Bengali dataset processor
├── trainer.py               # Unsloth trainer wrapper
├── env_config.py           # Environment configuration
├── config.py               # Legacy configuration
├── setup.bat               # Windows setup script
├── requirements.txt        # Package requirements
├── .env.bengali           # Bengali config template
└── README.md              # This file
```

### Dataset Information

- **Dataset**: spedrox-sac/bengali_chat_conv
- **Size**: 1,074 conversation pairs
- **Language**: Bengali
- **Format**: Chat conversations
- **Access**: Public (no HuggingFace token required)

### Hardware Requirements

- **GPU**: NVIDIA RTX 5080 (16GB+ VRAM)
- **CUDA**: 12.4+ (automatically installed)
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space
- **OS**: Windows 10/11

### Performance Tips

1. **Start with sample data** to test setup
2. **Use limited examples** (--max-examples 200) for quick testing
3. **Monitor GPU memory** usage during training
4. **Adjust batch size** if you encounter OOM errors
5. **Use FP16** for better memory efficiency

### Support

If you encounter issues:
1. Check GPU drivers are updated
2. Verify CUDA version with `nvidia-smi`
3. Test PyTorch CUDA with verification commands above
4. Check that virtual environment is activated

Last Updated: September 2025 for RTX 5080 compatibility
