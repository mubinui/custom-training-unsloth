# Environment Configuration Guide

This guide explains how to use the new environment-based configuration system for Unsloth custom training.

## Overview

The environment configuration system provides:
- **Simple Setup**: Configure training with `.env` files
- **Secure**: Keep API tokens and secrets separate from code
- **Windows-Friendly**: Perfect for deployment on Windows machines
- **Validation**: Built-in validation and error checking
- **Flexibility**: Override settings via command line or environment variables

## Quick Start

### 1. Setup
```bash
# Windows
setup_env.bat

# Linux/Mac
./setup_env.sh
```

### 2. Configure
Edit the `.env` file:
```env
MODEL_NAME=unsloth/gemma-2-9b-it-bnb-4bit
DATA_PATH=your_data.json
MAX_STEPS=100
LEARNING_RATE=2e-4
```

### 3. Train
```bash
# Windows
train_env.bat

# Any platform
python main_env.py
```

## Configuration Options

### Essential Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `unsloth/gemma-2-9b-it-bnb-4bit` | Model to fine-tune |
| `DATA_PATH` | `sample_data.json` | Path to training data |
| `OUTPUT_DIR` | `./outputs` | Where to save trained model |
| `NEW_MODEL_NAME` | `gemma-2-9b-custom-finetune` | Name for saved model |

### Training Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `2` | Batch size per GPU |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | Gradient accumulation steps |
| `MAX_STEPS` | `60` | Maximum training steps |
| `LEARNING_RATE` | `2e-4` | Learning rate |
| `WARMUP_STEPS` | `5` | Number of warmup steps |

### LoRA Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LORA_R` | `16` | LoRA rank (lower = fewer parameters) |
| `LORA_ALPHA` | `16` | LoRA alpha (scaling factor) |
| `LORA_DROPOUT` | `0.0` | LoRA dropout rate |
| `LORA_TARGET_MODULES` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Layers to apply LoRA |

### System Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_SEQ_LENGTH` | `2048` | Maximum sequence length |
| `LOAD_IN_4BIT` | `true` | Use 4-bit quantization |
| `USE_GRADIENT_CHECKPOINTING` | `unsloth` | Gradient checkpointing mode |
| `MEMORY_OPTIMIZATION` | `medium` | Memory optimization level |

### Hugging Face Integration

| Variable | Default | Description |
|----------|---------|-------------|
| `PUSH_TO_HUB` | `false` | Upload model to Hugging Face |
| `HF_TOKEN` | - | Your Hugging Face token |
| `HF_USERNAME` | - | Your Hugging Face username |

### Evaluation Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_EVALUATION` | `true` | Run evaluation after training |
| `EVAL_SAMPLES` | `10` | Number of samples to evaluate |
| `EVAL_TEMPERATURE` | `0.7` | Generation temperature |
| `EVAL_MAX_NEW_TOKENS` | `256` | Maximum tokens to generate |

### Logging and Monitoring

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `VERBOSE_LOGGING` | `false` | Enable detailed logging |
| `LOG_FILE` | - | Log file path (empty for console only) |

## Usage Examples

### Example 1: Quick Test
```bash
# Setup
setup_env.bat

# Test with sample data
python main_env.py --sample --eval
```

### Example 2: Custom Model Training
```env
# .env file
MODEL_NAME=unsloth/gemma-2-9b-it-bnb-4bit
DATA_PATH=my_training_data.json
MAX_STEPS=200
LEARNING_RATE=2e-4
NEW_MODEL_NAME=my-custom-gemma
OUTPUT_DIR=./my_models
```

```bash
python main_env.py
```

### Example 3: Production Setup with Hugging Face
```env
# .env file
MODEL_NAME=unsloth/gemma-2-9b-it-bnb-4bit
DATA_PATH=production_data.json
MAX_STEPS=500
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=1e-4
WARMUP_STEPS=50

# Hugging Face settings
PUSH_TO_HUB=true
HF_TOKEN=hf_your_token_here
HF_USERNAME=your_username
NEW_MODEL_NAME=custom-gemma-production

# Evaluation
RUN_EVALUATION=true
EVAL_SAMPLES=20
```

```bash
python main_env.py
```

### Example 4: Memory-Optimized Training
```env
# .env file for smaller GPUs
MODEL_NAME=unsloth/gemma-2-9b-it-bnb-4bit
DATA_PATH=data.json
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
MAX_SEQ_LENGTH=1024
MEMORY_OPTIMIZATION=high
USE_MIXED_PRECISION=true
DATALOADER_NUM_WORKERS=0
```

## Command Line Overrides

You can override environment variables with command line arguments:

```bash
# Override data file
python main_env.py --data different_data.json

# Force evaluation
python main_env.py --eval

# Force push to hub
python main_env.py --push

# Use sample data (ignores DATA_PATH)
python main_env.py --sample
```

## Environment File Management

### Multiple Environments
```bash
# Development environment
cp .env.example .env.dev
# Edit .env.dev for development settings

# Production environment  
cp .env.example .env.prod
# Edit .env.prod for production settings

# Use specific environment
cp .env.dev .env && python main_env.py
cp .env.prod .env && python main_env.py
```

### Environment Templates
```bash
# Create template for experiments
python main_env.py --create-env
# This creates .env with current settings
```

## Best Practices

### 1. Security
- Never commit `.env` files to version control
- Use different tokens for development and production
- Set appropriate file permissions on `.env` files

### 2. Organization
```
project/
├── .env                    # Current active settings
├── .env.example           # Template with comments
├── .env.development       # Development settings
├── .env.production        # Production settings
└── environments/
    ├── experiment1.env    # Experiment configurations
    └── experiment2.env
```

### 3. Validation
The system validates your configuration and provides helpful error messages:
```bash
# Invalid settings will show clear errors
ERROR: LEARNING_RATE must be positive
ERROR: SAVE_METHOD must be one of: ['merged_16bit', 'lora_only', 'gguf']
```

### 4. Logging
```env
# Enable detailed logging for debugging
LOG_LEVEL=DEBUG
VERBOSE_LOGGING=true
LOG_FILE=logs/training.log
```

## Migration from JSON Configuration

If you have existing JSON configurations, you can convert them:

### Manual Migration
```json
// Old config.json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "learning_rate": 2e-4,
  "max_steps": 100
}
```

```env
# New .env
MODEL_NAME=unsloth/gemma-2-9b-it-bnb-4bit
LEARNING_RATE=2e-4
MAX_STEPS=100
```

### Automatic Migration
The system still supports JSON configs for backward compatibility:
```bash
# Still works
python main.py --config old_config.json
```

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   # Install python-dotenv
   pip install python-dotenv
   ```

2. **Environment variables not loading**
   ```bash
   # Check .env file exists and has correct format
   # Ensure no spaces around = sign
   MODEL_NAME=unsloth/gemma-2-9b-it-bnb-4bit  # Correct
   MODEL_NAME = unsloth/gemma-2-9b-it-bnb-4bit  # Wrong
   ```

3. **Configuration validation errors**
   ```bash
   # Check variable types (true/false for booleans, numbers for integers)
   LOAD_IN_4BIT=true     # Correct
   LOAD_IN_4BIT=True     # Wrong
   MAX_STEPS=100         # Correct
   MAX_STEPS=100.0       # Wrong (should be integer)
   ```

### Getting Help

1. **Check configuration**:
   ```bash
   python main_env.py --create-env
   # This shows current settings
   ```

2. **Enable verbose logging**:
   ```env
   LOG_LEVEL=DEBUG
   VERBOSE_LOGGING=true
   ```

3. **Use sample data for testing**:
   ```bash
   python main_env.py --sample
   ```

## Advanced Usage

### Environment Hierarchy
Settings are loaded in this order (later overrides earlier):
1. Default values in code
2. `.env` file
3. System environment variables
4. Command line arguments

### Custom Environment Variables
You can set environment variables directly:
```bash
# Windows
set MODEL_NAME=unsloth/different-model
python main_env.py

# Linux/Mac
MODEL_NAME=unsloth/different-model python main_env.py
```

### Batch Processing
```bash
# Process multiple datasets
for dataset in data1.json data2.json data3.json; do
    DATA_PATH=$dataset OUTPUT_DIR=./outputs/$dataset python main_env.py
done
```

This environment configuration system makes it much easier to deploy and manage your training on Windows machines while maintaining security and flexibility.
