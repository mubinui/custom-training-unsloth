# Bengali Model Training Guide

This guide explains how to train a Bengali language model using the `spedrox-sac/bengali_chat_conv` dataset from Hugging Face.

## Quick Start

### 1. Switch to Bengali Training Branch
```bash
git checkout bengali-training
```

### 2. Setup Bengali Environment
```bash
python setup_bengali.py
```

### 3. Start Bengali Training
```bash
# Test with sample data first
python main_bengali.py --sample --eval

# Train with full dataset
python main_bengali.py --eval --push
```

## Bengali Dataset

### Dataset Information
- **Source**: `spedrox-sac/bengali_chat_conv` from Hugging Face
- **Type**: Conversational data in Bengali
- **Format**: Chat conversations converted to Alpaca format
- **Size**: Variable (can be limited with `--max-examples`)

### Data Processing Features
- Automatic conversion from chat conversations to Alpaca format
- Bengali text validation and cleaning
- Support for sample data generation for testing
- Streaming support for large datasets

## Configuration

### Environment Files
- `.env.bengali` - Bengali-specific configuration template
- `.env` - Active environment configuration (created from template)

### Key Bengali Settings
```bash
# Model Configuration
MODEL_NAME="unsloth/llama-3.2-3b-instruct-bnb-4bit"
MODEL_TYPE="llama"

# Bengali Dataset
DATASET_NAME="spedrox-sac/bengali_chat_conv"
DATASET_SPLIT="train"

# Enhanced LoRA for Bengali
LORA_R=32
LORA_ALPHA=32
LORA_DROPOUT=0.1

# Bengali-optimized training
LEARNING_RATE=3e-4
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
MAX_STEPS=500

# Memory optimization
USE_GRADIENT_CHECKPOINTING=true
DATALOADER_PIN_MEMORY=false
```

## Bengali Training Commands

### Basic Training
```bash
# Train with full Bengali dataset
python main_bengali.py

# Train with evaluation
python main_bengali.py --eval

# Train and push to Hugging Face Hub
python main_bengali.py --eval --push
```

### Testing and Development
```bash
# Quick test with sample data
python main_bengali.py --sample

# Test with limited examples
python main_bengali.py --max-examples 100 --eval

# Use different dataset split
python main_bengali.py --dataset-split validation
```

### Advanced Options
```bash
# Create new environment file
python main_bengali.py --create-env

# Run automated test
python test_bengali_training.py

# Setup and verify environment
python setup_bengali.py
```

## Data Format

### Input Format (Bengali Chat Conversations)
The dataset contains Bengali chat conversations that are automatically converted to Alpaca format:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "মেশিন লার্নিং কী?"
    },
    {
      "from": "gpt", 
      "value": "মেশিন লার্নিং হল কৃত্রিম বুদ্ধিমত্তার একটি শাখা..."
    }
  ]
}
```

### Output Format (Alpaca)
```json
{
  "instruction": "মেশিন লার্নিং কী?",
  "input": "",
  "output": "মেশিন লার্নিং হল কৃত্রিম বুদ্ধিমত্তার একটি শাখা..."
}
```

## Bengali Chat Template

The training uses a Bengali-specific chat template:

```
<|im_start|>system
আপনি একজন সহায়ক AI সহায়ক। আপনি বাংলায় উত্তর দেন।<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

## Evaluation Prompts

Bengali-specific evaluation prompts are used to test the model:

1. **Identity**: "আপনার নাম কি?" (What is your name?)
2. **Technical**: "মেশিন লার্নিং সম্পর্কে ব্যাখ্যা করুন" (Explain machine learning)
3. **Programming**: "পাইথনে একটি সিম্পল ফাংশন লিখুন" (Write a simple function in Python)
4. **Cultural**: "বাংলাদেশ সম্পর্কে কিছু বলুন" (Tell me about Bangladesh)
5. **Opinion**: "কৃত্রিম বুদ্ধিমত্তার ভবিষ্যৎ নিয়ে আপনার মতামত কি?" (What is your opinion about the future of AI?)

## Performance Optimization

### For Bengali Language
- **Higher LoRA rank** (32) for better Bengali language modeling
- **Optimized batch sizes** for memory efficiency
- **Gradient checkpointing** enabled for larger models
- **Enhanced learning rate** for multilingual training

### Memory Management
- Automatic memory optimization based on available GPU
- Support for gradient accumulation to handle larger effective batch sizes
- Pin memory disabled for better compatibility

## Troubleshooting

### Common Issues

#### 1. Dataset Loading Issues
```bash
# Test dataset access
python -c "from datasets import load_dataset; ds = load_dataset('spedrox-sac/bengali_chat_conv', split='train[:10]'); print(len(ds))"
```

#### 2. Memory Issues
- Reduce `BATCH_SIZE` in `.env`
- Enable `USE_GRADIENT_CHECKPOINTING=true`
- Increase `GRADIENT_ACCUMULATION_STEPS`

#### 3. CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 4. Import Errors on macOS
This is expected during development. The actual training should be done on a machine with proper GPU support and all dependencies installed.

### Debug Mode
Enable verbose logging in `.env`:
```bash
VERBOSE_LOGGING=true
DEBUG_MODE=true
```

## Output Files

### Training Results
- `outputs/bengali_model/` - Trained model files
- `outputs/bengali_model/bengali_training_summary.json` - Training summary
- Training logs in console output

### Model Artifacts
- Model weights and configuration
- Tokenizer files
- Training checkpoints (if enabled)
- LoRA adapter weights

## Integration with Main Project

### Branch Structure
- `main` - General training system
- `bengali-training` - Bengali-specific training

### File Organization
- `main_bengali.py` - Bengali training script
- `bengali_data_processor.py` - Bengali data processing
- `.env.bengali` - Bengali configuration template
- `setup_bengali.py` - Bengali setup script
- `BENGALI_TRAINING_GUIDE.md` - This guide

### Environment Management
The Bengali training system is fully compatible with the main environment configuration system, allowing seamless deployment on Windows machines with minimal configuration changes.

## Next Steps

1. **Test the setup**: Run `python setup_bengali.py`
2. **Quick test**: Run `python main_bengali.py --sample --eval`
3. **Full training**: Run `python main_bengali.py --eval`
4. **Deploy**: Copy to Windows machine and use the same commands

## Support

For issues specific to Bengali training:
1. Check that you're on the `bengali-training` branch
2. Verify `.env` configuration matches Bengali requirements
3. Test with sample data first (`--sample` flag)
4. Check GPU and CUDA availability
5. Review the comprehensive environment configuration guide in `ENV_CONFIG_GUIDE.md`
