# Quick Reference Card

## Essential Commands

### Setup
```bash
# Windows setup
setup.bat

# Linux/Mac setup
./setup.sh

# Activate environment
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### Training Commands
```bash
# Quick test
python main.py --sample --eval

# Train with your data
python main.py --data your_data.json --eval

# Custom config
python main.py --config custom.json --data your_data.json

# No save (testing)
python main.py --data your_data.json --no-save

# Push to Hub
python main.py --data your_data.json --push
```

### Evaluation Commands
```bash
# Interactive evaluation
python evaluate.py --model_path ./outputs --interactive

# Batch evaluation
python evaluate.py --model_path ./outputs

# Custom prompts
python evaluate.py --model_path ./outputs --prompts test.txt
```

### Monitoring
```bash
# Watch training logs
tail -f logs/training.log

# Monitor GPU
nvidia-smi -l 5

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## Common Configurations

### Quick Test (2GB GPU)
```json
{
  "model_name": "unsloth/gemma-2-2b-it-bnb-4bit",
  "max_seq_length": 1024,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "max_steps": 50,
  "learning_rate": 5e-4
}
```

### Production (16GB GPU)
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_seq_length": 2048,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "max_steps": 500,
  "learning_rate": 2e-4,
  "warmup_steps": 50
}
```

### Memory Optimization
```json
{
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "max_seq_length": 1024,
  "r": 8,
  "use_gradient_checkpointing": "unsloth"
}
```

## Data Format Examples

### Alpaca Format
```json
[
  {
    "instruction": "Task description",
    "input": "Optional context",
    "output": "Expected response"
  }
]
```

### Simple Text Format
```json
[
  {
    "text": "### Instruction:\nTask\n\n### Response:\nAnswer"
  }
]
```

## Troubleshooting Quick Fixes

### CUDA Out of Memory
1. Reduce `per_device_train_batch_size` to 1
2. Increase `gradient_accumulation_steps` to 8+
3. Reduce `max_seq_length` to 1024
4. Use smaller model (gemma-2-2b)

### Slow Training
1. Check GPU utilization: `nvidia-smi`
2. Increase batch size if memory allows
3. Enable `fp16` or `bf16`
4. Reduce `gradient_accumulation_steps`

### Poor Quality Output
1. Increase `max_steps`
2. Improve data quality
3. Increase `r` (LoRA rank)
4. Lower `learning_rate`

### Installation Issues
1. Try `requirements-exact.txt`
2. Install PyTorch separately first
3. Clear pip cache: `pip cache purge`
4. Use conda environment

## File Structure
```
project/
├── main.py              # Training script
├── evaluate.py          # Evaluation script
├── config.py            # Configuration classes
├── requirements.txt     # Dependencies
├── setup.bat           # Windows setup
├── sample_data.json    # Example data
├── outputs/            # Trained models
├── logs/              # Training logs
└── data/              # Your datasets
```

## Model Options
- `unsloth/gemma-2-2b-it-bnb-4bit` (Small, fast)
- `unsloth/gemma-2-9b-it-bnb-4bit` (Balanced)
- `unsloth/llama-3-8b-bnb-4bit` (Alternative)
- `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` (Instruction-tuned)

## Key Parameters
- `max_steps`: Training duration (50-1000+)
- `learning_rate`: 1e-5 to 5e-4
- `r`: LoRA rank (8-64)
- `max_seq_length`: Input/output length (1024-4096)
- `per_device_train_batch_size`: Memory vs speed (1-8)

## Support Resources
- DEPENDENCIES.md: Detailed compatibility info
- HARDWARE_REQUIREMENTS.md: GPU requirements
- README.md: Complete documentation
- logs/training.log: Training progress
