# Unsloth Custom Training System

A comprehensive system for fine-tuning language models using Unsloth, with environment-based configuration for easy deployment and management. **Now includes specialized Bengali language model training!**

## 🚀 Key Features

- **Environment Configuration**: Simple .env file configuration for easy deployment
- **Efficient Fine-tuning**: Uses Unsloth for 2x faster training with QLoRA
- **Flexible Data Loading**: Supports JSON, JSONL, CSV, and Alpaca formats
- **Memory Efficient**: 4-bit quantization and gradient checkpointing
- **Multiple Save Formats**: LoRA adapters, merged 16-bit, or GGUF
- **Evaluation Tools**: Built-in model evaluation capabilities
- **Comprehensive Logging**: Detailed logging for monitoring training progress
- **Windows Optimized**: Designed for Windows deployment with NVIDIA GPUs
- **🇧🇩 Bengali Training**: Specialized branch for Bengali language models using HuggingFace datasets

## ⚡ Quick Start

### 1. Setup (Windows)
```bash
setup_env.bat
```

### 2. Configure
Edit the `.env` file to set your preferences:
```env
MODEL_NAME=unsloth/gemma-2-9b-it-bnb-4bit
DATA_PATH=your_data.json
MAX_STEPS=100
LEARNING_RATE=2e-4
```

### 3. Train
```bash
train_env.bat                 # Use your configured data
train_env.bat --sample        # Test with sample data
```

## 🇧🇩 Bengali Training (NEW!)

This project now includes specialized Bengali language model training using the `spedrox-sac/bengali_chat_conv` dataset.

### Quick Start for Bengali Training

```bash
# Switch to Bengali training branch
git checkout bengali-training

# Setup Bengali environment
python setup_bengali.py

# Test with sample Bengali data
python main_bengali.py --sample --eval

# Train with full Bengali dataset
python main_bengali.py --eval --push
```

### Bengali Training Features
- **Specialized Data Processing**: Automatic conversion from Bengali chat conversations to Alpaca format
- **Bengali Chat Template**: Custom Bengali system prompts and conversation formatting
- **Enhanced LoRA Configuration**: Optimized for Bengali language modeling (rank 32, alpha 32)
- **Bengali Evaluation Prompts**: Built-in Bengali test prompts for model evaluation
- **HuggingFace Integration**: Direct loading from `spedrox-sac/bengali_chat_conv` dataset

📖 **Full Bengali Training Guide**: See [BENGALI_TRAINING_GUIDE.md](BENGALI_TRAINING_GUIDE.md) for comprehensive documentation.

## 📦 Installation Options
```

## 🔧 Requirements

- Windows machine with NVIDIA GPU (RTX 20 series or newer recommended)
- CUDA 11.8 or 12.1+
- Python 3.10, 3.11, or 3.12
- At least 8GB GPU memory (16GB+ recommended for larger models)

## 📦 Installation Options

### Option 1: Environment-Based Setup (Recommended)

**Windows:**
```bash
setup_env.bat
```

**Linux/Mac:**
```bash
chmod +x setup_env.sh
./setup_env.sh
```

### Option 2: Legacy JSON Configuration

**Windows:**
```bash
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

### Option 3: Manual Installation

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install PyTorch with CUDA support:
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

4. Install other dependencies:
```bash
pip install -r requirements.txt
```

**Note**: If you encounter dependency conflicts, try `requirements-exact.txt` instead. See `SETUP_GUIDE.md` for detailed system requirements and troubleshooting.

## Documentation

This project includes comprehensive documentation:

- **README.md**: Main documentation with complete usage guide
- **DATA_REQUIREMENTS_GUIDE.md**: Detailed guide on data requirements and fine-tuning strategies
- **SETUP_GUIDE.md**: System requirements, installation, and troubleshooting
- **QUICK_REFERENCE.md**: Quick command reference for daily use
- **ENV_CONFIG_GUIDE.md**: Complete environment configuration reference
- **🇧🇩 BENGALI_TRAINING_GUIDE.md**: Specialized Bengali training documentation

## How to Use

### Complete Workflow Guide

#### Step 1: Environment Setup

1. **Run automated setup** (recommended):
```bash
# Windows
setup.bat

# Linux/Mac
./setup.sh
```

2. **Verify installation**:
```bash
# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test Unsloth import
python -c "import unsloth; print('Unsloth ready!')"
```

#### Step 2: Prepare Your Data

The system supports multiple data formats. Choose the one that fits your data:

**Option A: Alpaca Format (Recommended)**
Create a JSON file with instruction-input-output format:
```json
[
  {
    "instruction": "Explain machine learning in simple terms",
    "input": "",
    "output": "Machine learning is a technology that enables computers to learn patterns from data and make predictions without being explicitly programmed for each task."
  },
  {
    "instruction": "Write a Python function",
    "input": "Create a function to calculate factorial",
    "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
  }
]
```

**Option B: Simple Text Format**
```json
[
  {
    "text": "### Instruction:\nExplain quantum computing\n\n### Response:\nQuantum computing uses quantum mechanical phenomena to process information in fundamentally different ways than classical computers."
  }
]
```

**Option C: Chat Format**
```json
[
  {
    "conversations": [
      {"role": "user", "content": "What is artificial intelligence?"},
      {"role": "assistant", "content": "AI is the simulation of human intelligence in machines designed to think and act like humans."}
    ]
  }
]
```

#### Step 3: Configure Training Parameters

🎯 **Environment-Based Configuration (Recommended)**

The new environment configuration system uses `.env` files for easy management:

1. **Copy the template**:
```bash
cp .env.example .env  # Linux/Mac
copy .env.example .env  # Windows
```

2. **Edit your settings**:
```env
# Essential Settings
MODEL_NAME=unsloth/gemma-2-9b-it-bnb-4bit
DATA_PATH=your_data.json
OUTPUT_DIR=./outputs
NEW_MODEL_NAME=my-custom-model

# Training Settings
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
MAX_STEPS=100
LEARNING_RATE=2e-4

# LoRA Settings
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.0

# Optional: Hugging Face Integration
PUSH_TO_HUB=false
# HF_TOKEN=your_hf_token_here
```

3. **Use environment configuration**:
```bash
# Train with environment settings
python main_env.py

# Test with sample data
python main_env.py --sample

# Override specific settings
python main_env.py --data my_data.json --eval
```

**Alternative: JSON Configuration (Legacy)**

Edit `config.json` or create a new configuration file:
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_seq_length": 2048,
  "load_in_4bit": true,
  "r": 16,
  "lora_alpha": 16,
  "lora_dropout": 0.0,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "learning_rate": 2e-4,
  "max_steps": 100,
  "warmup_steps": 10,
  "save_method": "merged_16bit",
  "output_dir": "./outputs",
  "save_model": true,
  "push_to_hub": false
}
```

**Key Parameters Explained:**
- `model_name`: Base model to fine-tune (see Model Options below)
- `max_seq_length`: Maximum input/output length (1024-4096)
- `r`: LoRA rank - higher = more parameters but more memory (8-64)
- `learning_rate`: Training speed (1e-5 to 5e-4)
- `max_steps`: Training duration (50-1000+ depending on data)
- `per_device_train_batch_size`: Batch size per GPU (1-8)
- `gradient_accumulation_steps`: Effective batch size multiplier

#### Step 4: Start Training

**Basic Training Commands:**

1. **Test with sample data first** (recommended):
```bash
python main.py --sample --eval
```

2. **Train with your data**:
```bash
python main.py --data training_data.json --eval
```

3. **Custom configuration**:
```bash
python main.py --config my_config.json --data training_data.json --eval
```

4. **Advanced options**:
```bash
# Skip model saving
python main.py --data training_data.json --no-save

# Push to Hugging Face Hub
python main.py --data training_data.json --push

# Specify output directory in config and data path
python main.py --config configs/experiment1.json --data datasets/my_data.json
```

#### Step 5: Monitor Training Progress

**Real-time Monitoring:**
```bash
# Watch training logs
tail -f logs/training.log

# Monitor GPU usage (separate terminal)
nvidia-smi -l 1
```

**Training Output Structure:**
```
outputs/
├── checkpoint-25/           # Training checkpoints
├── checkpoint-50/
├── adapter_config.json      # LoRA configuration
├── adapter_model.safetensors # LoRA weights
├── tokenizer_config.json    # Tokenizer settings
└── training_args.bin        # Training arguments

logs/
└── training.log             # Detailed training logs
```

#### Step 6: Evaluate Your Model

**Built-in Evaluation with Environment Config:**
```bash
# Set evaluation in .env file
RUN_EVALUATION=true
EVAL_SAMPLES=10

# Train and evaluate
python main_env.py

# Override evaluation settings
python main_env.py --eval
```

**Interactive Testing:**
```bash
python evaluate.py --model_path ./outputs --interactive
```
Then type prompts like:
```
### Instruction:
Explain how neural networks work
### Response:
```

**Batch Evaluation:**
```bash
# Use default test prompts
python evaluate.py --model_path ./outputs

# Use custom prompts from file
python evaluate.py --model_path ./outputs --prompts test_prompts.txt
```

### Detailed Usage Examples

#### Example 1: Quick Test Run (Environment Config)
```bash
# 1. Setup environment
setup_env.bat

# 2. Quick test (5 minutes)
python main_env.py --sample --eval
```

#### Example 2: Environment-Based Training Workflow
```bash
# 1. Setup
setup_env.bat

# 2. Configure training in .env
echo MODEL_NAME=unsloth/gemma-2-9b-it-bnb-4bit >> .env
echo DATA_PATH=my_training_data.json >> .env
echo MAX_STEPS=200 >> .env
echo NEW_MODEL_NAME=my-custom-gemma >> .env

# 3. Train
python main_env.py

# 4. Test the model
python main_env.py --eval
```

#### Example 3: Legacy JSON Configuration
```bash
# 1. Setup environment
setup.bat

# 2. Quick test (5 minutes)
python main.py --sample --eval
```

#### Example 2: Small Dataset Training
```bash
# 1. Prepare data (50-100 examples)
# Save as training_data.json in Alpaca format

# 2. Quick training session
python main.py --data training_data.json --config quick_config.json --eval
```

Quick config (`quick_config.json`):
```json
{
  "model_name": "unsloth/gemma-2-2b-it-bnb-4bit",
  "max_steps": 50,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "learning_rate": 5e-4,
  "save_method": "lora"
}
```

#### Example 3: Production Training
```bash
# 1. Prepare large dataset (1000+ examples)
# 2. Configure for longer training
python main.py --data large_dataset.json --config production_config.json
```

Production config (`production_config.json`):
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_steps": 500,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "learning_rate": 2e-4,
  "warmup_steps": 50,
  "save_method": "merged_16bit",
  "push_to_hub": true,
  "hf_token": "your_token_here",
  "new_model_name": "your-username/custom-gemma-model"
}
```

#### Example 4: Multi-Experiment Workflow
```bash
# Create experiment directory structure
mkdir experiments
mkdir experiments/configs
mkdir experiments/data
mkdir experiments/results

# Experiment 1: Different learning rates
python main.py --config experiments/configs/lr_1e4.json --data experiments/data/dataset.json
python main.py --config experiments/configs/lr_2e4.json --data experiments/data/dataset.json
python main.py --config experiments/configs/lr_5e4.json --data experiments/data/dataset.json

# Compare results
python evaluate.py --model_path experiments/results/lr_1e4 --prompts test_prompts.txt
python evaluate.py --model_path experiments/results/lr_2e4 --prompts test_prompts.txt
python evaluate.py --model_path experiments/results/lr_5e4 --prompts test_prompts.txt
```

### Command Line Reference

**Main Training Script (`main.py`):**
```bash
python main.py [OPTIONS]

Options:
  --config PATH          Configuration file path (default: config.json)
  --data PATH           Training data file (JSON/JSONL/CSV)
  --sample              Use built-in sample data for testing
  --eval                Run evaluation after training
  --no-save             Skip saving the trained model
  --push                Push model to Hugging Face Hub
  -h, --help            Show help message
```

**Evaluation Script (`evaluate.py`):**
```bash
python evaluate.py --model_path PATH [OPTIONS]

Required:
  --model_path PATH     Path to trained model directory

Options:
  --config PATH         Configuration file (default: config.json)
  --prompts PATH        File with evaluation prompts (one per line)
  --interactive         Interactive evaluation mode
  --max_tokens INT      Maximum tokens to generate (default: 256)
  -h, --help            Show help message
```

### Quick Start

### 1. Test with Sample Data

```bash
python main.py --sample --eval
```

This will:
- Use built-in sample data
- Train for a short period
- Run evaluation with test prompts

### 2. Train with Your Data

```bash
python main.py --data path/to/your/data.json --eval
```

### 3. Custom Configuration

```bash
python main.py --config custom_config.json --data your_data.json
```

## Data Formats

### Alpaca Format (Recommended)
```json
[
  {
    "instruction": "Explain machine learning",
    "input": "",
    "output": "Machine learning is a subset of AI..."
  }
]
```

### Simple Text Format
```json
[
  {
    "text": "### Instruction:\nExplain machine learning\n\n### Response:\nMachine learning is..."
  }
]
```

### Chat Format
```json
[
  {
    "conversations": [
      {"role": "user", "content": "What is AI?"},
      {"role": "assistant", "content": "AI stands for..."}
    ]
  }
]
```

**For detailed data requirements, quality guidelines, and strategies for different dataset sizes, see `DATA_REQUIREMENTS_GUIDE.md`.**

## Configuration

The system uses a JSON configuration file. Key parameters:

```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_seq_length": 2048,
  "load_in_4bit": true,
  "r": 16,
  "lora_alpha": 16,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "learning_rate": 2e-4,
  "max_steps": 60,
  "save_method": "merged_16bit",
  "output_dir": "./outputs"
}
```

### Key Parameters:

- model_name: Base model to fine-tune
- max_seq_length: Maximum sequence length (2048 recommended for Gemma)
- r: LoRA rank (higher = more parameters, more memory)
- learning_rate: Learning rate (2e-4 is good starting point)
- max_steps: Number of training steps
- save_method: "lora", "merged_16bit", or "merged_4bit"

## Model Options

### Available Models:
- unsloth/gemma-2-9b-it-bnb-4bit (9B parameters, instruction-tuned)
- unsloth/gemma-2-2b-it-bnb-4bit (2B parameters, smaller/faster)
- unsloth/llama-3-8b-bnb-4bit (Llama 3 8B)
- unsloth/mistral-7b-instruct-v0.3-bnb-4bit (Mistral 7B)

## Usage Examples

### Basic Training
```bash
# Train with sample data
python main.py --sample

# Train with your data
python main.py --data training_data.json

# Train and evaluate
python main.py --data training_data.json --eval
```

### Advanced Options
```bash
# Custom configuration
python main.py --config my_config.json --data data.json

# Skip saving model
python main.py --data data.json --no-save

# Push to Hugging Face Hub
python main.py --data data.json --push
```

### Configuration Management
```bash
# The system automatically creates config.json on first run
# Edit config.json to customize training parameters
# Or create custom configs for different experiments
```

## Memory Requirements

| Model Size | Min GPU Memory | Recommended |
|------------|----------------|-------------|
| Gemma 2B   | 6GB           | 8GB         |
| Gemma 9B   | 12GB          | 16GB        |
| Llama 8B   | 10GB          | 16GB        |

## Output Structure

```
outputs/
├── checkpoint-*/          # Training checkpoints
├── adapter_config.json    # LoRA configuration
├── adapter_model.safetensors  # LoRA weights
├── tokenizer_config.json  # Tokenizer configuration
└── training_args.bin      # Training arguments

logs/
└── training.log          # Training logs

data/
└── processed_dataset/    # Processed dataset cache
```

## Troubleshooting

### CUDA Out of Memory
- Reduce per_device_train_batch_size
- Increase gradient_accumulation_steps
- Reduce max_seq_length
- Use smaller model

### Slow Training
- Increase per_device_train_batch_size
- Reduce gradient_accumulation_steps
- Ensure you're using GPU (check CUDA installation)

### Model Quality Issues
- Increase max_steps
- Adjust learning_rate
- Use more training data
- Increase LoRA r parameter

## Advanced Features

### Custom Data Processing
```python
from data_processor import DataProcessor

processor = DataProcessor()
dataset = processor.create_instruction_dataset(
    instructions=["Explain AI", "Write code"],
    inputs=["", "Python function"],
    outputs=["AI is...", "def function():..."]
)
```

### Model Evaluation
```python
from trainer import UnslothTrainer
from config import TrainingConfig

config = TrainingConfig()
trainer = UnslothTrainer(config)
trainer.load_model()

responses = trainer.evaluate_model([
    "### Instruction:\nExplain quantum computing\n\n### Response:\n"
])
```

### Hugging Face Hub Integration
Set your HF token in config:
```json
{
  "push_to_hub": true,
  "hf_token": "your_hf_token_here",
  "new_model_name": "your-username/custom-model"
}
```

## Best Practices and Tips

### Data Preparation Best Practices

#### Data Quality Guidelines
1. **Instruction Clarity**: Make instructions specific and clear
   ```json
   // Good
   {"instruction": "Write a Python function to calculate the factorial of a number", "output": "def factorial(n):..."}
   
   // Avoid
   {"instruction": "Write code", "output": "def factorial(n):..."}
   ```

2. **Response Quality**: Ensure outputs are accurate and well-formatted
   ```json
   // Good
   {"instruction": "Explain photosynthesis", "output": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen..."}
   
   // Avoid
   {"instruction": "Explain photosynthesis", "output": "plants make food from sun"}
   ```

3. **Data Diversity**: Include varied instruction types and domains
4. **Length Balance**: Mix short and long responses (50-500 tokens)
5. **Formatting Consistency**: Use consistent formatting across all examples

#### Dataset Size Recommendations
- **Testing/Learning**: 10-50 examples
- **Small Projects**: 100-500 examples  
- **Production**: 1000+ examples
- **Domain Specific**: 500-2000 examples per domain

### Training Configuration Strategies

#### Memory-Constrained Training (8GB GPU)
```json
{
  "model_name": "unsloth/gemma-2-2b-it-bnb-4bit",
  "max_seq_length": 1024,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "r": 8,
  "max_steps": 200
}
```

#### Balanced Training (16GB GPU)
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_seq_length": 2048,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "r": 16,
  "max_steps": 500
}
```

#### High-Performance Training (24GB+ GPU)
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_seq_length": 4096,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 2,
  "r": 32,
  "max_steps": 1000
}
```

### Training Monitoring and Optimization

#### Monitoring Training Progress
1. **Watch training logs in real-time**:
   ```bash
   # In separate terminal
   tail -f logs/training.log
   ```

2. **Monitor GPU usage**:
   ```bash
   # Check GPU memory and utilization
   nvidia-smi -l 5
   
   # Or use nvidia-ml-py for detailed monitoring
   watch -n 2 nvidia-smi
   ```

3. **Check training metrics**:
   - Loss should generally decrease over time
   - Learning rate should follow the schedule
   - GPU utilization should be >80%

#### Signs of Good Training
- **Loss decreases steadily**: Not too fast (overfitting) or too slow
- **No memory errors**: Training runs without CUDA OOM errors
- **Reasonable speed**: 2-10 steps per minute depending on hardware
- **Good evaluation results**: Model responses improve over time

#### Signs of Training Issues
- **Loss plateaus early**: Learning rate too low or data too easy
- **Loss oscillates wildly**: Learning rate too high
- **Memory errors**: Batch size too large, reduce or increase accumulation
- **Very slow training**: Check GPU utilization, might need optimization

### Advanced Usage Scenarios

#### Scenario 1: Domain-Specific Fine-tuning (Medical)
```bash
# Prepare medical instruction dataset
# Use longer training with lower learning rate for specialized knowledge
```

Config for medical domain:
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_seq_length": 2048,
  "learning_rate": 1e-4,
  "max_steps": 1000,
  "warmup_steps": 100,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4
}
```

#### Scenario 2: Code Generation Fine-tuning
```bash
# Prepare code instruction dataset with various programming languages
```

Config for code generation:
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_seq_length": 4096,
  "learning_rate": 3e-4,
  "max_steps": 800,
  "r": 32,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
```

#### Scenario 3: Multilingual Training
```bash
# Prepare dataset with instructions in multiple languages
```

Config for multilingual:
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_seq_length": 2048,
  "learning_rate": 2e-4,
  "max_steps": 1500,
  "warmup_steps": 150
}
```

### Troubleshooting Common Issues

#### Installation Problems

**Problem**: `ImportError: No module named 'torch'`
**Solution**:
```bash
# Install PyTorch first
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

**Problem**: `CUDA out of memory during installation`
**Solution**:
```bash
# Clear pip cache and try exact requirements
pip cache purge
pip install -r requirements-exact.txt
```

**Problem**: `Microsoft Visual C++ 14.0 is required`
**Solution**:
- Install Visual Studio Build Tools
- Or use conda: `conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia`

#### Training Problems

**Problem**: `CUDA out of memory during training`
**Solutions**:
1. Reduce batch size:
   ```json
   {"per_device_train_batch_size": 1}
   ```
2. Increase gradient accumulation:
   ```json
   {"gradient_accumulation_steps": 8}
   ```
3. Reduce sequence length:
   ```json
   {"max_seq_length": 1024}
   ```
4. Use smaller model:
   ```json
   {"model_name": "unsloth/gemma-2-2b-it-bnb-4bit"}
   ```

**Problem**: `Training is very slow`
**Solutions**:
1. Check GPU utilization: `nvidia-smi`
2. Increase batch size if memory allows
3. Enable mixed precision: `{"fp16": true}` or `{"bf16": true}`
4. Use gradient checkpointing: `{"use_gradient_checkpointing": "unsloth"}`

**Problem**: `Loss not decreasing`
**Solutions**:
1. Increase learning rate: `{"learning_rate": 5e-4}`
2. Check data quality and formatting
3. Increase training steps: `{"max_steps": 500}`
4. Add warmup steps: `{"warmup_steps": 50}`

**Problem**: `Model outputs are poor quality`
**Solutions**:
1. Increase LoRA rank: `{"r": 32}`
2. Train for more steps
3. Improve data quality and quantity
4. Use evaluation to monitor progress: `--eval`

#### Data Problems

**Problem**: `UnicodeDecodeError when loading data`
**Solution**:
```bash
# Save your data with UTF-8 encoding
# Or specify encoding in your data file
```

**Problem**: `Dataset format not recognized`
**Solution**:
- Ensure JSON is valid: use online JSON validator
- Check data structure matches expected format
- Use sample_data.json as reference

**Problem**: `Training data too large for memory`
**Solution**:
- Split data into smaller files
- Use streaming dataset loading
- Reduce max_seq_length

### Performance Optimization

#### GPU Optimization
1. **Use appropriate CUDA version**: Match PyTorch CUDA version
2. **Enable mixed precision**: Reduces memory usage by ~40%
3. **Optimize batch size**: Find maximum that fits in memory
4. **Use gradient checkpointing**: Trades compute for memory

#### Training Speed Optimization
1. **Increase batch size**: Better GPU utilization
2. **Use compiled models**: Enable torch.compile if available
3. **Optimize data loading**: Use multiple workers for data processing
4. **Profile training**: Identify bottlenecks with torch profiler

#### Memory Optimization
1. **Use 4-bit quantization**: Default setting, reduces memory by ~75%
2. **Gradient accumulation**: Simulate larger batches without more memory
3. **Sequence length optimization**: Use minimum required length
4. **LoRA rank tuning**: Balance between quality and memory usage

### Model Deployment and Usage

#### Saving Models for Production
```bash
# Save as merged 16-bit model (recommended for inference)
# Set in config: "save_method": "merged_16bit"

# Save as LoRA adapters (smallest size)
# Set in config: "save_method": "lora"

# Save as merged 4-bit model (good balance)
# Set in config: "save_method": "merged_4bit"
```

#### Loading Trained Models
```python
# Load for inference
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./outputs",  # Path to your trained model
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Enable inference mode
FastLanguageModel.for_inference(model)

# Generate response
inputs = tokenizer(["### Instruction:\nExplain AI\n\n### Response:\n"], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### Hugging Face Hub Integration
```bash
# Set up Hugging Face token
huggingface-cli login

# Configure for hub upload
```

Config for hub upload:
```json
{
  "push_to_hub": true,
  "hf_token": "your_hf_token_here",
  "new_model_name": "your-username/my-custom-model"
}
```

### Quality Assurance

#### Evaluation Best Practices
1. **Use diverse test prompts**: Cover different instruction types
2. **Manual review**: Check a sample of generated responses
3. **Automated metrics**: Use perplexity, BLEU, or custom metrics
4. **A/B testing**: Compare with baseline models

#### Model Validation Checklist
- [ ] Model loads without errors
- [ ] Generates coherent responses
- [ ] Follows instruction format
- [ ] Handles edge cases gracefully
- [ ] Performance meets requirements
- [ ] No harmful or biased outputs

## Performance Tips

1. Batch Size: Start with 2, increase if you have more memory
2. Learning Rate: 2e-4 works well for most cases
3. LoRA Rank: 16 is a good balance between quality and speed
4. Sequence Length: Use the minimum needed for your data
5. Mixed Precision: Enable fp16 or bf16 for faster training

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Unsloth for the efficient training framework
- Hugging Face for the transformers library
- Google for the Gemma models
