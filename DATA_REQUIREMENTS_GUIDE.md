# Data Requirements and Fine-tuning Guide

## Table of Contents
1. [Data Requirements Overview](#data-requirements-overview)
2. [Dataset Size Guidelines](#dataset-size-guidelines)
3. [Data Quality Standards](#data-quality-standards)
4. [Fine-tuning with Limited Data](#fine-tuning-with-limited-data)
5. [Domain-Specific Requirements](#domain-specific-requirements)
6. [Data Preparation Best Practices](#data-preparation-best-practices)
7. [Configuration Strategies by Data Size](#configuration-strategies-by-data-size)

## Data Requirements Overview

The quality and quantity of training data directly impacts fine-tuning success. This guide provides detailed requirements and strategies for different scenarios.

### Minimum Requirements
- **Absolute minimum**: 10-20 high-quality examples
- **Recommended minimum**: 50-100 examples
- **Production quality**: 500-1000+ examples
- **Specialized domains**: 1000-5000+ examples

### Data Types That Work Best
1. **Instruction-Response pairs**: Clear task with expected output
2. **Question-Answer pairs**: Specific questions with accurate answers
3. **Code examples**: Programming tasks with correct solutions
4. **Conversational data**: Natural dialogue patterns
5. **Domain-specific content**: Specialized knowledge areas

## Dataset Size Guidelines

### Ultra-Small Datasets (10-50 examples)
**Use Case**: Proof of concept, style transfer, specific formatting

**Strategy**:
- Focus on very specific tasks
- Use high learning rates (5e-4 to 1e-3)
- More training steps per example (20-40 steps per example)
- Lower LoRA rank (r=8-16)
- Careful validation to avoid overfitting

**Configuration Example**:
```json
{
  "model_name": "unsloth/gemma-2-2b-it-bnb-4bit",
  "max_steps": 500,
  "learning_rate": 5e-4,
  "r": 8,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "warmup_steps": 50
}
```

**Expected Results**:
- Model learns basic patterns and formatting
- Good for specific style or format adaptation
- May overfit to training examples
- Limited generalization capability

### Small Datasets (50-200 examples)
**Use Case**: Personal assistants, specific writing styles, simple task automation

**Strategy**:
- Ensure high diversity in examples
- Use moderate learning rates (2e-4 to 5e-4)
- 5-15 steps per example
- Standard LoRA rank (r=16)
- Regular evaluation to monitor overfitting

**Configuration Example**:
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_steps": 800,
  "learning_rate": 3e-4,
  "r": 16,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 80
}
```

**Expected Results**:
- Good performance on trained task types
- Some generalization within domain
- Maintains base model capabilities
- Suitable for many practical applications

### Medium Datasets (200-1000 examples)
**Use Case**: Customer service, content creation, educational tools

**Strategy**:
- Balance between different task types
- Standard learning rates (1e-4 to 3e-4)
- 3-8 steps per example
- Higher LoRA rank possible (r=16-32)
- Validation split recommended

**Configuration Example**:
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_steps": 1500,
  "learning_rate": 2e-4,
  "r": 24,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 150
}
```

**Expected Results**:
- Strong performance across task variations
- Good generalization within domain
- Retains most base model knowledge
- Production-ready for many use cases

### Large Datasets (1000+ examples)
**Use Case**: Production systems, multi-domain applications, research

**Strategy**:
- Comprehensive coverage of use cases
- Conservative learning rates (5e-5 to 2e-4)
- 1-5 steps per example
- Higher LoRA rank (r=32-64)
- Proper train/validation/test splits

**Configuration Example**:
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_steps": 3000,
  "learning_rate": 1e-4,
  "r": 32,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 2,
  "warmup_steps": 300
}
```

**Expected Results**:
- Excellent performance across all trained scenarios
- Strong generalization capabilities
- Maintains base model strengths
- Suitable for production deployment

## Data Quality Standards

### High-Quality Examples
```json
{
  "instruction": "Explain the process of photosynthesis in plants",
  "input": "",
  "output": "Photosynthesis is the biological process by which plants convert light energy, usually from the sun, into chemical energy stored in glucose. This process occurs in two main stages: the light-dependent reactions in the thylakoids and the light-independent reactions (Calvin cycle) in the stroma of chloroplasts. During photosynthesis, plants take in carbon dioxide from the air and water from the soil, and with the help of chlorophyll, they produce glucose and release oxygen as a byproduct. The overall equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂."
}
```

**Quality Characteristics**:
- Clear, specific instruction
- Accurate, comprehensive response
- Appropriate length and detail
- Proper grammar and formatting
- Educational value

### Poor-Quality Examples
```json
{
  "instruction": "plants",
  "input": "",
  "output": "make food from sun"
}
```

**Problems**:
- Vague instruction
- Incomplete response
- Lack of detail
- Poor grammar
- No educational value

### Data Quality Checklist
- [ ] Instructions are clear and specific
- [ ] Responses are accurate and complete
- [ ] Language is natural and fluent
- [ ] Examples cover diverse scenarios
- [ ] No contradictory information
- [ ] Appropriate response length
- [ ] Consistent formatting
- [ ] Free of harmful content

## Fine-tuning with Limited Data

### Strategies for Small Datasets

#### 1. Data Augmentation Techniques
```python
# Example: Create variations of existing examples
original = {
    "instruction": "Write a Python function to calculate factorial",
    "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
}

# Variations
variations = [
    {
        "instruction": "Create a Python function that computes the factorial of a number",
        "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
    },
    {
        "instruction": "Implement factorial calculation in Python",
        "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
    }
]
```

#### 2. Progressive Training Strategy
```bash
# Stage 1: Train on core examples (20-50 examples)
python main.py --data core_examples.json --config stage1_config.json

# Stage 2: Add variations and train more (100+ examples)
python main.py --data expanded_examples.json --config stage2_config.json

# Stage 3: Fine-tune with full dataset
python main.py --data complete_examples.json --config final_config.json
```

#### 3. Transfer Learning Approach
Start with a model already fine-tuned on similar tasks:
```json
{
  "model_name": "unsloth/gemma-2-9b-instruct-bnb-4bit",
  "max_steps": 200,
  "learning_rate": 1e-4,
  "r": 8
}
```

#### 4. Few-Shot Learning Enhancement
Include examples in your training data that demonstrate few-shot learning:
```json
{
  "instruction": "Learn from these examples and answer the question",
  "input": "Example 1: Q: What is 2+2? A: 4\nExample 2: Q: What is 3+3? A: 6\nQuestion: What is 4+4?",
  "output": "Following the pattern from the examples, 4+4 = 8"
}
```

### Configuration for Limited Data

#### Ultra-Conservative (Prevent Overfitting)
```json
{
  "model_name": "unsloth/gemma-2-2b-it-bnb-4bit",
  "max_steps": 100,
  "learning_rate": 1e-4,
  "r": 4,
  "lora_alpha": 8,
  "lora_dropout": 0.1,
  "warmup_steps": 20,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 4
}
```

#### Balanced Approach
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_steps": 300,
  "learning_rate": 2e-4,
  "r": 8,
  "lora_alpha": 16,
  "warmup_steps": 30,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 2
}
```

## Domain-Specific Requirements

### Code Generation
**Data Requirements**: 200-1000 examples
**Recommended Examples**:
- Various programming languages
- Different complexity levels
- Common programming patterns
- Error handling examples
- Documentation examples

**Sample Data Structure**:
```json
{
  "instruction": "Write a Python function to implement binary search",
  "input": "The function should take a sorted list and a target value",
  "output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1"
}
```

### Medical/Healthcare
**Data Requirements**: 1000-5000 examples
**Critical Considerations**:
- Accuracy is paramount
- Include disclaimers
- Cover common conditions
- Emergency vs non-emergency
- Professional language

**Configuration**:
```json
{
  "learning_rate": 5e-5,
  "max_steps": 2000,
  "r": 32,
  "warmup_steps": 200
}
```

### Creative Writing
**Data Requirements**: 500-2000 examples
**Recommended Examples**:
- Different genres and styles
- Various lengths
- Character development
- Plot structures
- Dialogue examples

### Customer Service
**Data Requirements**: 300-1500 examples
**Essential Categories**:
- Common questions
- Complaint handling
- Product information
- Escalation scenarios
- Positive interactions

### Educational Content
**Data Requirements**: 500-2500 examples
**Coverage Areas**:
- Different subjects
- Various difficulty levels
- Explanation styles
- Practice problems
- Assessment questions

## Data Preparation Best Practices

### Data Collection
1. **Diversify Sources**: Collect from multiple channels
2. **Quality Over Quantity**: Better to have fewer high-quality examples
3. **Real-World Scenarios**: Use actual use cases
4. **Edge Cases**: Include unusual but valid scenarios
5. **Balanced Distribution**: Ensure even coverage of different types

### Data Cleaning
```python
# Example data cleaning checklist
def clean_data(dataset):
    cleaned = []
    for example in dataset:
        # Remove duplicates
        if example not in cleaned:
            # Check length limits
            if 10 <= len(example['output']) <= 2000:
                # Validate format
                if validate_format(example):
                    # Clean text
                    example['output'] = clean_text(example['output'])
                    cleaned.append(example)
    return cleaned
```

### Data Validation
```json
{
  "validation_rules": {
    "instruction_min_length": 10,
    "instruction_max_length": 500,
    "output_min_length": 5,
    "output_max_length": 2000,
    "required_fields": ["instruction", "output"],
    "forbidden_content": ["harmful", "inappropriate"],
    "language_check": "english"
  }
}
```

### Data Splitting Strategy
```python
# Recommended splits by dataset size
dataset_splits = {
    "small (50-200)": {"train": 0.9, "val": 0.1},
    "medium (200-1000)": {"train": 0.8, "val": 0.2},
    "large (1000+)": {"train": 0.8, "val": 0.1, "test": 0.1}
}
```

## Configuration Strategies by Data Size

### 10-50 Examples (Proof of Concept)
```json
{
  "model_name": "unsloth/gemma-2-2b-it-bnb-4bit",
  "max_seq_length": 1024,
  "max_steps": 200,
  "learning_rate": 5e-4,
  "r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "warmup_steps": 20,
  "save_steps": 50,
  "eval_steps": 25,
  "logging_steps": 10
}
```

### 50-200 Examples (Small Project)
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_seq_length": 2048,
  "max_steps": 500,
  "learning_rate": 3e-4,
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 50,
  "save_steps": 100,
  "eval_steps": 50,
  "logging_steps": 25
}
```

### 200-1000 Examples (Production Ready)
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_seq_length": 2048,
  "max_steps": 1500,
  "learning_rate": 2e-4,
  "r": 24,
  "lora_alpha": 48,
  "lora_dropout": 0.1,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 150,
  "save_steps": 300,
  "eval_steps": 100,
  "logging_steps": 50
}
```

### 1000+ Examples (Enterprise)
```json
{
  "model_name": "unsloth/gemma-2-9b-it-bnb-4bit",
  "max_seq_length": 4096,
  "max_steps": 3000,
  "learning_rate": 1e-4,
  "r": 32,
  "lora_alpha": 64,
  "lora_dropout": 0.1,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 2,
  "warmup_steps": 300,
  "save_steps": 500,
  "eval_steps": 200,
  "logging_steps": 100
}
```

## Success Metrics and Evaluation

### Training Metrics to Monitor
1. **Training Loss**: Should decrease steadily
2. **Learning Rate**: Should follow schedule
3. **GPU Utilization**: Should be >80%
4. **Steps per Second**: Consistent performance
5. **Memory Usage**: Should be stable

### Quality Evaluation Methods
1. **Manual Review**: Check response quality
2. **Automated Testing**: Use test prompts
3. **Benchmark Comparison**: Compare to baseline
4. **User Feedback**: Real-world testing
5. **Domain Experts**: Specialist validation

### Warning Signs
- Loss plateaus early (underfitting)
- Loss becomes erratic (overfitting)
- Responses become repetitive
- Model forgets base capabilities
- Inconsistent quality across examples

This comprehensive guide provides everything needed to understand data requirements and successfully fine-tune models with any amount of data, from proof-of-concept to enterprise-scale deployments.
