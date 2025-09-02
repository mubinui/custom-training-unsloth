"""
Environment-based configuration management for Unsloth custom training.
This module provides a robust configuration system that loads settings from:
1. Environment variables (highest priority)
2. .env file
3. Default values (lowest priority)
"""

import os
import torch
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

# Try to import python-dotenv, install if not available
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Environment variables will still work, but .env file will be ignored.")

logger = logging.getLogger(__name__)

def str_to_bool(value: Union[str, bool]) -> bool:
    """Convert string to boolean safely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)

def str_to_list(value: Union[str, List[str]]) -> List[str]:
    """Convert comma-separated string to list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    return []

def get_env_dtype(env_value: str) -> Optional[torch.dtype]:
    """Convert environment string to torch dtype."""
    if env_value.lower() == 'auto':
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    elif env_value.lower() == 'bfloat16':
        return torch.bfloat16
    elif env_value.lower() == 'float16':
        return torch.float16
    return None

@dataclass
class EnvConfig:
    """
    Environment-based configuration system for Unsloth training.
    
    This class automatically loads configuration from environment variables
    and .env files, with sensible defaults for all parameters.
    """
    
    # ==================== MODEL CONFIGURATION ====================
    model_name: str = field(default_factory=lambda: os.getenv('MODEL_NAME', 'unsloth/gemma-2-9b-it-bnb-4bit'))
    max_seq_length: int = field(default_factory=lambda: int(os.getenv('MAX_SEQ_LENGTH', '2048')))
    load_in_4bit: bool = field(default_factory=lambda: str_to_bool(os.getenv('LOAD_IN_4BIT', 'true')))
    dtype: Optional[torch.dtype] = field(default_factory=lambda: get_env_dtype(os.getenv('DTYPE', 'auto')))
    
    # ==================== LORA CONFIGURATION ====================
    lora_r: int = field(default_factory=lambda: int(os.getenv('LORA_R', '16')))
    lora_alpha: int = field(default_factory=lambda: int(os.getenv('LORA_ALPHA', '16')))
    lora_dropout: float = field(default_factory=lambda: float(os.getenv('LORA_DROPOUT', '0.0')))
    lora_target_modules: List[str] = field(default_factory=lambda: str_to_list(
        os.getenv('LORA_TARGET_MODULES', 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    ))
    lora_bias: str = field(default_factory=lambda: os.getenv('LORA_BIAS', 'none'))
    use_gradient_checkpointing: str = field(default_factory=lambda: os.getenv('USE_GRADIENT_CHECKPOINTING', 'unsloth'))
    random_state: int = field(default_factory=lambda: int(os.getenv('RANDOM_STATE', '3407')))
    use_rslora: bool = field(default_factory=lambda: str_to_bool(os.getenv('USE_RSLORA', 'false')))
    
    # ==================== TRAINING PARAMETERS ====================
    per_device_train_batch_size: int = field(default_factory=lambda: int(os.getenv('PER_DEVICE_TRAIN_BATCH_SIZE', '2')))
    gradient_accumulation_steps: int = field(default_factory=lambda: int(os.getenv('GRADIENT_ACCUMULATION_STEPS', '4')))
    warmup_steps: int = field(default_factory=lambda: int(os.getenv('WARMUP_STEPS', '5')))
    max_steps: int = field(default_factory=lambda: int(os.getenv('MAX_STEPS', '60')))
    learning_rate: float = field(default_factory=lambda: float(os.getenv('LEARNING_RATE', '2e-4')))
    fp16: bool = field(default_factory=lambda: get_fp16_setting())
    bf16: bool = field(default_factory=lambda: get_bf16_setting())
    logging_steps: int = field(default_factory=lambda: int(os.getenv('LOGGING_STEPS', '1')))
    optim: str = field(default_factory=lambda: os.getenv('OPTIM', 'adamw_8bit'))
    weight_decay: float = field(default_factory=lambda: float(os.getenv('WEIGHT_DECAY', '0.01')))
    lr_scheduler_type: str = field(default_factory=lambda: os.getenv('LR_SCHEDULER_TYPE', 'linear'))
    seed: int = field(default_factory=lambda: int(os.getenv('SEED', '3407')))
    
    # ==================== DATA CONFIGURATION ====================
    data_path: str = field(default_factory=lambda: os.getenv('DATA_PATH', 'sample_data.json'))
    dataset_text_field: str = field(default_factory=lambda: os.getenv('DATASET_TEXT_FIELD', 'text'))
    packing: bool = field(default_factory=lambda: str_to_bool(os.getenv('PACKING', 'false')))
    data_validation: str = field(default_factory=lambda: os.getenv('DATA_VALIDATION', 'strict'))
    
    # ==================== OUTPUT AND SAVING ====================
    output_dir: str = field(default_factory=lambda: os.getenv('OUTPUT_DIR', './outputs'))
    save_model: bool = field(default_factory=lambda: str_to_bool(os.getenv('SAVE_MODEL', 'true')))
    save_method: str = field(default_factory=lambda: os.getenv('SAVE_METHOD', 'merged_16bit'))
    new_model_name: str = field(default_factory=lambda: os.getenv('NEW_MODEL_NAME', 'gemma-2-9b-custom-finetune'))
    
    # ==================== HUGGING FACE INTEGRATION ====================
    push_to_hub: bool = field(default_factory=lambda: str_to_bool(os.getenv('PUSH_TO_HUB', 'false')))
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv('HF_TOKEN'))
    hf_username: Optional[str] = field(default_factory=lambda: os.getenv('HF_USERNAME'))
    hf_repo_name: Optional[str] = field(default_factory=lambda: os.getenv('HF_REPO_NAME'))
    
    # ==================== SYSTEM CONFIGURATION ====================
    num_workers: int = field(default_factory=lambda: int(os.getenv('NUM_WORKERS', '4')))
    cuda_device: int = field(default_factory=lambda: int(os.getenv('CUDA_DEVICE', '-1')))
    memory_optimization: str = field(default_factory=lambda: os.getenv('MEMORY_OPTIMIZATION', 'medium'))
    use_mixed_precision: bool = field(default_factory=lambda: str_to_bool(os.getenv('USE_MIXED_PRECISION', 'true')))
    cache_dir: str = field(default_factory=lambda: os.getenv('CACHE_DIR', './cache'))
    
    # ==================== LOGGING AND MONITORING ====================
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    verbose_logging: bool = field(default_factory=lambda: str_to_bool(os.getenv('VERBOSE_LOGGING', 'false')))
    log_file: Optional[str] = field(default_factory=lambda: os.getenv('LOG_FILE') or None)
    use_wandb: bool = field(default_factory=lambda: str_to_bool(os.getenv('USE_WANDB', 'false')))
    wandb_project: str = field(default_factory=lambda: os.getenv('WANDB_PROJECT', 'unsloth-custom-training'))
    wandb_entity: Optional[str] = field(default_factory=lambda: os.getenv('WANDB_ENTITY'))
    
    # ==================== EVALUATION CONFIGURATION ====================
    run_evaluation: bool = field(default_factory=lambda: str_to_bool(os.getenv('RUN_EVALUATION', 'true')))
    eval_batch_size: int = field(default_factory=lambda: int(os.getenv('EVAL_BATCH_SIZE', '1')))
    eval_samples: int = field(default_factory=lambda: int(os.getenv('EVAL_SAMPLES', '10')))
    eval_temperature: float = field(default_factory=lambda: float(os.getenv('EVAL_TEMPERATURE', '0.7')))
    eval_max_new_tokens: int = field(default_factory=lambda: int(os.getenv('EVAL_MAX_NEW_TOKENS', '256')))
    
    # ==================== ADVANCED CONFIGURATION ====================
    dataset_shuffle: bool = field(default_factory=lambda: str_to_bool(os.getenv('DATASET_SHUFFLE', 'true')))
    dataset_seed: int = field(default_factory=lambda: int(os.getenv('DATASET_SEED', '42')))
    resume_from_checkpoint: Optional[str] = field(default_factory=lambda: os.getenv('RESUME_FROM_CHECKPOINT'))
    early_stopping_patience: int = field(default_factory=lambda: int(os.getenv('EARLY_STOPPING_PATIENCE', '0')))
    validation_split: float = field(default_factory=lambda: float(os.getenv('VALIDATION_SPLIT', '0.1')))
    use_deepspeed: bool = field(default_factory=lambda: str_to_bool(os.getenv('USE_DEEPSPEED', 'false')))
    dataloader_num_workers: int = field(default_factory=lambda: int(os.getenv('DATALOADER_NUM_WORKERS', '0')))
    dataloader_pin_memory: bool = field(default_factory=lambda: str_to_bool(os.getenv('DATALOADER_PIN_MEMORY', 'true')))
    
    def __post_init__(self):
        """Post-initialization setup and validation."""
        # Set up logging
        self._setup_logging()
        
        # Validate configuration
        self._validate_config()
        
        # Create necessary directories
        self._create_directories()
        
        # Log configuration summary
        if self.verbose_logging:
            self._log_config_summary()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        # Configure logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set up handlers
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        
        # File handler if specified
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            handlers=handlers,
            force=True
        )
    
    def _validate_config(self):
        """Validate configuration settings."""
        # Check required files exist
        if not os.path.exists(self.data_path) and self.data_path != 'sample_data.json':
            logger.warning(f"Data file not found: {self.data_path}")
        
        # Validate numeric ranges
        if self.lora_r < 1:
            raise ValueError("LORA_R must be positive")
        if self.per_device_train_batch_size < 1:
            raise ValueError("PER_DEVICE_TRAIN_BATCH_SIZE must be positive")
        if self.learning_rate <= 0:
            raise ValueError("LEARNING_RATE must be positive")
        
        # Validate string choices
        valid_save_methods = ['merged_16bit', 'lora_only', 'gguf']
        if self.save_method not in valid_save_methods:
            raise ValueError(f"SAVE_METHOD must be one of: {valid_save_methods}")
        
        valid_memory_opts = ['low', 'medium', 'high']
        if self.memory_optimization not in valid_memory_opts:
            raise ValueError(f"MEMORY_OPTIMIZATION must be one of: {valid_memory_opts}")
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [self.output_dir, self.cache_dir]
        if self.log_file:
            directories.append(os.path.dirname(self.log_file))
        
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
    
    def _log_config_summary(self):
        """Log a summary of the current configuration."""
        logger.info("=== Configuration Summary ===")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Data: {self.data_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Batch size: {self.per_device_train_batch_size}")
        logger.info(f"Max steps: {self.max_steps}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"LoRA rank: {self.lora_r}")
        logger.info("============================")
    
    def to_training_config(self):
        """Convert to legacy TrainingConfig format for compatibility."""
        from config import TrainingConfig
        
        return TrainingConfig(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            r=self.lora_r,
            target_modules=self.lora_target_modules,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            random_state=self.random_state,
            use_rslora=self.use_rslora,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            fp16=self.fp16,
            bf16=self.bf16,
            logging_steps=self.logging_steps,
            optim=self.optim,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            seed=self.seed,
            output_dir=self.output_dir,
            dataset_text_field=self.dataset_text_field,
            packing=self.packing,
            save_model=self.save_model,
            save_method=self.save_method,
            push_to_hub=self.push_to_hub,
            hf_token=self.hf_token,
            new_model_name=self.new_model_name,
        )
    
    def save_to_env_file(self, filepath: str = ".env"):
        """Save current configuration to .env file."""
        env_content = []
        env_content.append("# Generated configuration file")
        env_content.append("# Edit this file to customize your training settings")
        env_content.append("")
        
        # Model configuration
        env_content.append("# Model Configuration")
        env_content.append(f"MODEL_NAME={self.model_name}")
        env_content.append(f"MAX_SEQ_LENGTH={self.max_seq_length}")
        env_content.append(f"LOAD_IN_4BIT={str(self.load_in_4bit).lower()}")
        env_content.append("")
        
        # Training parameters
        env_content.append("# Training Parameters")
        env_content.append(f"PER_DEVICE_TRAIN_BATCH_SIZE={self.per_device_train_batch_size}")
        env_content.append(f"GRADIENT_ACCUMULATION_STEPS={self.gradient_accumulation_steps}")
        env_content.append(f"MAX_STEPS={self.max_steps}")
        env_content.append(f"LEARNING_RATE={self.learning_rate}")
        env_content.append("")
        
        # LoRA configuration
        env_content.append("# LoRA Configuration")
        env_content.append(f"LORA_R={self.lora_r}")
        env_content.append(f"LORA_ALPHA={self.lora_alpha}")
        env_content.append(f"LORA_DROPOUT={self.lora_dropout}")
        env_content.append(f"LORA_TARGET_MODULES={','.join(self.lora_target_modules)}")
        env_content.append("")
        
        # Data and output
        env_content.append("# Data and Output")
        env_content.append(f"DATA_PATH={self.data_path}")
        env_content.append(f"OUTPUT_DIR={self.output_dir}")
        env_content.append(f"NEW_MODEL_NAME={self.new_model_name}")
        env_content.append("")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(env_content))
        
        logger.info(f"Configuration saved to {filepath}")

def get_fp16_setting() -> bool:
    """Get FP16 setting from environment with auto-detection."""
    env_value = os.getenv('FP16', 'auto').lower()
    if env_value == 'auto':
        return not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else True
    return str_to_bool(env_value)

def get_bf16_setting() -> bool:
    """Get BF16 setting from environment with auto-detection."""
    env_value = os.getenv('BF16', 'auto').lower()
    if env_value == 'auto':
        return torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    return str_to_bool(env_value)

def check_gpu_compatibility():
    """Check GPU compatibility and return system information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            info[f"gpu_{i}"] = {
                "name": device_props.name,
                "memory_gb": round(device_props.total_memory / 1024**3, 2),
                "compute_capability": f"{device_props.major}.{device_props.minor}"
            }
    
    return info

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about the specified model."""
    model_info = {
        "model_name": model_name,
        "supports_4bit": True,
        "recommended_max_seq_length": 2048,
    }
    
    if "gemma" in model_name.lower():
        model_info.update({
            "model_type": "Gemma",
            "architecture": "Transformer",
            "recommended_batch_size": 2,
            "recommended_learning_rate": 2e-4,
        })
    elif "llama" in model_name.lower():
        model_info.update({
            "model_type": "Llama",
            "architecture": "Transformer",
            "recommended_batch_size": 2,
            "recommended_learning_rate": 2e-4,
        })
    elif "mistral" in model_name.lower():
        model_info.update({
            "model_type": "Mistral",
            "architecture": "Transformer",
            "recommended_batch_size": 2,
            "recommended_learning_rate": 2e-4,
        })
    
    return model_info

# Convenience function to load configuration
def load_config() -> EnvConfig:
    """Load configuration from environment variables and .env file."""
    return EnvConfig()

# Backward compatibility
TrainingConfig = EnvConfig
