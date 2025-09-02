import torch
import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.
    
    This class contains all the hyperparameters and settings needed for training
    a language model using Unsloth with LoRA adapters.
    """
    
    # Model configuration
    model_name: str = "unsloth/gemma-2-9b-it-bnb-4bit"
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = True
    
    # LoRA configuration
    r: int = 16
    target_modules: Optional[List[str]] = None
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Dict] = None
    
    # Training parameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    fp16: bool = not torch.cuda.is_bf16_supported()
    bf16: bool = torch.cuda.is_bf16_supported()
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "./outputs"
    
    # Data configuration
    dataset_text_field: str = "text"
    packing: bool = False
    
    # Save and push configuration
    save_model: bool = True
    save_method: str = "merged_16bit"
    push_to_hub: bool = False
    hf_token: Optional[str] = None
    new_model_name: str = "gemma-2-9b-custom-finetune"
    
    def __post_init__(self):
        """
        Post-initialization method to set default values and validate configuration.
        
        Sets default target modules for LoRA if not provided and configures
        dtype based on GPU capabilities.
        """
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
        
        # Set dtype based on GPU capabilities
        if self.dtype is None:
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Dictionary representation of the configuration
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.dtype):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    def save_config(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path where to save the configuration file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'TrainingConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            TrainingConfig instance loaded from file
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert dtype string back to torch.dtype
        if 'dtype' in config_dict and isinstance(config_dict['dtype'], str):
            if 'bfloat16' in config_dict['dtype']:
                config_dict['dtype'] = torch.bfloat16
            elif 'float16' in config_dict['dtype']:
                config_dict['dtype'] = torch.float16
            else:
                config_dict['dtype'] = None
        
        return cls(**config_dict)

def check_gpu_compatibility():
    """
    Check GPU compatibility and return system information.
    
    Returns:
        Dictionary containing GPU and CUDA information
    """
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
    """
    Get information about the specified model.
    
    Args:
        model_name: Name of the model to get information for
        
    Returns:
        Dictionary containing model information and recommended settings
    """
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
    
    return model_info
