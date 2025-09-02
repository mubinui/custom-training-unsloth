import os
import torch
import logging
from datetime import datetime
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from config import TrainingConfig, check_gpu_compatibility
from data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnslothTrainer:
    """
    Main training class using Unsloth for efficient fine-tuning.
    
    This class handles the complete training pipeline including model loading,
    LoRA adapter setup, training execution, and model saving.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: TrainingConfig instance containing all training parameters
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.data_processor = DataProcessor()
        
        # Check system compatibility
        self.system_info = check_gpu_compatibility()
        logger.info(f"System info: {self.system_info}")
        
        if not self.system_info["cuda_available"]:
            logger.warning("CUDA not available. Training will be very slow on CPU.")
    
    def load_model(self):
        """Load the model and tokenizer using Unsloth"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=self.config.dtype,
                load_in_4bit=self.config.load_in_4bit,
            )
            
            logger.info("Model loaded successfully")
            
            # Add LoRA adapters
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.r,
                target_modules=self.config.target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias=self.config.bias,
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                random_state=self.config.random_state,
                use_rslora=self.config.use_rslora,
                loftq_config=self.config.loftq_config,
            )
            
            logger.info("LoRA adapters added successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def setup_chat_template(self):
        """Setup chat template for the tokenizer"""
        try:
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="chatml"  # You can change this to other templates like "alpaca", "vicuna", etc.
            )
            logger.info("Chat template configured")
        except Exception as e:
            logger.warning(f"Could not set chat template: {str(e)}")
    
    def prepare_training_args(self) -> TrainingArguments:
        """Prepare training arguments"""
        return TrainingArguments(
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            seed=self.config.seed,
            output_dir=self.config.output_dir,
            report_to=None,  # Disable wandb by default
            save_strategy="steps",
            save_steps=self.config.max_steps // 4,  # Save 4 times during training
            evaluation_strategy="no",  # Can be changed to "steps" if validation data is provided
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
    
    def train(self, dataset, validation_dataset=None):
        """
        Start the training process
        
        Args:
            dataset: Training dataset
            validation_dataset: Optional validation dataset
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Starting training...")
        
        # Prepare training arguments
        training_args = self.prepare_training_args()
        
        # Setup trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            eval_dataset=validation_dataset,
            dataset_text_field=self.config.dataset_text_field,
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=2,
            packing=self.config.packing,
            args=training_args,
        )
        
        # Start training
        try:
            start_time = datetime.now()
            logger.info(f"Training started at: {start_time}")
            
            self.trainer.train()
            
            end_time = datetime.now()
            training_duration = end_time - start_time
            logger.info(f"Training completed at: {end_time}")
            logger.info(f"Training duration: {training_duration}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def save_model(self, save_path: str = None):
        """
        Save the trained model
        
        Args:
            save_path: Path to save the model. If None, uses config.output_dir
        """
        if save_path is None:
            save_path = self.config.output_dir
        
        logger.info(f"Saving model using method: {self.config.save_method}")
        
        if self.config.save_method == "lora":
            # Save only LoRA adapters
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"LoRA adapters saved to {save_path}")
            
        elif self.config.save_method == "merged_16bit":
            # Save merged model in 16-bit
            self.model.save_pretrained_merged(save_path, self.tokenizer, save_method="merged_16bit")
            logger.info(f"Merged 16-bit model saved to {save_path}")
            
        elif self.config.save_method == "merged_4bit":
            # Save merged model in 4-bit
            self.model.save_pretrained_merged(save_path, self.tokenizer, save_method="merged_4bit")
            logger.info(f"Merged 4-bit model saved to {save_path}")
    
    def push_to_hub(self):
        """Push the model to Hugging Face Hub"""
        if not self.config.push_to_hub:
            logger.info("Push to hub is disabled in config")
            return
        
        if not self.config.hf_token:
            logger.warning("Hugging Face token not provided. Cannot push to hub.")
            return
        
        try:
            if self.config.save_method == "lora":
                self.model.push_to_hub_merged(
                    self.config.new_model_name,
                    self.tokenizer,
                    save_method="lora",
                    token=self.config.hf_token
                )
            else:
                self.model.push_to_hub_merged(
                    self.config.new_model_name,
                    self.tokenizer,
                    save_method=self.config.save_method,
                    token=self.config.hf_token
                )
            logger.info(f"Model pushed to hub: {self.config.new_model_name}")
        except Exception as e:
            logger.error(f"Error pushing to hub: {str(e)}")
            raise
    
    def evaluate_model(self, test_texts, max_new_tokens=256):
        """
        Evaluate the model on test texts
        
        Args:
            test_texts: List of input texts for evaluation
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            List of generated responses
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")
        
        logger.info("Evaluating model...")
        responses = []
        
        FastLanguageModel.for_inference(self.model)  # Enable inference mode
        
        for text in test_texts:
            inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        
        logger.info(f"Generated {len(responses)} responses")
        return responses
    
    def get_training_stats(self):
        """Get training statistics"""
        if self.trainer is None:
            return None
        
        return {
            "train_loss": self.trainer.state.log_history,
            "total_steps": self.trainer.state.global_step,
            "epochs": self.trainer.state.epoch,
        }
