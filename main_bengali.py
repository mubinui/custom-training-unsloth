#!/usr/bin/env python3
"""
Bengali Model Training Script
Specialized training script for Bengali language models using the bengali_chat_conv dataset
Usage: python main_bengali.py [--sample] [--eval] [--push] [--max-examples N]
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import with error handling
try:
    from env_config import load_config, check_gpu_compatibility, get_model_info
    from bengali_data_processor import BengaliDataProcessor
    from trainer import UnslothTrainer
except ImportError as e:
    logging.error(f"Import error: {e}")
    logging.error("Please ensure all dependencies are installed:")
    logging.error("pip install -r requirements.txt")
    sys.exit(1)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments for Bengali training"""
    parser = argparse.ArgumentParser(description="Unsloth Bengali Training System")
    
    parser.add_argument(
        "--sample", 
        action="store_true",
        help="Use sample Bengali data for testing"
    )
    
    parser.add_argument(
        "--eval", 
        action="store_true",
        help="Run evaluation after training with Bengali prompts"
    )
    
    parser.add_argument(
        "--push", 
        action="store_true",
        help="Push model to Hugging Face Hub after training"
    )
    
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Maximum number of examples to use from the dataset"
    )
    
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use (train, test, validation)"
    )
    
    parser.add_argument(
        "--create-env", 
        action="store_true",
        help="Create a new .env file with Bengali settings"
    )
    
    return parser.parse_args()

def load_bengali_environment():
    """Load Bengali-specific environment configuration"""
    # First load the Bengali template if .env doesn't exist
    if not os.path.exists('.env') and os.path.exists('.env.bengali'):
        logger.info("Using Bengali configuration template")
        import shutil
        shutil.copy('.env.bengali', '.env')
    
    return load_config()

def prepare_bengali_data(args, bengali_processor: BengaliDataProcessor, config):
    """Prepare Bengali training data"""
    
    if args.sample:
        logger.info("Using sample Bengali data for testing")
        sample_data = bengali_processor.create_sample_bengali_data(num_samples=50)
        
        # Convert to dataset format
        try:
            from datasets import Dataset
            dataset = Dataset.from_list(sample_data)
            
            # Split the dataset
            split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
            return {
                'train': split_dataset['train'],
                'validation': split_dataset['test']
            }
        except ImportError:
            logger.error("datasets library not available")
            return None
    
    else:
        logger.info("Loading Bengali chat conversation dataset from Hugging Face")
        try:
            # Load the full dataset
            dataset = bengali_processor.load_bengali_dataset(
                split=args.dataset_split,
                streaming=False
            )
            
            # Limit examples if specified
            if args.max_examples and args.max_examples > 0:
                dataset = dataset.select(range(min(args.max_examples, len(dataset))))
                logger.info(f"Limited dataset to {len(dataset)} examples")
            
            # Convert to Alpaca format
            logger.info("Converting Bengali dataset to Alpaca format...")
            alpaca_data = bengali_processor.convert_to_alpaca_format(dataset)
            
            if not alpaca_data:
                logger.error("No data converted to Alpaca format")
                return None
            
            # Create dataset from converted data
            from datasets import Dataset
            alpaca_dataset = Dataset.from_list(alpaca_data)
            
            # Split the dataset
            split_dataset = alpaca_dataset.train_test_split(
                test_size=config.validation_split, 
                seed=config.dataset_seed
            )
            
            logger.info(f"Training samples: {len(split_dataset['train'])}")
            logger.info(f"Validation samples: {len(split_dataset['test'])}")
            
            return {
                'train': split_dataset['train'],
                'validation': split_dataset['test']
            }
            
        except Exception as e:
            logger.error(f"Failed to load Bengali dataset: {e}")
            logger.info("Falling back to sample data...")
            return prepare_bengali_data(
                argparse.Namespace(sample=True), 
                bengali_processor, 
                config
            )

def setup_bengali_chat_template(trainer):
    """Setup Bengali-specific chat template"""
    bengali_template = """<|im_start|>system
আপনি একজন সহায়ক AI সহায়ক। আপনি বাংলায় উত্তর দেন।<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
    
    try:
        if hasattr(trainer, 'tokenizer') and trainer.tokenizer:
            trainer.tokenizer.chat_template = bengali_template
            logger.info("Set Bengali chat template")
    except Exception as e:
        logger.warning(f"Could not set Bengali chat template: {e}")

def create_bengali_evaluation_prompts():
    """Create Bengali-specific evaluation prompts"""
    return [
        "### নির্দেশ:\nআপনার নাম কি?\n\n### উত্তর:\n",
        "### নির্দেশ:\nমেশিন লার্নিং সম্পর্কে ব্যাখ্যা করুন\n\n### উত্তর:\n",
        "### নির্দেশ:\nপাইথনে একটি সিম্পল ফাংশন লিখুন\n\n### উত্তর:\n",
        "### নির্দেশ:\nবাংলাদেশ সম্পর্কে কিছু বলুন\n\n### উত্তর:\n",
        "### নির্দেশ:\nকৃত্রিম বুদ্ধিমত্তার ভবিষ্যৎ নিয়ে আপনার মতামত কি?\n\n### উত্তর:\n"
    ]

def main():
    """Main Bengali training function"""
    args = parse_arguments()
    
    logger.info("=== Unsloth Bengali Training System ===")
    
    # Load Bengali configuration
    config = load_bengali_environment()
    
    # Handle --create-env argument
    if args.create_env:
        config.save_to_env_file(".env")
        logger.info("Created .env file with Bengali configuration")
        return
    
    # Override config with command line arguments
    if args.eval:
        config.run_evaluation = True
    if args.push:
        config.push_to_hub = True
    
    # Check system compatibility
    system_info = check_gpu_compatibility()
    logger.info(f"System Information: {system_info}")
    
    if not system_info["cuda_available"]:
        logger.error("CUDA is not available. This script requires a GPU with CUDA support.")
        sys.exit(1)
    
    # Get model information
    model_info = get_model_info(config.model_name)
    logger.info(f"Model Information: {model_info}")
    
    try:
        # Convert to legacy config format for trainer compatibility
        training_config = config.to_training_config()
        
        # Initialize Bengali data processor and trainer
        bengali_processor = BengaliDataProcessor()
        trainer = UnslothTrainer(training_config)
        
        # Load model
        logger.info("Loading model for Bengali training...")
        trainer.load_model()
        
        # Setup Bengali chat template
        setup_bengali_chat_template(trainer)
        
        # Prepare Bengali data
        logger.info("Preparing Bengali data...")
        dataset_dict = prepare_bengali_data(args, bengali_processor, config)
        
        if not dataset_dict:
            logger.error("Failed to prepare Bengali data")
            sys.exit(1)
        
        logger.info(f"Training samples: {len(dataset_dict['train'])}")
        logger.info(f"Validation samples: {len(dataset_dict['validation'])}")
        
        # Log some sample data
        if config.verbose_logging:
            logger.info("Sample training data:")
            for i in range(min(3, len(dataset_dict['train']))):
                example = dataset_dict['train'][i]
                logger.info(f"Sample {i+1}: {example}")
        
        # Start training
        logger.info("Starting Bengali model training...")
        trainer.train(
            dataset=dataset_dict['train'],
            validation_dataset=dataset_dict['validation']
        )
        
        # Get training statistics
        stats = trainer.get_training_stats()
        if stats:
            logger.info(f"Training completed. Final stats: {stats}")
        
        # Save model
        if config.save_model:
            logger.info("Saving Bengali model...")
            trainer.save_model()
        
        # Push to hub if requested
        if config.push_to_hub:
            logger.info("Pushing Bengali model to Hugging Face Hub...")
            trainer.push_to_hub()
        
        # Evaluation with Bengali prompts
        if config.run_evaluation:
            logger.info("Running evaluation with Bengali prompts...")
            bengali_prompts = create_bengali_evaluation_prompts()
            
            responses = trainer.evaluate_model(
                bengali_prompts, 
                max_new_tokens=config.eval_max_new_tokens
            )
            
            for i, (prompt, response) in enumerate(zip(bengali_prompts, responses)):
                logger.info(f"=== Bengali Evaluation {i+1} ===")
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Response: {response}")
                logger.info("=" * 50)
        
        logger.info("Bengali training pipeline completed successfully!")
        
        # Save training summary
        summary = {
            "model_name": config.model_name,
            "dataset": "spedrox-sac/bengali_chat_conv",
            "training_samples": len(dataset_dict['train']),
            "validation_samples": len(dataset_dict['validation']),
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "lora_r": config.lora_r,
            "output_dir": config.output_dir,
            "model_saved": config.save_model,
            "pushed_to_hub": config.push_to_hub
        }
        
        with open(os.path.join(config.output_dir, "bengali_training_summary.json"), 'w', encoding='utf-8') as f:
            import json
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Training summary saved to {config.output_dir}/bengali_training_summary.json")
        
    except Exception as e:
        logger.error(f"Bengali training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
