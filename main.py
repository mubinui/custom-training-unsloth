#!/usr/bin/env python3
"""
Main training script for Unsloth-based custom fine-tuning
Usage: python main.py [--data data.json] [--sample]
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from env_config import load_config, check_gpu_compatibility, get_model_info
from data_processor import DataProcessor
from trainer import UnslothTrainer

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Unsloth Custom Training System")
    
    parser.add_argument(
        "--data", 
        type=str, 
        help="Path to training data file (overrides DATA_PATH env var)"
    )
    
    parser.add_argument(
        "--sample", 
        action="store_true",
        help="Use sample data for testing"
    )
    
    parser.add_argument(
        "--eval", 
        action="store_true",
        help="Run evaluation after training (overrides RUN_EVALUATION env var)"
    )
    
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Skip saving the model"
    )
    
    parser.add_argument(
        "--push", 
        action="store_true",
        help="Push model to Hugging Face Hub after training (overrides PUSH_TO_HUB env var)"
    )
    
    parser.add_argument(
        "--create-env", 
        action="store_true",
        help="Create a new .env file with current settings"
    )
    
    return parser.parse_args()

def prepare_data(args, data_processor: DataProcessor, config):
    """Prepare training data"""
    data_path = args.data if args.data else config.data_path
    
    if args.sample:
        logger.info("Using sample data for testing")
        dataset = data_processor.create_sample_data(num_samples=50)
        return data_processor.split_dataset(dataset, train_ratio=0.8)
    
    elif data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        
        # Check if it's Alpaca format
        if data_path.endswith('.json'):
            try:
                dataset = data_processor.load_alpaca_format(data_path)
                return data_processor.split_dataset(dataset, train_ratio=0.9)
            except Exception as e:
                logger.warning(f"Could not load as Alpaca format: {e}")
                # Fall back to generic loading
                df = data_processor.load_data_from_file(data_path)
                # Assume it has 'text' column or convert it
                if 'text' not in df.columns:
                    if 'instruction' in df.columns and 'output' in df.columns:
                        instructions = df['instruction'].tolist()
                        inputs = df.get('input', [""]*len(df)).tolist() if 'input' in df.columns else [""]*len(df)
                        outputs = df['output'].tolist()
                        dataset = data_processor.create_instruction_dataset(instructions, inputs, outputs)
                        return data_processor.split_dataset(dataset)
                    else:
                        raise ValueError("Data format not recognized. Please provide 'text' column or Alpaca format.")
                else:
                    from datasets import Dataset
                    dataset = Dataset.from_pandas(df)
                    return data_processor.split_dataset(dataset)
        else:
            df = data_processor.load_data_from_file(args.data)
            from datasets import Dataset
            dataset = Dataset.from_pandas(df)
            return data_processor.split_dataset(dataset)
    
    else:
        raise ValueError("Either --data or --sample must be specified")

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("=== Unsloth Custom Training System ===")
    
    # Load configuration
    config = load_or_create_config(args.config)
    
    # Override config with command line arguments
    if args.no_save:
        config.save_model = False
    if args.push:
        config.push_to_hub = True
    
    # Setup directories
    setup_directories(config)
    
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
        # Initialize data processor and trainer
        data_processor = DataProcessor()
        trainer = UnslothTrainer(config)
        
        # Load model
        logger.info("Loading model...")
        trainer.load_model()
        trainer.setup_chat_template()
        
        # Prepare data
        logger.info("Preparing data...")
        dataset_dict = prepare_data(args, data_processor)
        
        logger.info(f"Training samples: {len(dataset_dict['train'])}")
        logger.info(f"Validation samples: {len(dataset_dict['validation'])}")
        
        # Save dataset for reference
        data_processor.save_dataset(dataset_dict, "data/processed_dataset")
        
        # Start training
        logger.info("Starting training...")
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
            logger.info("Saving model...")
            trainer.save_model()
        
        # Push to hub if requested
        if config.push_to_hub:
            logger.info("Pushing to Hugging Face Hub...")
            trainer.push_to_hub()
        
        # Evaluation
        if args.eval:
            logger.info("Running evaluation...")
            test_prompts = [
                "### Instruction:\nExplain what machine learning is\n\n### Response:\n",
                "### Instruction:\nWrite a Python function to reverse a string\n\n### Response:\n",
                "### Instruction:\nWhat are the benefits of using renewable energy?\n\n### Response:\n"
            ]
            
            responses = trainer.evaluate_model(test_prompts, max_new_tokens=256)
            
            for i, (prompt, response) in enumerate(zip(test_prompts, responses)):
                logger.info(f"=== Evaluation {i+1} ===")
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Response: {response}")
                logger.info("=" * 50)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
