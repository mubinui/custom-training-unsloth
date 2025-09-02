#!/usr/bin/env python3
"""
Evaluation script for testing the trained model
"""

import argparse
import sys
from pathlib import Path
import logging

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from config import TrainingConfig
from trainer import UnslothTrainer

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Unsloth Fine-tuned Model")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the trained model directory"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--prompts", 
        type=str,
        help="File containing evaluation prompts (one per line)"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Interactive evaluation mode"
    )
    
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=256,
        help="Maximum tokens to generate"
    )
    
    return parser.parse_args()

def load_prompts_from_file(filepath: str):
    """Load evaluation prompts from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def get_default_prompts():
    """Get default evaluation prompts"""
    return [
        "### Instruction:\nExplain what machine learning is\n\n### Response:\n",
        "### Instruction:\nWrite a Python function to check if a number is prime\n\n### Response:\n",
        "### Instruction:\nWhat are the benefits of renewable energy?\n\n### Response:\n",
        "### Instruction:\nDescribe how photosynthesis works\n\n### Response:\n",
        "### Instruction:\nExplain the concept of recursion in programming\n\n### Response:\n"
    ]

def interactive_evaluation(trainer: UnslothTrainer, max_tokens: int):
    """Interactive evaluation mode"""
    print("\n=== Interactive Evaluation Mode ===")
    print("Type your prompts below. Type 'quit' to exit.")
    print("Use ### Instruction: and ### Response: format for best results.\n")
    
    while True:
        try:
            prompt = input("Enter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
                
            if not prompt:
                continue
            
            # Format prompt if not already formatted
            if not prompt.startswith("###"):
                prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
            
            print(f"\nGenerating response...")
            responses = trainer.evaluate_model([prompt], max_new_tokens=max_tokens)
            
            print(f"\n{'='*50}")
            print(f"Response:\n{responses[0]}")
            print(f"{'='*50}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    try:
        config = TrainingConfig.load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Configuration file {args.config} not found")
        sys.exit(1)
    
    # Initialize trainer
    trainer = UnslothTrainer(config)
    
    try:
        # Load the trained model
        logger.info(f"Loading model from {args.model_path}")
        
        # Update config to load from the trained model path
        config.model_name = args.model_path
        trainer.config = config
        
        trainer.load_model()
        trainer.setup_chat_template()
        
        logger.info("Model loaded successfully")
        
        if args.interactive:
            # Interactive mode
            interactive_evaluation(trainer, args.max_tokens)
        else:
            # Batch evaluation
            if args.prompts:
                prompts = load_prompts_from_file(args.prompts)
            else:
                prompts = get_default_prompts()
            
            logger.info(f"Evaluating {len(prompts)} prompts...")
            
            responses = trainer.evaluate_model(prompts, max_new_tokens=args.max_tokens)
            
            # Display results
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                print(f"\n{'='*60}")
                print(f"Evaluation {i+1}/{len(prompts)}")
                print(f"{'='*60}")
                print(f"Prompt:\n{prompt}")
                print(f"\nResponse:\n{response}")
                print(f"{'='*60}")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
