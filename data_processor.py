import pandas as pd
import json
import os
from datasets import Dataset, DatasetDict
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Class to handle data loading and preprocessing for fine-tuning"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    
    def load_data_from_file(self, filepath: str, format_type: str = "auto") -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            filepath: Path to the data file
            format_type: Format of the file ('json', 'csv', 'jsonl', 'auto')
        
        Returns:
            pandas DataFrame with the loaded data
        """
        if format_type == "auto":
            format_type = self._detect_format(filepath)
        
        if format_type == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif format_type == "jsonl":
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            df = pd.DataFrame(data)
        elif format_type == "csv":
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Loaded {len(df)} samples from {filepath}")
        return df
    
    def _detect_format(self, filepath: str) -> str:
        """Auto-detect file format based on extension"""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".json":
            return "json"
        elif ext == ".jsonl":
            return "jsonl"
        elif ext == ".csv":
            return "csv"
        else:
            raise ValueError(f"Cannot auto-detect format for extension: {ext}")
    
    def create_instruction_dataset(self, 
                                 instructions: List[str], 
                                 inputs: List[str], 
                                 outputs: List[str]) -> Dataset:
        """
        Create a dataset in instruction format
        
        Args:
            instructions: List of instruction prompts
            inputs: List of input contexts (can be empty strings)
            outputs: List of expected outputs
        
        Returns:
            Hugging Face Dataset object
        """
        assert len(instructions) == len(inputs) == len(outputs), \
            "All lists must have the same length"
        
        formatted_texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            if input_text.strip():
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            formatted_texts.append(text)
        
        dataset = Dataset.from_dict({"text": formatted_texts})
        logger.info(f"Created instruction dataset with {len(dataset)} samples")
        return dataset
    
    def create_chat_dataset(self, conversations: List[List[Dict[str, str]]]) -> Dataset:
        """
        Create a dataset in chat format
        
        Args:
            conversations: List of conversations, where each conversation is a list of 
                         {"role": "user/assistant", "content": "message"} dictionaries
        
        Returns:
            Hugging Face Dataset object
        """
        formatted_texts = []
        for conversation in conversations:
            text = self._format_conversation(conversation)
            formatted_texts.append(text)
        
        dataset = Dataset.from_dict({"text": formatted_texts})
        logger.info(f"Created chat dataset with {len(dataset)} samples")
        return dataset
    
    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format a single conversation into text"""
        formatted_parts = []
        for message in conversation:
            role = message["role"]
            content = message["content"]
            if role == "user":
                formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        return "\n".join(formatted_parts)
    
    def load_alpaca_format(self, filepath: str) -> Dataset:
        """
        Load data in Alpaca format (instruction, input, output)
        
        Args:
            filepath: Path to the JSON file containing Alpaca-formatted data
        
        Returns:
            Hugging Face Dataset object
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        instructions = [item["instruction"] for item in data]
        inputs = [item.get("input", "") for item in data]
        outputs = [item["output"] for item in data]
        
        return self.create_instruction_dataset(instructions, inputs, outputs)
    
    def split_dataset(self, dataset: Dataset, 
                     train_ratio: float = 0.9,
                     seed: int = 42) -> DatasetDict:
        """
        Split dataset into train and validation sets
        
        Args:
            dataset: Input dataset
            train_ratio: Ratio of data to use for training
            seed: Random seed for reproducibility
        
        Returns:
            DatasetDict with 'train' and 'validation' splits
        """
        split_dataset = dataset.train_test_split(
            test_size=1-train_ratio, 
            seed=seed
        )
        
        dataset_dict = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })
        
        logger.info(f"Split dataset: {len(dataset_dict['train'])} train, "
                   f"{len(dataset_dict['validation'])} validation samples")
        
        return dataset_dict
    
    def create_sample_data(self, num_samples: int = 10) -> Dataset:
        """
        Create sample data for testing
        
        Args:
            num_samples: Number of sample entries to create
        
        Returns:
            Dataset with sample instruction-response pairs
        """
        sample_instructions = [
            "Explain the concept of machine learning in simple terms.",
            "Write a Python function to calculate the factorial of a number.",
            "Describe the benefits of renewable energy.",
            "What are the main differences between supervised and unsupervised learning?",
            "Explain how a neural network works.",
            "Write a short story about a robot learning to paint.",
            "What are the key principles of good software design?",
            "Describe the process of photosynthesis.",
            "How does blockchain technology work?",
            "Explain the concept of recursion in programming."
        ]
        
        sample_outputs = [
            "Machine learning is a type of artificial intelligence where computers learn patterns from data to make predictions or decisions without being explicitly programmed for each task.",
            "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)",
            "Renewable energy offers several benefits: it's environmentally friendly, reduces greenhouse gas emissions, provides energy security, creates jobs, and becomes more cost-effective over time.",
            "Supervised learning uses labeled data to train models for prediction tasks, while unsupervised learning finds patterns in unlabeled data without specific target outcomes.",
            "A neural network consists of interconnected nodes (neurons) organized in layers. It processes input data through weighted connections, applies activation functions, and adjusts weights through training to learn patterns.",
            "In a small art studio, robot R2-D2 discovered painting. Initially creating mere smudges, through patient practice and observation of human artists, R2-D2 eventually developed its own unique abstract style, proving that creativity knows no boundaries.",
            "Good software design follows principles like modularity, separation of concerns, DRY (Don't Repeat Yourself), SOLID principles, clear naming conventions, and maintaining clean, readable code.",
            "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll, providing energy for the plant and oxygen for other organisms.",
            "Blockchain is a distributed ledger technology that maintains a continuously growing list of records (blocks) linked and secured using cryptography, ensuring transparency and immutability without central authority.",
            "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem, typically having a base case to stop the recursion and prevent infinite loops."
        ]
        
        # Cycle through samples if more are requested
        instructions = (sample_instructions * ((num_samples // len(sample_instructions)) + 1))[:num_samples]
        outputs = (sample_outputs * ((num_samples // len(sample_outputs)) + 1))[:num_samples]
        inputs = [""] * num_samples  # Empty inputs for simplicity
        
        return self.create_instruction_dataset(instructions, inputs, outputs)
    
    def save_dataset(self, dataset: Union[Dataset, DatasetDict], output_dir: str):
        """
        Save dataset to disk
        
        Args:
            dataset: Dataset or DatasetDict to save
            output_dir: Directory to save the dataset
        """
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(output_dir)
        logger.info(f"Dataset saved to {output_dir}")
    
    def load_dataset_from_disk(self, dataset_path: str) -> Union[Dataset, DatasetDict]:
        """
        Load dataset from disk
        
        Args:
            dataset_path: Path to the saved dataset
        
        Returns:
            Loaded dataset
        """
        if os.path.exists(os.path.join(dataset_path, "dataset_dict.json")):
            dataset = DatasetDict.load_from_disk(dataset_path)
        else:
            dataset = Dataset.load_from_disk(dataset_path)
        
        logger.info(f"Dataset loaded from {dataset_path}")
        return dataset
