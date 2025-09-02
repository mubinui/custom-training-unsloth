"""
Bengali Dataset Processor
Specialized data processor for Bengali chat conversation datasets
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class BengaliDataProcessor:
    """
    Specialized data processor for Bengali chat conversation datasets.
    Handles the spedrox-sac/bengali_chat_conv dataset format.
    """
    
    def __init__(self):
        self.dataset_name = "spedrox-sac/bengali_chat_conv"
        self.supported_formats = ['huggingface', 'json', 'jsonl']
    
    def load_bengali_dataset(self, split: str = 'train', streaming: bool = False):
        """
        Load the Bengali chat conversation dataset from Hugging Face.
        
        Args:
            split: Dataset split to load ('train', 'test', 'validation')
            streaming: Whether to use streaming mode for large datasets
            
        Returns:
            Loaded dataset
        """
        try:
            from datasets import load_dataset
            
            logger.info(f"Loading Bengali dataset: {self.dataset_name}")
            ds = load_dataset(self.dataset_name, split=split, streaming=streaming)
            
            logger.info(f"Loaded {len(ds)} examples from {split} split")
            return ds
            
        except ImportError:
            logger.error("datasets library not available. Install with: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Failed to load Bengali dataset: {e}")
            raise
    
    def explore_dataset_structure(self):
        """
        Explore and log the structure of the Bengali dataset.
        
        Returns:
            Dictionary with dataset information
        """
        try:
            from datasets import load_dataset
            
            ds = load_dataset(self.dataset_name)
            
            info = {
                "dataset_name": self.dataset_name,
                "splits": list(ds.keys()),
                "total_examples": sum(len(ds[split]) for split in ds.keys()),
            }
            
            if 'train' in ds:
                train_ds = ds['train']
                info.update({
                    "train_examples": len(train_ds),
                    "columns": train_ds.column_names,
                    "features": str(train_ds.features)
                })
                
                # Sample a few examples to understand the format
                sample_examples = []
                for i in range(min(3, len(train_ds))):
                    sample_examples.append(train_ds[i])
                info["sample_examples"] = sample_examples
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to explore dataset: {e}")
            return {"error": str(e)}
    
    def convert_to_alpaca_format(self, dataset, max_examples: Optional[int] = None):
        """
        Convert Bengali chat conversations to Alpaca format.
        
        Args:
            dataset: The loaded Bengali dataset
            max_examples: Maximum number of examples to convert
            
        Returns:
            List of dictionaries in Alpaca format
        """
        alpaca_data = []
        
        try:
            examples_to_process = min(len(dataset), max_examples) if max_examples else len(dataset)
            
            for i in range(examples_to_process):
                example = dataset[i]
                
                # Extract the conversation
                # Assuming the dataset has 'conversations' or 'messages' field
                if 'conversations' in example:
                    conversations = example['conversations']
                elif 'messages' in example:
                    conversations = example['messages']
                elif 'text' in example:
                    # If it's already in text format, try to parse it
                    text = example['text']
                    alpaca_data.append({
                        "instruction": "বাংলায় উত্তর দিন।",  # "Answer in Bengali"
                        "input": "",
                        "output": text
                    })
                    continue
                else:
                    logger.warning(f"Unknown format in example {i}: {list(example.keys())}")
                    continue
                
                # Convert conversation to instruction-response format
                if isinstance(conversations, list) and len(conversations) >= 2:
                    # Take pairs of human-assistant messages
                    for j in range(0, len(conversations) - 1, 2):
                        if j + 1 < len(conversations):
                            human_msg = conversations[j]
                            assistant_msg = conversations[j + 1]
                            
                            # Extract content based on message format
                            instruction = self._extract_message_content(human_msg)
                            response = self._extract_message_content(assistant_msg)
                            
                            if instruction and response:
                                alpaca_data.append({
                                    "instruction": instruction,
                                    "input": "",
                                    "output": response
                                })
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1} examples...")
            
            logger.info(f"Converted {len(alpaca_data)} conversation pairs to Alpaca format")
            return alpaca_data
            
        except Exception as e:
            logger.error(f"Error converting to Alpaca format: {e}")
            raise
    
    def _extract_message_content(self, message):
        """
        Extract content from a message object.
        
        Args:
            message: Message object from the dataset
            
        Returns:
            String content of the message
        """
        if isinstance(message, str):
            return message
        elif isinstance(message, dict):
            # Try common field names
            for field in ['content', 'text', 'message', 'value']:
                if field in message:
                    return message[field]
            # If no common field, return string representation
            return str(message)
        else:
            return str(message)
    
    def create_bengali_training_data(self, output_file: str = "bengali_training_data.json", 
                                   max_examples: Optional[int] = None):
        """
        Create Bengali training data in Alpaca format and save to file.
        
        Args:
            output_file: Path to save the training data
            max_examples: Maximum number of examples to process
        """
        try:
            # Load the dataset
            dataset = self.load_bengali_dataset(split='train')
            
            # Convert to Alpaca format
            alpaca_data = self.convert_to_alpaca_format(dataset, max_examples)
            
            # Save to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(alpaca_data)} examples to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to create Bengali training data: {e}")
            raise
    
    def create_sample_bengali_data(self, num_samples: int = 50) -> List[Dict[str, str]]:
        """
        Create sample Bengali training data for testing.
        
        Args:
            num_samples: Number of sample conversations to create
            
        Returns:
            List of sample conversations in Alpaca format
        """
        sample_data = []
        
        # Bengali instruction-response pairs for testing
        bengali_samples = [
            {
                "instruction": "আপনার নাম কি?",
                "input": "",
                "output": "আমি একটি কৃত্রিম বুদ্ধিমত্তা সহায়ক। আমি বিভিন্ন প্রশ্নের উত্তর দিতে এবং সাহায্য করতে পারি।"
            },
            {
                "instruction": "মেশিন লার্নিং কি?",
                "input": "",
                "output": "মেশিন লার্নিং হলো কৃত্রিম বুদ্ধিমত্তার একটি শাখা যেখানে কম্পিউটার ডেটা থেকে স্বয়ংক্রিয়ভাবে শিখতে এবং উন্নতি করতে পারে।"
            },
            {
                "instruction": "পাইথন প্রোগ্রামিং সম্পর্কে বলুন",
                "input": "",
                "output": "পাইথন একটি উচ্চ-স্তরের প্রোগ্রামিং ভাষা যা সহজ সিনট্যাক্স এবং শক্তিশালী লাইব্রেরির জন্য পরিচিত। এটি ডেটা সায়েন্স, ওয়েব ডেভেলপমেন্ট এবং AI-তে ব্যাপকভাবে ব্যবহৃত হয়।"
            },
            {
                "instruction": "বাংলাদেশের রাজধানী কি?",
                "input": "",
                "output": "বাংলাদেশের রাজধানী ঢাকা। এটি দেশের বৃহত্তম শহর এবং অর্থনৈতিক কেন্দ্র।"
            },
            {
                "instruction": "একটি গল্প বলুন",
                "input": "ছোট একটি গল্প",
                "output": "একবার এক ছোট মেয়ে ছিল যার নাম রিমা। সে প্রতিদিন বই পড়তে ভালোবাসত। একদিন সে একটি জাদুর বই পেল যা তাকে অনেক কিছু শেখাল।"
            }
        ]
        
        # Repeat and vary the samples to reach the desired number
        for i in range(num_samples):
            base_sample = bengali_samples[i % len(bengali_samples)]
            sample_data.append(base_sample.copy())
        
        logger.info(f"Created {len(sample_data)} sample Bengali conversations")
        return sample_data
    
    def validate_bengali_text(self, text: str) -> bool:
        """
        Validate if text contains Bengali characters.
        
        Args:
            text: Text to validate
            
        Returns:
            True if text contains Bengali characters
        """
        if not text:
            return False
        
        # Bengali Unicode range: U+0980 to U+09FF
        bengali_chars = any('\u0980' <= char <= '\u09FF' for char in text)
        return bengali_chars
    
    def get_dataset_stats(self):
        """
        Get statistics about the Bengali dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        try:
            dataset = self.load_bengali_dataset(split='train')
            
            stats = {
                "total_examples": len(dataset),
                "avg_length": 0,
                "bengali_ratio": 0,
                "sample_texts": []
            }
            
            total_length = 0
            bengali_count = 0
            
            # Sample up to 1000 examples for statistics
            sample_size = min(1000, len(dataset))
            
            for i in range(sample_size):
                example = dataset[i]
                
                # Get text content (adapt based on actual dataset structure)
                text_content = ""
                if 'text' in example:
                    text_content = example['text']
                elif 'conversations' in example:
                    # Concatenate conversation content
                    text_content = " ".join([
                        self._extract_message_content(msg) 
                        for msg in example['conversations']
                    ])
                
                if text_content:
                    total_length += len(text_content)
                    if self.validate_bengali_text(text_content):
                        bengali_count += 1
                    
                    # Store first few examples
                    if len(stats["sample_texts"]) < 5:
                        stats["sample_texts"].append(text_content[:200] + "...")
            
            stats["avg_length"] = total_length / sample_size if sample_size > 0 else 0
            stats["bengali_ratio"] = bengali_count / sample_size if sample_size > 0 else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting dataset stats: {e}")
            return {"error": str(e)}

def main():
    """Main function to test the Bengali data processor"""
    processor = BengaliDataProcessor()
    
    print("=== Bengali Dataset Processor ===")
    
    try:
        # Explore dataset structure
        print("1. Exploring dataset structure...")
        info = processor.explore_dataset_structure()
        print(f"Dataset info: {json.dumps(info, indent=2, ensure_ascii=False)}")
        
        # Get dataset statistics
        print("\n2. Getting dataset statistics...")
        stats = processor.get_dataset_stats()
        print(f"Dataset stats: {json.dumps(stats, indent=2, ensure_ascii=False)}")
        
        # Create sample data
        print("\n3. Creating sample Bengali data...")
        sample_data = processor.create_sample_bengali_data(10)
        print(f"Created {len(sample_data)} sample conversations")
        
        # Save sample data
        with open("bengali_sample_data.json", 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print("Saved sample data to bengali_sample_data.json")
        
    except Exception as e:
        print(f"Error: {e}")
        print("This is expected on macOS without proper ML environment.")

if __name__ == "__main__":
    main()
