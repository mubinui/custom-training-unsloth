#!/usr/bin/env python3
"""
Bengali Dataset Explorer
Explores the structure of the Bengali chat conversation dataset
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def explore_bengali_dataset():
    """Explore the Bengali chat conversation dataset structure"""
    try:
        from datasets import load_dataset
        
        print("Loading Bengali chat conversation dataset...")
        ds = load_dataset("spedrox-sac/bengali_chat_conv")
        
        print(f"Dataset structure: {ds}")
        print(f"Keys: {list(ds.keys())}")
        
        if 'train' in ds:
            train_ds = ds['train']
            print(f"Training samples: {len(train_ds)}")
            print(f"Columns: {train_ds.column_names}")
            
            # Show first few examples
            print("\nFirst 3 examples:")
            for i in range(min(3, len(train_ds))):
                print(f"\nExample {i+1}:")
                example = train_ds[i]
                for key, value in example.items():
                    print(f"  {key}: {value}")
        
        return ds
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("This is expected on macOS without proper ML environment.")
        return None

if __name__ == "__main__":
    explore_bengali_dataset()
