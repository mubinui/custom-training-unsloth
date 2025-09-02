#!/usr/bin/env python3
"""
Test script to verify Bengali dataset access without token
"""

def test_dataset_access():
    """Test if we can access the Bengali dataset without a token"""
    
    print("Testing Bengali dataset access...")
    print("Dataset: spedrox-sac/bengali_chat_conv")
    print("-" * 50)
    
    try:
        # Test 1: Load with datasets library
        print("Test 1: Loading with datasets library...")
        from datasets import load_dataset
        
        # Load a small sample first
        dataset = load_dataset("spedrox-sac/bengali_chat_conv", split="train[:5]")
        print(f"✅ Successfully loaded {len(dataset)} samples")
        
        # Show first example
        if len(dataset) > 0:
            first_example = dataset[0]
            print(f"Sample data structure: {list(first_example.keys())}")
            print(f"First question: {first_example.get('Question', 'N/A')}")
            print(f"First answer: {first_example.get('Answer', 'N/A')}")
        
        print("✅ Test 1 PASSED - No token required!")
        
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        return False
    
    try:
        # Test 2: Check dataset info
        print("\nTest 2: Getting dataset info...")
        from datasets import get_dataset_config_names, get_dataset_split_names
        
        configs = get_dataset_config_names("spedrox-sac/bengali_chat_conv")
        splits = get_dataset_split_names("spedrox-sac/bengali_chat_conv")
        
        print(f"Available configs: {configs}")
        print(f"Available splits: {splits}")
        print("✅ Test 2 PASSED - Dataset info accessible!")
        
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        return False
    
    try:
        # Test 3: Load full train split info
        print("\nTest 3: Loading full dataset info...")
        full_dataset = load_dataset("spedrox-sac/bengali_chat_conv", split="train")
        print(f"✅ Full dataset size: {len(full_dataset)} samples")
        print(f"Dataset features: {full_dataset.features}")
        print("✅ Test 3 PASSED - Full dataset accessible!")
        
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Bengali dataset is publicly accessible")
    print("✅ No Hugging Face token required")
    print("✅ Ready for training!")
    
    return True

if __name__ == "__main__":
    try:
        success = test_dataset_access()
        if not success:
            print("\n⚠️  Some tests failed. You may need to install the datasets library:")
            print("pip install datasets")
    except ImportError:
        print("❌ datasets library not found. Install with:")
        print("pip install datasets")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
