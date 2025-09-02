#!/usr/bin/env python3
"""
Setup script for Bengali training
This script helps set up the environment and test the Bengali training pipeline
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'unsloth',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} is missing")
    
    return missing_packages

def setup_bengali_environment():
    """Setup Bengali training environment"""
    logger.info("Setting up Bengali training environment...")
    
    # Important note about dataset access
    logger.info("📋 DATASET ACCESS INFO:")
    logger.info("✅ spedrox-sac/bengali_chat_conv is a PUBLIC dataset")
    logger.info("✅ NO Hugging Face token required")
    logger.info("✅ Direct download available")
    
    # Copy Bengali environment template if .env doesn't exist
    if not os.path.exists('.env'):
        if os.path.exists('.env.bengali'):
            import shutil
            shutil.copy('.env.bengali', '.env')
            logger.info("Created .env file from Bengali template")
        else:
            logger.error(".env.bengali template not found")
            return False
    else:
        logger.info(".env file already exists")
    
    # Create output directory
    output_dir = "./outputs/bengali_model"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    return True

def test_bengali_data_processor():
    """Test the Bengali data processor"""
    logger.info("Testing Bengali data processor...")
    
    try:
        from bengali_data_processor import BengaliDataProcessor
        
        processor = BengaliDataProcessor()
        
        # Test sample data creation
        sample_data = processor.create_sample_bengali_data(num_samples=5)
        logger.info(f"✓ Created {len(sample_data)} sample Bengali data entries")
        
        # Print first sample
        if sample_data:
            logger.info(f"Sample Bengali data: {sample_data[0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Bengali data processor test failed: {e}")
        return False

def test_environment_config():
    """Test environment configuration loading"""
    logger.info("Testing environment configuration...")
    
    try:
        from env_config import load_config, check_gpu_compatibility
        
        config = load_config()
        logger.info(f"✓ Environment configuration loaded")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Max steps: {config.max_steps}")
        logger.info(f"Learning rate: {config.learning_rate}")
        
        # Check GPU
        gpu_info = check_gpu_compatibility()
        logger.info(f"GPU available: {gpu_info['cuda_available']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Environment configuration test failed: {e}")
        return False

def create_bengali_test_script():
    """Create a simple Bengali test script"""
    test_script_content = '''#!/usr/bin/env python3
"""
Quick Bengali training test
Run this to test Bengali training with sample data
"""

import sys
import subprocess

def main():
    print("Testing Bengali training with sample data...")
    
    # Run Bengali training with sample data
    cmd = [
        sys.executable,
        "main_bengali.py",
        "--sample",
        "--max-examples", "10",
        "--eval"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Bengali training test completed successfully!")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("✗ Bengali training test failed!")
        print("Error:", e.stderr)
        return False
    
    return True

if __name__ == "__main__":
    main()
'''
    
    with open("test_bengali_training.py", 'w', encoding='utf-8') as f:
        f.write(test_script_content)
    
    logger.info("Created test_bengali_training.py")

def main():
    """Main setup function"""
    logger.info("=== Bengali Training Setup ===")
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        logger.error("Missing packages detected. Please install:")
        for package in missing_packages:
            logger.error(f"  pip install {package}")
        return False
    
    # Setup environment
    if not setup_bengali_environment():
        logger.error("Failed to setup Bengali environment")
        return False
    
    # Test data processor
    if not test_bengali_data_processor():
        logger.error("Bengali data processor test failed")
        return False
    
    # Test environment config
    if not test_environment_config():
        logger.error("Environment configuration test failed")
        return False
    
    # Create test script
    create_bengali_test_script()
    
    logger.info("=== Setup Complete ===")
    logger.info("Bengali training environment is ready!")
    logger.info("")
    logger.info("To start Bengali training:")
    logger.info("  python main_bengali.py --sample  # Test with sample data")
    logger.info("  python main_bengali.py           # Train with full dataset")
    logger.info("  python test_bengali_training.py  # Quick test")
    logger.info("")
    logger.info("Available options:")
    logger.info("  --sample              Use sample data for testing")
    logger.info("  --eval                Run evaluation after training")
    logger.info("  --push                Push model to Hugging Face Hub")
    logger.info("  --max-examples N      Limit number of training examples")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
