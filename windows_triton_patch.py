"""
Windows Triton Compatibility Patch
This module patches triton imports to make Unsloth work on Windows
"""
import sys
import importlib.util

def patch_triton_for_windows():
    """Create a mock triton module for Windows compatibility"""
    if 'triton' not in sys.modules:
        # Create mock triton module
        spec = importlib.util.spec_from_loader('triton', loader=None)
        triton_module = importlib.util.module_from_spec(spec)
        
        # Add required attributes
        triton_module.__version__ = "2.0.0"
        
        # Mock triton.language
        spec = importlib.util.spec_from_loader('triton.language', loader=None)
        language_module = importlib.util.module_from_spec(spec)
        language_module.device = lambda: 'cuda'
        triton_module.language = language_module
        
        # Mock triton.compiler
        spec = importlib.util.spec_from_loader('triton.compiler', loader=None)
        compiler_module = importlib.util.module_from_spec(spec)
        compiler_module.make_launcher = lambda *args, **kwargs: None
        triton_module.compiler = compiler_module
        
        # Add to sys.modules
        sys.modules['triton'] = triton_module
        sys.modules['triton.language'] = language_module
        sys.modules['triton.compiler'] = compiler_module
        
        print("Applied Windows triton compatibility patch")
        return True
    return False

# Apply patch automatically when imported
patch_triton_for_windows()
