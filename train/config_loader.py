#!/usr/bin/env python3
"""
Configuration loader utility for Mistral-7B NLI fine-tuning.
Loads Python files as configurations with proper error handling.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a Python configuration file as a dictionary of parameters.
    
    Args:
        config_path: Path to the Python configuration file
        
    Returns:
        Dictionary containing all configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ImportError: If config file cannot be imported
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    # Add the parent directory to sys.path temporarily to allow imports
    parent_dir = str(config_path.parent.absolute())
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        # Load the module
        module_name = config_path.stem
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Convert module attributes to a dictionary, excluding special attributes
        config = {
            key: getattr(config_module, key) 
            for key in dir(config_module) 
            if not key.startswith('__') and not key.endswith('__')
        }
        
        return config
        
    except Exception as e:
        raise ImportError(f"Failed to load config from {config_path}: {str(e)}") 