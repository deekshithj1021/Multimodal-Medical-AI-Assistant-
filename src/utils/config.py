"""
Configuration utilities for the Multimodal Medical AI Assistant
"""

import os
from typing import Dict, Any
import yaml


class Config:
    """
    Configuration manager for the multimodal system.
    """
    
    # Default model configurations
    DEFAULT_CONFIG = {
        "audio": {
            "model_size": "base",  # tiny, base, small, medium, large
            "device": "cpu",
            "language": "en"
        },
        "visual": {
            "model_name": "Salesforce/blip-image-captioning-base",
            "device": "cpu"
        },
        "llm": {
            "model_name": "Qwen/Qwen-1_8B-Chat",
            "device": "cpu",
            "max_length": 512,
            "temperature": 0.7
        },
        "fusion": {
            "device": "cpu"
        }
    }
    
    @classmethod
    def load_config(cls, config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file or return defaults.
        
        Args:
            config_path: Path to configuration YAML file
        
        Returns:
            Configuration dictionary
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return {**cls.DEFAULT_CONFIG, **config}
        
        return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], config_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
        """
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    @classmethod
    def get_device(cls):
        """
        Automatically detect and return the best available device.
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
