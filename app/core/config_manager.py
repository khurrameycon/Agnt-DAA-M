"""
Configuration manager for sagax1
Updated to use installation directory for config paths on macOS
"""

import os
import json
import logging
from typing import Dict, Any, Optional

DEFAULT_CONFIG = {
    "api_keys": {
        "huggingface": "",
        "openai": "",
        "gemini": "",
        "groq": "",
        "anthropic": ""  # Add Anthropic API key
    },
    "api_providers": {
        "openai": {
            "default_model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1"
        },
        "gemini": {
            "default_model": "gemini-2.0-flash-exp",
            "base_url": "https://generativelanguage.googleapis.com/v1beta"
        },
        "groq": {
            "default_model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1"
        },
        "anthropic": {  # Add Anthropic configuration
            "default_model": "claude-sonnet-4-20250514",
            "base_url": "https://api.anthropic.com"
        }
    },
    "models": {
        "default_model": "meta-llama/Llama-3.2-3B-Instruct",
        "cache_dir": "~/.cache/sagax1/models"
    },
    "ui": {
        "theme": "light",
        "font_size": 12
    },
    "agents": {
        "default_agent": "text_completion",
        "max_history": 100
    },
    "execution": {
        "python_executor": "local",  # Options: local, docker, e2b
        "max_execution_time": 30,    # seconds
        "authorized_imports": [
            "numpy", "pandas", "matplotlib", "PIL", "requests", 
            "bs4", "datetime", "math", "re", "os", "csv", "json"
        ]
    },
    "web": {
        "browser_user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "browser_width": 1280,
        "browser_height": 800
    }
}

class ConfigManager:
    """Manages the application configuration"""
    
    def __init__(self, config_path: str = None):
        """Initialize the configuration manager
        
        Args:
            config_path: Path to the configuration file (optional)
        """
        
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Use installation directory as base for config
        if config_path is None:
            # Get current working directory (installation directory)
            base_dir = os.getcwd()
            config_path = os.path.join(base_dir, "config", "config.json")
        
        self.config_path = config_path
        self.config = self._load_config()
        
        # Debug logging for troubleshooting
        self.logger.info(f"Config path set to: {self.config_path}")
        self.logger.info(f"Current working directory: {os.getcwd()}")
        self.logger.info(f"Config file exists: {os.path.exists(self.config_path)}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default if it doesn't exist"""
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                self.logger.info(f"Loaded existing config from {self.config_path}")
                return loaded_config
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"Error loading config: {e}")
                self.logger.info("Using default configuration")
                return DEFAULT_CONFIG.copy()
        else:
            # Create config directory if it doesn't exist
            config_dir = os.path.dirname(self.config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
                self.logger.info(f"Created config directory: {config_dir}")
            
            # Save default config
            self._save_config(DEFAULT_CONFIG)
            self.logger.info(f"Created new config file at {self.config_path}")
            return DEFAULT_CONFIG.copy()
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            self.logger.debug(f"Config saved to {self.config_path}")
        except IOError as e:
            self.logger.error(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key
        
        Args:
            key: Key path using dot notation (e.g., 'agents.default_agent')
            default: Default value to return if key is not found
            
        Returns:
            The configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any, save: bool = True) -> None:
        """Set a configuration value by key
        
        Args:
            key: Key path using dot notation (e.g., 'agents.default_agent')
            value: Value to set
            save: Whether to save the configuration to file
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the right level
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        if save:
            self._save_config(self.config)
    
    # def get_hf_api_key(self) -> Optional[str]:
    #     """Get the Hugging Face API key
        
    #     Returns:
    #         The API key or None if not set
    #     """
    #     # First check environment variable
    #     api_key = os.environ.get("HF_API_KEY")
        
    #     # If not in environment, check config
    #     if not api_key:
    #         api_key = self.get("api_keys.huggingface")
        
    #     return api_key if api_key else None
    
    def set_hf_api_key(self, api_key: str) -> None:
        """Set the Hugging Face API key
        
        Args:
            api_key: The API key to set
        """
        self.set("api_keys.huggingface", api_key)
    
    def get_hf_api_key(self) -> Optional[str]:
        """Get the Hugging Face API key - config only"""
        api_key = self.get("api_keys.huggingface")
        return api_key if api_key else None
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get the OpenAI API key - config only"""
        return self.get("api_keys.openai")

    def get_gemini_api_key(self) -> Optional[str]:
        """Get the Gemini API key - config only"""
        return self.get("api_keys.gemini")

    def get_groq_api_key(self) -> Optional[str]:
        """Get the Groq API key - config only"""
        return self.get("api_keys.groq")

    def get_anthropic_api_key(self) -> Optional[str]:
        """Get the Anthropic API key - config only"""
        return self.get("api_keys.anthropic")

    def set_openai_api_key(self, api_key: str) -> None:
        self.set("api_keys.openai", api_key)

    def set_gemini_api_key(self, api_key: str) -> None:
        self.set("api_keys.gemini", api_key)

    def set_groq_api_key(self, api_key: str) -> None:
        self.set("api_keys.groq", api_key)

    def set_anthropic_api_key(self, api_key: str) -> None:
        """Set the Anthropic API key
        
        Args:
            api_key: The API key to set
        """
        self.set("api_keys.anthropic", api_key)
    
    def save(self) -> None:
        """Save the current configuration to file"""
        self._save_config(self.config)