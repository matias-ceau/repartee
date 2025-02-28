"""
Configuration module for Repartee.

Handles loading and parsing of configuration settings from 'settings.yaml'.
"""

import os
import platform
from argparse import Namespace
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_xdg_data_home():
    """Get the XDG_DATA_HOME directory according to the XDG Base Directory spec."""
    xdg_data_home = os.environ.get("XDG_DATA_HOME", "")
    if not xdg_data_home:
        if platform.system() == "Windows":
            xdg_data_home = os.path.join(os.environ["APPDATA"], "repartee")
        else:
            xdg_data_home = os.path.expanduser("~/.local/share")
    return xdg_data_home


def get_data_dir():
    """Get the data directory for repartee."""
    data_dir = os.path.join(get_xdg_data_home(), "repartee")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_api_key(key_name):
    """
    Get API key from environment variables or settings file.
    
    Args:
        key_name: Name of the API key
        
    Returns:
        The API key or None if not found
    """
    # Try to get the API key from environment variables
    api_key = os.getenv(key_name)
    if api_key:
        return api_key

    # If not found in environment variables, try to load from settings.yaml
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config.get(key_name)

    return None


class ReparteeConfig:
    """Configuration settings for Repartee."""
    
    def __init__(self):
        # Load default configuration
        self.load_defaults()
        
        # Load user configuration if available
        self.load_user_config()
    
    def load_defaults(self):
        """Load default configuration values."""
        self.models = Namespace(
            claude="claude-3-5-sonnet-20241022",
            openai="gpt-4o",
            gemini="gemini-1.5-pro",
            perplexity="sonar-small-online"
        )
        self.system_prompt = "You are a helpful AI assistant."
        self.data_dir = get_data_dir()
        self.embeddings = {
            "provider": "openai",
            "model": "text-embedding-3-small"
        }
        self.knowledge_dirs = []
        
    def load_user_config(self):
        """Load user configuration from settings.yaml."""
        config_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml"),
            os.path.expanduser("~/.config/repartee/settings.yaml"),
            os.path.join(self.data_dir, "settings.yaml")
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                with open(config_path, "r") as file:
                    config = yaml.safe_load(file)
                    if not config:
                        continue
                        
                    # Update configuration with user settings
                    if "models" in config:
                        for provider, model in config["models"].items():
                            setattr(self.models, provider, model)
                            
                    if "system_prompt" in config:
                        self.system_prompt = config["system_prompt"]
                        
                    if "data_dir" in config:
                        self.data_dir = config["data_dir"]
                        os.makedirs(self.data_dir, exist_ok=True)
                        
                    if "embeddings" in config:
                        self.embeddings.update(config["embeddings"])
                        
                    if "knowledge_dirs" in config:
                        self.knowledge_dirs = config["knowledge_dirs"]


# Create a global configuration instance
config = ReparteeConfig()

# For backward compatibility
ReparteeDefaults = Namespace(
    models=config.models,
    system_prompt=config.system_prompt
)
