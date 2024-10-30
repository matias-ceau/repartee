import os
from dotenv import load_dotenv
import yaml

"""
Configuration module for Repartee.

Handles loading and parsing of configuration settings from 'settings.yaml'.
"""

# ...existing code...
# Load environment variables from .env file
load_dotenv()

def get_api_key(key_name):
    # Try to get the API key from environment variables
    api_key = os.getenv(key_name)
    if api_key:
        return api_key

    # If not found in environment variables, try to load from settings.yaml
    config_path = os.path.join(os.path.dirname(__file__), 'settings.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config.get(key_name)

    return None