"""
Startup utilities for handling initialization issues and network connectivity.
"""

import os
import logging
import asyncio
import socket
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def check_internet_connectivity(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """
    Check if internet connectivity is available.
    
    Args:
        host: Host to check connectivity to (default: Google DNS)
        port: Port to check (default: 53 for DNS)
        timeout: Timeout in seconds
        
    Returns:
        bool: True if connectivity is available, False otherwise
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False


def check_model_server_connectivity(url: str = "https://app.coqui.ai", timeout: int = 5) -> bool:
    """
    Check if the TTS model server is accessible.
    
    Args:
        url: URL to check
        timeout: Timeout in seconds
        
    Returns:
        bool: True if server is accessible, False otherwise
    """
    try:
        import requests
        response = requests.head(url, timeout=timeout)
        return response.status_code < 400
    except Exception:
        return False


def set_offline_mode_env_vars():
    """Set environment variables for offline mode operation."""
    # Disable automatic model downloads
    os.environ['TTS_CACHE_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    
    # Set cache directories
    cache_dir = os.path.expanduser("~/.cache/tts")
    os.environ['TTS_CACHE'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
    
    logger.info("Offline mode environment variables set")


def configure_model_cache():
    """Configure model cache directories."""
    cache_dir = os.path.expanduser("~/.cache/tts")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set TTS cache directory
    os.environ['TTS_CACHE'] = cache_dir
    
    logger.info(f"Model cache configured at: {cache_dir}")


async def wait_for_connectivity(max_wait: int = 60, check_interval: int = 5) -> bool:
    """
    Wait for internet connectivity to become available.
    
    Args:
        max_wait: Maximum time to wait in seconds
        check_interval: Interval between checks in seconds
        
    Returns:
        bool: True if connectivity became available, False if timeout
    """
    elapsed = 0
    while elapsed < max_wait:
        if check_internet_connectivity():
            logger.info("Internet connectivity restored")
            return True
        
        logger.info(f"Waiting for connectivity... ({elapsed}/{max_wait}s)")
        await asyncio.sleep(check_interval)
        elapsed += check_interval
    
    logger.warning("Connectivity timeout reached")
    return False


def suppress_common_warnings():
    """Suppress common warnings that don't affect functionality."""
    import warnings
    
    # Suppress Pydantic V2 warnings
    warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    
    # Suppress pkg_resources deprecation warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    warnings.filterwarnings("ignore", category=UserWarning, module="jieba")
    
    # Suppress TTS model warnings
    warnings.filterwarnings("ignore", message=".*use_capacitron_vae.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="TTS")
    
    logger.info("Common warnings suppressed")


def get_available_models_offline() -> list:
    """Get list of models available offline."""
    cache_dir = os.path.expanduser("~/.cache/tts")
    available_models = []
    
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            if os.path.isdir(os.path.join(cache_dir, item)):
                # Convert cache directory name back to model name
                model_name = item.replace('--', '/')
                available_models.append(model_name)
    
    return available_models


def create_fallback_model_config() -> dict:
    """Create a fallback configuration for when models fail to load."""
    return {
        "fallback_enabled": True,
        "fallback_models": [
            "tts_models/multilingual/multi-dataset/your_tts",
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/glow-tts"
        ],
        "offline_mode": not check_internet_connectivity(),
        "retry_attempts": 3,
        "retry_delay": 5
    }


async def initialize_with_fallback(init_func, max_retries: int = 3, delay: int = 5):
    """
    Initialize a service with fallback and retry logic.
    
    Args:
        init_func: Async function to initialize
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Result of init_func or raises the last exception
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await init_func()
        except Exception as e:
            last_exception = e
            logger.warning(f"Initialization attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
    
    # If all attempts failed, raise the last exception
    raise last_exception