"""Health Monitoring with Wearables - Edge AI System.

A comprehensive edge AI system for health monitoring using wearable devices.
Designed for research and educational purposes only.
"""

__version__ = "0.1.0"
__author__ = "Edge AI Team"
__email__ = "team@example.com"

# Set deterministic behavior
import os
import random
from typing import Optional

import numpy as np
import torch
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    
    # Ensure deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> str:
    """Get the appropriate device for computation.
    
    Args:
        device: Preferred device ("cuda", "cpu", "auto").
        
    Returns:
        Device string for PyTorch/TensorFlow.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


# Set default seed
set_seed(42)
