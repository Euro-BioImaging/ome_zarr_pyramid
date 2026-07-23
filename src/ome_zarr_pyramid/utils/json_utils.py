"""JSON serialization and conversion utilities for numpy types and nested structures."""

import json
import warnings
from typing import Any

import numpy as np


def convert_np_types(obj: Any) -> Any:
    """Convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Object to convert
    
    Returns:
        Object with NumPy types replaced by Python equivalents
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def is_valid_json(my_dict: Any) -> bool:
    """Check if object is JSON serializable.
    
    Args:
        my_dict: Object to check
    
    Returns:
        True if object can be serialized to JSON
    """
    try:
        json.dumps(my_dict)
        return True
    except (TypeError, ValueError):
        warnings.warn("Object is not a valid JSON!")
        return False


def make_json_safe(obj: Any) -> Any:
    """Recursively convert object to JSON-safe representation.
    
    Handles nested dictionaries, lists, and NumPy types recursively,
    ensuring all values are JSON serializable.
    
    Args:
        obj: Object to convert (can be nested)
    
    Returns:
        Object with all NumPy types and nested structures converted to JSON-safe equivalents
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj
