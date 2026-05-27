import numpy as np
import os
from typing import Dict, List, Tuple, Union, Optional, Any

class Randomloader():
    """
    Generate random numpy array
    """
    def __init__(self):
        """
        Initialize the Random loader.
        """
        pass
    
    def load(self, shape: Tuple[int, ...], dtype: np.dtype, format: str = None, seed: int = 0) -> np.ndarray:
        """
        Generate random data with the specified shape and dtype.
    
        Args:
            shape (Tuple[int, ...]): Shape of the data
            dtype (np.dtype): Datatype of the data
            format (str, optional): Format (NCHW or NHWC). Default: None
            seed  (int, optional): Random seed. Default: 0

        Returns:
            np.ndarray: The loaded data
        """
        # Check for dynamic shapes
        for i, dim in enumerate(shape):
            if not isinstance(dim, (int, np.integer)):
                raise ValueError(f"[ERROR] Random loader does not support dynamic shape {shape}")

        np.random.seed(seed)
        data = np.random.randn(*shape).astype(dtype)
        return data
        
    def reset(self):
        """
        Reset the loader state. For RandomLoader, this is a no-op since there's no state to reset.
        """
        pass
        
    def get_remaining_items(self) -> int:
        """
        Get the number of data items remaining. For RandomLoader, this always returns a large number
        since it can generate unlimited random data.
        
        Returns:
            int: A large number representing unlimited data availability
        """
        return 1000000  # Return a large number to indicate unlimited data
