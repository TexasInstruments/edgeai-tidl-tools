import numpy as np
import os
from typing import Dict, List, Tuple, Union, Optional, Any

DTYPE_RANGES = {
    np.int8: (-(2**7), 2**7 - 1),
    np.uint8: (0, 2**8 - 1),
    np.int16: (-(2**15), 2**15 - 1),
    np.uint16: (0, 2**16 - 1),
    np.int32: (-(2**31), 2**31 - 1),
    np.uint32: (0, 2**32 - 1),
    np.int64: (-(2**63), 2**63 - 1),
    np.uint64: (0, 2**63),  # Uses uniform() since randint() doesn't support ranges beyond int64 max
}

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
        if dtype == np.bool_:
            prob_true = 0.5
            data = np.random.binomial(1, prob_true, size=shape).astype(dtype)
        elif dtype in DTYPE_RANGES:
            low, high = DTYPE_RANGES[dtype]
            if dtype == np.uint64:
                data = np.random.uniform(low, high, size=shape).astype(dtype)
            else:
                data = np.random.randint(low, high + 1, size=shape).astype(dtype)
        else:
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
