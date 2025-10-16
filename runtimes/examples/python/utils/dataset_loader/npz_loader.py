import numpy as np
import os
from typing import Dict, List, Tuple, Union, Optional, Any
import math

class NPZloader():
    """
    Loader for NPZ files.
    """
    def __init__(self, file_path: Tuple[str, ...]):
        """
        Initialize the NPZ loader.
        
        Args:
            file_path (Tuple[str, ...]): Path to the NPZ file
        """
        self.file_path = file_path[0]

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[ERROR] {self.file_path} not found")

        if not self.file_path.endswith('.npz'):
            raise ValueError(f"[ERROR] {self.file_path} is not a NPZ file")

        self.data = np.load(self.file_path)
        # Convert to list once for efficiency
        self.data_values = list(self.data.values())
        self.data_count = len(self.data_values)
        
        if self.data_count == 0:
            raise ValueError(f"[ERROR] {self.file_path} contains no data")
            
        self.curr_data = 0

    def load(self, shape: Tuple[int, ...], dtype: np.dtype, format: str = None) -> np.ndarray:
        """
        Load data from the NPZ file.
    
        Args:
            shape (Tuple[int, ...]): Expected shape of the data
            dtype (np.dtype): Expected type of the data
            format (str, optional): Format (NCHW or NHWC). Default: None

        Returns:
            np.ndarray: The loaded data
        """
        if self.curr_data >= self.data_count:
            self.curr_data = 0
            
        data = self.data_values[self.curr_data]
    
        data_shape = list(data.shape)
        x_shape = list(shape)

        while data_shape[0] == 1:
            data_shape.pop(0)

        while x_shape[0] == 1:
            x_shape.pop(0)

        # if data_shape != x_shape:
        #     raise ValueError(f"[ERROR] Loaded data shape {data.shape} does not match expected shape {shape}")

        if math.prod(data_shape) != math.prod(x_shape):
            raise ValueError(f"[ERROR] Loaded data shape {data.shape} volume does not match expected shape {shape} volume")

        if data.dtype != dtype:
            raise ValueError(f"[ERROR] Loaded data type {data.dtype} does not match expected type {dtype}")

        data = data.reshape(shape)

        self.curr_data += 1
        
        return data
        
    def reset(self):
        """
        Reset the loader to start reading from the beginning of the file again.
        """
        self.curr_data = 0
        
    def get_remaining_items(self) -> int:
        """
        Get the number of data items remaining before wrapping around.
        
        Returns:
            int: Number of data items remaining
        """
        return self.data_count - self.curr_data
