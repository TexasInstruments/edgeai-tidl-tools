import numpy as np
import os
from typing import Dict, List, Tuple, Union, Optional, Any

class BINloader():
    """
    Loader for BIN files.
    Supports continuous loading from a long binary file with different datatypes.
    """
    def __init__(self, file_path: Tuple[str, ...]):
        """
        Initialize the BIN loader.
        
        Args:
            file_path (Tuple[str, ...]): Path to the BIN file
        """
        self.file_path = file_path[0]

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[ERROR] {self.file_path} not found")
        
        with open(self.file_path, 'rb') as f:
            self.binary_data = f.read()

        self.start_idx = 0
        self.binary_size = len(self.binary_data)
    
    def load(self, shape: Tuple[int, ...], dtype: np.dtype, format: str = None) -> np.ndarray:
        """
        Load data from the BIN file starting from the current position.
        Each call to load can use a different datatype and will continue
        from where the previous load left off.
    
        Args:
            shape (Tuple[int, ...]): Shape of the data
            dtype (np.dtype): Type of the data for this specific load
            format (str, optional): Format (NCHW or NHWC). Default: None

        Returns:
            np.ndarray: The loaded data
        """
        # Check for dynamic shapes
        for i, dim in enumerate(shape):
            if not isinstance(dim, int):
                raise ValueError(f"[ERROR] Binary loader does not support dynamic shape {shape}")
                
        # Calculate the byte size of the requested data
        element_size = np.dtype(dtype).itemsize
        total_bytes_needed = int(np.prod(shape)) * element_size
        
        # Check if we have enough bytes left in the binary data
        if (self.start_idx + total_bytes_needed) > self.binary_size:
            raise ValueError(f"[ERROR] Not enough data left in the binary file. "
                            f"Requested {total_bytes_needed} bytes starting at position {self.start_idx}, "
                            f"but only {self.binary_size - self.start_idx} bytes are available.")
        
        # Extract the relevant portion of binary data for this load
        chunk = self.binary_data[self.start_idx:self.start_idx + total_bytes_needed]
        
        # Convert to numpy array with the specified dtype
        data = np.frombuffer(chunk, dtype=dtype)
        
        # Reshape according to the requested shape
        data = data.reshape(shape)
        
        # Update the start index for the next load
        self.start_idx += total_bytes_needed
        
        return data
        
    def reset(self):
        """
        Reset the loader to start reading from the beginning of the file again.
        """
        self.start_idx = 0
        
    def get_remaining_bytes(self) -> int:
        """
        Get the number of bytes remaining in the binary file.
        
        Returns:
            int: Number of bytes remaining
        """
        return self.binary_size - self.start_idx
