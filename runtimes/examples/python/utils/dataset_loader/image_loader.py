import numpy as np
import os
from typing import Dict, List, Tuple, Union, Optional, Any
import PIL
from PIL import Image

class Imageloader():
    """
    Loader for image files using PIL.
    Supports loading individual images and batches of images.
    """
    def __init__(self, file_path: Tuple[str, ...]):
        """
        Initialize the Image loader.
        
        Args:
            file_path (Tuple[str, ...]): Path to one or more image files. For batch processing, 
                                         provide multiple file paths in the tuple.
        """
        self.file_path = file_path
        supported_extensions = ['.jpg', '.jpeg', '.png']
        
        for file in self.file_path:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"[ERROR] {file} not found")
            
            if os.path.splitext(file.lower())[-1] not in supported_extensions:
                raise ValueError(f"[ERROR] {file} is not a supported image file. Supported formats: {', '.join(supported_extensions)}")

        self.data_count = len(self.file_path)
        self.curr_data = 0

    def load(self, shape: Tuple[int, ...], dtype: np.dtype, format: str = "NCHW") -> np.ndarray:
        """
        Load an image or batch of images and convert to a numpy array with the specified shape and dtype.
        
        Args:
            shape (Tuple[int, ...]): Shape of the output array (e.g., (1, 3, 224, 224) for single RGB image in NCHW format
                                     or (N, 3, 224, 224) for batch of N images in NCHW format)
            dtype (np.dtype): Data type of the output array
            format (str, optional): Format (NCHW or NHWC). Default: "NCHW"
            
        Returns:
            np.ndarray: The loaded image(s) as a numpy array with the specified shape and dtype
        """
        # Check for dynamic shapes
        for i, dim in enumerate(shape):
            if not isinstance(dim, int):
                raise ValueError(f"[ERROR] Image loader does not support dynamic shape {shape}")
        
        if len(shape) < 3:
            raise ValueError(f"[ERROR] Invalid shape {shape} for image. Expected at least 3 dimensions")
        
        supported_formats = ("NCHW", "NHWC")
        if format not in ("NCHW", "NHWC"):
            raise ValueError(f"[ERROR] Invalid format {format}. Supported formats: {', '.join(supported_formats)}")

        if format == "NCHW":
            width = shape[-1]
            height = shape[-2]
        else:
            width = shape[-2]
            height = shape[-3]

        input_data = np.zeros(shape).astype(dtype)
        batch = shape[0]
        for i in range(batch):
            if self.curr_data >= self.data_count:
                self.curr_data = 0

            image = Image.open(self.file_path[self.curr_data]).convert("RGB").resize((width, height), PIL.Image.LANCZOS)
            t_input_data = np.expand_dims(image, axis=0)

            if len(t_input_data.shape) < 3:
                raise ValueError(f"[ERROR] Invalid shape {t_input_data.shape} for loaded image. Expected at least 3 dimensions")

            if format == "NCHW":
                t_input_data = np.transpose(t_input_data, (0, 3, 1, 2))

            input_data[i] = t_input_data[0]

            self.curr_data += 1

        input_data = input_data.reshape(shape).astype(dtype)

        return input_data

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
