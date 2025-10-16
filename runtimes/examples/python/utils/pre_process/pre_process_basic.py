import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any

class PreProcessBasic():
    """
    Basic Pre Process Class with mean and scale
    """
    def __init__(self, params: Dict = None):
        """
        Initialize the Basic Pre Process
        """
        self.input_mean = params.get('input_mean', [0, 0, 0])
        self.input_scale = params.get('input_scale', [1, 1, 1])

    def process(self, input: np.ndarray, format : str) -> np.ndarray:
        """
        Execute post processing on the image
        
        Args:
            input (np.ndarray): Input data to do pre-processing on
            format (str): Format (NCHW or NHWC)
            
        Returns:
            np.ndarray: Pre Processed input data
        """
        supported_formats = ("NCHW", "NHWC")
        if format not in ("NCHW", "NHWC"):
            raise ValueError(f"[ERROR] Invalid format {format}. Supported formats: {', '.join(supported_formats)}")
        
        if format == "NCHW":
            for mean, scale, ch in zip(self.input_mean, self.input_scale, range(input.shape[1])):
                input[:, ch, :, :] = (input[:, ch, :, :] - mean) * scale
        else:
            for mean, scale, ch in zip(self.input_mean, self.input_scale, range(input.shape[3])):
                input[:, :, :, ch] = (input[:, :, :, ch] - mean) * scale

        return input