# Pre Process

A flexible and extensible module for pre-processing input data for AI model inference.

## Overview

The Pre Process module provides a unified interface for pre-processing input data before feeding it to AI models. Currently, it supports:

- Basic pre-processing with mean subtraction and scaling

This module is particularly useful for machine learning workflows where you need to prepare input data for model inference, ensuring it meets the expected format and normalization requirements.

## Components

### PreProcess Factory

The `PreProcess` class is a factory that creates the appropriate pre-processor based on the specified type:

```python
from pre_process import PreProcess

# Create a basic pre-processor with default parameters
basic_processor = PreProcess.create_pre_process('basic')

# Create a basic pre-processor with custom parameters
basic_processor_with_params = PreProcess.create_pre_process('basic', params={
    'input_mean': [123.675, 116.28, 103.53],
    'input_scale': [0.017125, 0.017507, 0.017429]
})
```

### Pre-Processor Types

#### PreProcessBasic

Applies mean subtraction and scaling to input data, which are common operations for normalizing inputs to neural networks.

```python
# Create a basic pre-processor with ImageNet mean and scale values
processor = PreProcess.create_pre_process('basic', params={
    'input_mean': [123.675, 116.28, 103.53],  # RGB mean values
    'input_scale': [0.017125, 0.017507, 0.017429]  # 1/std for each channel
})

# Process input data
processed_data = processor.process(input_data, format="NCHW")
```

Key features:
- Supports both NCHW and NHWC data formats
- Applies channel-wise mean subtraction
- Applies channel-wise scaling
- Preserves input dimensions and data type

## Error Handling

The pre-processor includes robust error handling:

- Validation of input format (NCHW or NHWC)
- Graceful handling of missing parameters with sensible defaults
- Validation of pre-processor types

## Creating Your Own Custom Pre-Processor

You can extend the pre_process module by creating your own custom pre-processor. Here's a step-by-step guide:

### 1. Create a Pre-Processor Class

Create a new Python file (e.g., `pre_process_custom.py`) with a class that implements the pre-processor interface:

```python
import numpy as np
from typing import Dict

class PreProcessCustom:
    """
    Custom pre-processor implementation.
    """
    def __init__(self, params: Dict = None):
        """
        Initialize the custom pre-processor.
        
        Args:
            params (Dict, optional): Parameters for the pre-processor
        """
        self.param1 = params.get('param1', default_value1) if params else default_value1
        self.param2 = params.get('param2', default_value2) if params else default_value2
        
    def process(self, input: np.ndarray, format: str) -> np.ndarray:
        """
        Execute pre-processing on the input data.
        
        Args:
            input (np.ndarray): Input data to do pre-processing on
            format (str): Format (NCHW or NHWC)
            
        Returns:
            np.ndarray: Pre-processed input data
        """
        # Validate format
        supported_formats = ("NCHW", "NHWC")
        if format not in supported_formats:
            raise ValueError(f"[ERROR] Invalid format {format}. Supported formats: {', '.join(supported_formats)}")
        
        # Implementation specific to your pre-processor
        # Must return a numpy array with the same shape as input
        
        # Example implementation:
        processed_input = input.copy()
        
        # Apply custom pre-processing operations
        # ...
        
        return processed_input
```

### 2. Integrate with the PreProcess Factory

Modify the `pre_process.py` file to include your custom pre-processor:

```python
# Add import for your custom pre-processor
from pre_process_custom import PreProcessCustom

class PreProcess:
    """
    Factory class for appropriate pre processing.
    """
    @staticmethod
    def create_pre_process(pre_process_type: str, **kwargs):
        """
        Create a pre process of the specified type.
        """
        pre_process_type = pre_process_type.strip()
 
        if pre_process_type.lower() == 'basic':    
            if 'params' in kwargs:
                return PreProcessBasic(kwargs['params'])
            else:
                return PreProcessBasic()
                
        # Add your custom pre-processor
        elif pre_process_type.lower() == 'custom':
            if 'params' in kwargs:
                return PreProcessCustom(kwargs['params'])
            else:
                return PreProcessCustom()

        else:
            raise ValueError(f"Unsupported pre processing type: {pre_process_type}")
```