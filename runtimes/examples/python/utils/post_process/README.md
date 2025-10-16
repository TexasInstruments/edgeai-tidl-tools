# Post Process

A flexible and extensible module for post-processing inference outputs from various AI models.

## Overview

The Post Process module provides a unified interface for post-processing inference outputs from different types of AI models:

- Image Classification
- Object Detection
- Semantic Segmentation

This module is particularly useful for machine learning workflows where you need to visualize or extract meaningful information from raw model outputs.

## Components

### PostProcess Factory

The `PostProcess` class is a factory that creates the appropriate post-processor based on the specified type:

```python
from post_process import PostProcess

# Create a classification post-processor
classification_processor = PostProcess.create_post_process('classification')

# Create a classification post-processor with parameters
classification_processor_with_params = PostProcess.create_post_process('classification', params={'labels': 'path/to/labels.txt', 'label_offset': 1})

# Create a detection post-processor
detection_processor = PostProcess.create_post_process('detection', params={'od_type': 'YoloV5'})

# Create a segmentation post-processor
segmentation_processor = PostProcess.create_post_process('segmentation')
```

### Post-Processor Types

#### PostProcessClassification

Processes outputs from image classification models, displaying class labels and confidence scores.

```python
# Create a classification post-processor
processor = PostProcess.create_post_process('classification', params={
    'labels': 'path/to/labels.txt',  # Optional: Path to text file with class labels
    'label_offset': 0                # Optional: Offset to apply to label indices
})

# Process model outputs
classes, output_image = processor.process(input_image, model_outputs)
```

Key features:
- Supports loading class labels from a text file
- Displays top-5 predictions with confidence scores
- Overlays results on the input image
- Handles label index offsets if needed

#### PostProcessDetection

Processes outputs from object detection models, visualizing bounding boxes and confidence scores.

```python
# Create a detection post-processor
processor = PostProcess.create_post_process('detection', params={
    'od_type': 'YoloV5'  # Required: Type of object detection model
})

# Process model outputs
detections, output_image = processor.process(input_image, model_outputs)
```

Key features:
- Supports multiple object detection frameworks and model types:
  - MMDetection framework
  - SSD models
  - YoloV5 models
  - Models with built-in detection post-processing layers
  - EfficientDet Lite models
- Visualizes bounding boxes with different colors for each class
- Filters detections based on confidence threshold (default: 0.3)
- Returns detection information in a structured format

#### PostProcessSegmentation

Processes outputs from semantic segmentation models, creating color-coded segmentation masks.

```python
# Create a segmentation post-processor
processor = PostProcess.create_post_process('segmentation')

# Process model outputs
mask, output_image = processor.process(input_image, model_outputs)
```

Key features:
- Handles both single-channel class maps and multi-channel probability maps
- Automatically resizes the segmentation mask to match input dimensions
- Creates color-coded visualization using a predefined color palette
- Blends the segmentation mask with the original image for better visualization

## Utility Functions

The module includes utility functions in `post_process_utils.py`:

- `RGB2YUV`: Converts RGB images to YUV color space
- `YUV2RGB`: Converts YUV images back to RGB color space
- `COLORS_LIST`: A predefined list of colors used for visualization

These utilities are used internally by the post-processors but can also be used directly if needed.

## Error Handling

All post-processors include robust error handling:

- Validation of required parameters
- Support for different output formats from various model architectures
- Graceful handling of missing label files
- Validation of post-processor types

## Creating Your Own Custom Post-Processor

You can extend the post_process module by creating your own custom post-processor. Here's a step-by-step guide:

### 1. Create a Post-Processor Class

Create a new Python file (e.g., `post_process_custom.py`) with a class that implements the post-processor interface:

```python
import numpy as np
from typing import Dict, Tuple
from PIL import Image, ImageDraw

class PostProcessCustom:
    """
    Custom post-processor implementation.
    """
    def __init__(self, params: Dict = None):
        """
        Initialize the custom post-processor.
        
        Args:
            params (Dict, optional): Parameters for the post-processor
        """
        self.some_parameter = params.get('some_parameter', default_value) if params else default_value
        
    def process(self, input: Image, outputs: np.ndarray) -> Tuple[str, Image]:
        """
        Execute post-processing on the image.
        
        Args:
            input (Image): Input image to do post-processing on
            outputs (np.ndarray): Inference outputs
            
        Returns:
            str: Processed information in text format
            Image: Post-processed output image
        """
        # Implementation specific to your post-processor
        # Must return a tuple of (information_string, processed_image)
        
        # Example implementation:
        img = input.copy().convert("RGBA")
        draw = ImageDraw.Draw(img)
        
        # Process the outputs
        # ...
        
        # Create information string
        info = "Custom processing results:\n"
        # ...
        
        img = img.convert("RGB")
        return info, img
```

### 2. Integrate with the PostProcess Factory

Modify the `post_process.py` file to include your custom post-processor:

```python
# Add import for your custom post-processor
from post_process_custom import PostProcessCustom

class PostProcess:
    """
    Factory class for appropriate post processing.
    """
    @staticmethod
    def create_post_process(post_process_type: str, **kwargs):
        """
        Create a post process of the specified type.
        """
        post_process_type = post_process_type.strip()
 
        if post_process_type.lower() == 'detection':
            required_args = ['params']
            for arg in required_args:
                if arg not in kwargs:
                    raise ValueError(f"Missing required argument '{arg}' for PostProcessDetection")
    
            return PostProcessDetection(kwargs['params'])
        
        elif post_process_type.lower() == 'classification':
            if 'params' in kwargs:
                return PostProcessClassification(kwargs['params'])
            else:
                return PostProcessClassification()
        
        elif post_process_type.lower() == 'segmentation':
            return PostProcessSegmentation()
            
        # Add your custom post-processor
        elif post_process_type.lower() == 'custom':
            if 'params' in kwargs:
                return PostProcessCustom(kwargs['params'])
            else:
                return PostProcessCustom()

        else:
            raise ValueError(f"Unsupported post processing type: {post_process_type}")
```
