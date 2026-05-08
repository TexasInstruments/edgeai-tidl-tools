# Dataset Loader

A flexible and extensible module for loading input data from various sources.

## Overview

The Dataset Loader module provides a unified interface for loading input data from different sources:

- Random data generation
- NumPy .npz files
- Binary files
- Image files (jpg, jpeg, png)

This module is particularly useful for machine learning workflows where you need to load input data for model training, validation, or inference.

## Components

### DatasetLoader Factory

The `DatasetLoader` class is a factory that creates the appropriate loader based on the specified type:

```python
from dataset_loader import DatasetLoader

# Create a random loader
random_loader = DatasetLoader.create_loader('random')

# Create an NPZ loader
npz_loader = DatasetLoader.create_loader('npz', file_path='path/to/data.npz')

# Create a BIN loader
bin_loader = DatasetLoader.create_loader('bin', file_path='path/to/data.bin')

# Create an Image loader
img_loader = DatasetLoader.create_loader('img', file_path='path/to/image.jpg')
```

### Loader Types

#### RandomLoader

Generates random numpy arrays with specified shape and dtype.

```python
# Create a random loader
loader = DatasetLoader.create_loader('random')

# Generate random data
data = loader.load(shape=(1, 3, 224, 224), dtype=np.float32, seed=42)
```

#### NPZLoader

Loads data from NumPy .npz files. Supports cycling through multiple arrays in a single file.

```python
# Create an NPZ loader
loader = DatasetLoader.create_loader('npz', file_path='path/to/data.npz')

# Load data with specific shape and dtype
data = loader.load(shape=(1, 3, 224, 224), dtype=np.float32)

# Reset the loader to start from the beginning
loader.reset()

# Check how many items are remaining
remaining = loader.get_remaining_items()
```

Key features:
- **Important Note**: Data is loaded in sequence as arrays appear in the file, NOT based on the keys in the npz file. Make sure your arrays are in the correct order.
- Automatically wraps around to the beginning when all arrays have been used
- Flexible shape validation:
  - Resolves dynamic shapes using the dimensions in the input file
  - Removes leading dimensions of size 1 before comparison
  - Validates that total volume (product of dimensions) matches rather than exact shape
  - Allows for shape flexibility while ensuring data size compatibility
- Verifies dtype match the expected values

#### BINLoader

Loads data from binary files. Supports continuous loading with different datatypes.

```python
# Create a BIN loader
loader = DatasetLoader.create_loader('bin', file_path='path/to/data.bin')

# Load data with specific shape and dtype
data1 = loader.load(shape=(1, 3, 224, 224), dtype=np.float32)

# Load next chunk with different shape and dtype
data2 = loader.load(shape=(1, 1, 112, 112), dtype=np.int16)

# Reset the loader to start from the beginning
loader.reset()

# Check how many bytes are remaining
remaining_bytes = loader.get_remaining_bytes()
```

Key features:
- Maintains a position pointer in the binary file
- Supports different datatypes for each load call
- Memory-efficient as it only loads the requested portion of the binary file

#### ImageLoader

Loads image data from supported image file formats (jpg, jpeg, png) using PIL. Supports both single images and batches of images.

```python
# Create an Image loader with a single image
loader = DatasetLoader.create_loader('img', file_path='path/to/image.jpg')

# Load a single image with specific shape, dtype, and format
data = loader.load(shape=(1, 3, 224, 224), dtype=np.float32, format="NCHW")

# Create an Image loader with multiple images for batch processing
loader = DatasetLoader.create_loader('img', file_path=('path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg'))

# Load a batch of images
batch_data = loader.load(shape=(3, 3, 224, 224), dtype=np.float32, format="NCHW")
```

Key features:
- Supports common image formats: jpg, jpeg, png
- Automatically resizes images to match the requested dimensions
- Supports both NCHW and NHWC data formats
- Converts images to the specified data type
- **Batch processing**: Can load multiple images into a batch when the shape has a batch dimension > 1
- **Automatic cycling**: If the number of images needed for a batch exceeds the available images, it will cycle back to the beginning
- **State management**: Maintains the current position in the image list and provides methods to reset it

## Usage Examples

### Example 1: Loading Random Data

```python
from dataset_loader import DatasetLoader
import numpy as np

# Create a random loader
loader = DatasetLoader.create_loader('random')

# Generate random data with seed=0
data = loader.load(shape=(1, 3, 224, 224), dtype=np.float32)
```

### Example 2: Loading Data from NPZ File

```python
from dataset_loader import DatasetLoader
import numpy as np

# Create an NPZ loader
loader = DatasetLoader.create_loader('npz', file_path='input_data.npz')

# Process multiple frames
for i in range(5):
    # Load data (will cycle through available arrays)
    data = loader.load(shape=(1, 3, 224, 224), dtype=np.float32)
    # Process data...
```

### Example 3: Loading Data from Binary File

```python
from dataset_loader import DatasetLoader
import numpy as np

# Create a BIN loader
loader = DatasetLoader.create_loader('bin', file_path='input_data.bin')

# First load - float32 data
frame1 = loader.load(shape=(1, 3, 224, 224), dtype=np.float32)

# Second load - int16 data (continues from where the first load ended)
frame2 = loader.load(shape=(1, 1, 112, 112), dtype=np.int16)

# Reset to start from the beginning
loader.reset()
```

### Example 4: Loading Data from Image File

```python
from dataset_loader import DatasetLoader
import numpy as np

# Create an Image loader for a single image
loader = DatasetLoader.create_loader('img', file_path='input_image.jpg')

# Load image in NCHW format (batch, channels, height, width)
image_nchw = loader.load(shape=(1, 3, 224, 224), dtype=np.float32, format="NCHW")

# Load image in NHWC format (batch, height, width, channels)
image_nhwc = loader.load(shape=(1, 224, 224, 3), dtype=np.float32, format="NHWC")

# Create an Image loader for batch processing
loader = DatasetLoader.create_loader('img', file_path=('image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg'))

# Load a batch of images in NCHW format
batch_nchw = loader.load(shape=(4, 3, 224, 224), dtype=np.float32, format="NCHW")

# Reset the loader to start from the first image again
loader.reset()

# Check how many images are remaining
remaining = loader.get_remaining_items()
```

## Error Handling

All loaders include robust error handling:

- File not found errors for NPZ, BIN, and Image loaders
- Shape and dtype validation for NPZ loader
- Insufficient data checks for BIN loader
- Format validation for Image loader
- Unsupported loader type errors for the factory class

## Creating Your Own Custom Loader

You can extend the dataset_loader module by creating your own custom loader. Here's a step-by-step guide:

### 1. Create a Loader Class

Create a new Python file (e.g., `custom_loader.py`) with a class that implements the loader interface:

```python
import numpy as np
from typing import Tuple, Any

class CustomLoader:
    """
    Custom loader implementation.
    """
    def __init__(self, file_path: str, **kwargs):
        """
        Initialize the custom loader.
        
        Args:
            file_path (str): Path to the data file
            **kwargs: Additional arguments specific to this loader
        """
        self.file_path = file_path
        # Initialize any other required attributes
        self.some_attribute = kwargs.get('some_attribute', default_value)
        
        # Perform any necessary setup
        self._setup()
    
    def _setup(self):
        """
        Perform setup operations (e.g., loading data, initializing state).
        """
        # Implementation specific to your loader
        pass
    
    def load(self, shape: Tuple[int, ...], dtype: np.dtype, **kwargs) -> np.ndarray:
        """
        Load data with the specified shape and dtype.
        
        Args:
            shape (Tuple[int, ...]): Shape of the data to load
            dtype (np.dtype): Data type of the data to load
            **kwargs: Additional arguments specific to this load operation
            
        Returns:
            np.ndarray: The loaded data with the specified shape and dtype
        """
        # Implementation specific to your loader
        # Must return a numpy array with the specified shape and dtype
        pass
    
    def reset(self):
        """
        Reset the loader state (optional but recommended).
        """
        # Implementation specific to your loader
        pass
    
    # Additional helper methods as needed
```

### 2. Integrate with the DatasetLoader Factory

Modify the `dataset_loader.py` file to include your custom loader:

```python
# Add import for your custom loader
from custom_loader import CustomLoader

class DatasetLoader:
    """
    Factory class to create appropriate dataset loaders.
    """
    @staticmethod
    def create_loader(loader_type: str, **kwargs):
        """
        Create a dataset loader of the specified type.
        """
        loader_type = loader_type.strip()
 
        if loader_type.lower() == 'random':
            return Randomloader()
        
        elif loader_type.lower() == 'npz':
            required_args = ['file_path']
            for arg in required_args:
                if arg not in kwargs:
                    raise ValueError(f"Missing required argument '{arg}' for NPZLoader")
            
            return NPZloader(kwargs['file_path'])
        
        elif loader_type.lower() == 'bin':
            required_args = ['file_path']
            for arg in required_args:
                if arg not in kwargs:
                    raise ValueError(f"Missing required argument '{arg}' for BINloader")
            
            return BINloader(kwargs['file_path'])
        
        elif loader_type.lower() == 'img':
            required_args = ['file_path']
            for arg in required_args:
                if arg not in kwargs:
                    raise ValueError(f"Missing required argument '{arg}' for Imageloader")
            
            return Imageloader(kwargs['file_path'])
        
        # Add your custom loader
        elif loader_type.lower() == 'custom':
            required_args = ['file_path']
            for arg in required_args:
                if arg not in kwargs:
                    raise ValueError(f"Missing required argument '{arg}' for CustomLoader")
            
            return CustomLoader(kwargs['file_path'], **kwargs)
        
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")
```

### 3. Best Practices for Custom Loaders

1. **Consistent Interface**: Implement the same methods as the existing loaders (`load`, `reset`, etc.)
2. **Error Handling**: Include robust error checking for file existence, data shape, etc.
3. **Documentation**: Add clear docstrings explaining parameters and behavior
4. **Performance**: Consider memory and processing efficiency, especially for large datasets
5. **State Management**: Maintain proper state (e.g., current position) and provide methods to reset it
