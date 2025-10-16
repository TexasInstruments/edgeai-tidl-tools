import os
import sys
curr_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curr_dir)

from random_loader import Randomloader
from npz_loader import NPZloader
from bin_loader import BINloader
from image_loader import Imageloader


class DatasetLoader:
    """
    Factory class to create appropriate dataset loaders.
    """
    @staticmethod
    def create_loader(loader_type: str, **kwargs):
        """
        Create a dataset loader of the specified type.
        
        Args:
            loader_type (str): Type of loader to create ('random', 'npz', 'bin', 'img')
        
        Returns:
            The created dataset loader class based on loader_type
            
        Raises:
            ValueError: If the loader type is not supported
        """
        loader_type = loader_type.strip()
 
        if loader_type.lower() == 'random':
            return Randomloader()
        
        elif loader_type.lower() == 'npz':
            required_args = ['file_path']
            for arg in required_args:
                if arg not in kwargs:
                    raise ValueError(f"Missing required argument '{arg}' for NPZloader")
            
            # Ensure file_path is properly formatted for the loader
            file_path = kwargs['file_path']
            if isinstance(file_path, list):
                file_path = tuple(file_path)
            elif not isinstance(file_path, tuple):
                file_path = (file_path,)
                
            return NPZloader(file_path)
        
        elif loader_type.lower() == 'bin':
            required_args = ['file_path']
            for arg in required_args:
                if arg not in kwargs:
                    raise ValueError(f"Missing required argument '{arg}' for BINloader")
            
            # Ensure file_path is properly formatted for the loader
            file_path = kwargs['file_path']
            if isinstance(file_path, list):
                file_path = tuple(file_path)
            elif not isinstance(file_path, tuple):
                file_path = (file_path,)
                
            return BINloader(file_path)
        
        elif loader_type.lower() == 'img':
            required_args = ['file_path']
            for arg in required_args:
                if arg not in kwargs:
                    raise ValueError(f"Missing required argument '{arg}' for Imageloader")
            
            # Ensure file_path is properly formatted for the loader
            file_path = kwargs['file_path']
            if isinstance(file_path, list):
                file_path = tuple(file_path)
            elif not isinstance(file_path, tuple):
                file_path = (file_path,)
                
            return Imageloader(file_path)
        
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")
