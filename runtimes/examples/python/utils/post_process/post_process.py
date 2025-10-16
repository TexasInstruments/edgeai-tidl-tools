import os
import sys
curr_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curr_dir)

from post_process_classification import PostProcessClassification
from post_process_detection import PostProcessDetection
from post_process_segmentation import PostProcessSegmentation

class PostProcess:
    """
    Factory class for appropriate post processing.
    """
    @staticmethod
    def create_post_process(post_process_type: str, **kwargs):
        """
        Create a post process of the specified type.
        
        Args:
            post_process_type (str): Type of post process to create ('classification', 'detection', 'segmentation')
        
        Returns:
            The created post process class based on post_process_type
            
        Raises:
            ValueError: If the post_process_type is not supported
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

        else:
            raise ValueError(f"Unsupported post processing type: {post_process_type}")
