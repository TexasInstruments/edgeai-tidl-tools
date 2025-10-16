import os
import sys
curr_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curr_dir)

from pre_process_basic import PreProcessBasic

class PreProcess:
    """
    Factory class for appropriate pre processing.
    """
    @staticmethod
    def create_pre_process(pre_process_type: str, **kwargs):
        """
        Create a pre process of the specified type.
        
        Args:
            pre_process_type (str): Type of pre process to create ('basic')
        
        Returns:
            The created pre process class based on pre_process_type
            
        Raises:
            ValueError: If the pre_process_type is not supported
        """
        pre_process_type = pre_process_type.strip()
 
        if pre_process_type.lower() == 'basic':    
            if 'params' in kwargs:
                return PreProcessBasic(kwargs['params'])
            else:
                return PreProcessBasic()

        else:
            raise ValueError(f"Unsupported pre processing type: {pre_process_type}")
