import tflite_runtime.interpreter as tflite
import os
import sys
import numpy as np
import re

class TFLiteRT:
    """
    This class provides simple apis to import/infer models for TIDL using
    tflite_runtime interface
    """
    def __init__(self, model_path : str, tidl_offload : bool = True):
        """
        Initializes a new TFLiteRT object.

        Args:
            model_path (str): Path to tflite model.
            tidl_offload (int): Optional argument to enable/disable tidl offload. Default: True.
        """
        self.model_path = model_path
        self.tidl_offload = tidl_offload
        self.is_import = False
        self._import_created = False
        self._infer_created = False
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def create_import(self, options : dict = None):
        """
        Create import session

        Args:
            options (dict): Dictionary containing import options            
        """
        if self._import_created:
            return self.interpreter

        self.is_import = True
        self.interpreter = self._create_interpreter(options = options)
        self._import_created = True

    def run_import(self, input : dict, output_keys : list = None):
        """
        Run import

        Args:
            input (dict): Input data dictionary in {'input_name' : input_data} format
            output_keys (list): Optional list to filter output based on output name

        Returns:
            dict: Output in {'output_name' : output_data}
        """
        if not self._import_created:
            self.create_import()
        return self._run(input, output_keys)

    def create_infer(self, options : dict = None):
        """
        Create infer session

        Args:
            options (dict): Dictionary containing infer options            
        """
        if self._infer_created:
            return self.interpreter

        self.is_import = False
        self.interpreter = self._create_interpreter(options = options)
        self._infer_created = True
    
    def run_infer(self, input : dict, output_keys : list = None):
        """
        Run inference

        Args:
            input (dict): Input data dictionary in {'input_name' : input_data} format
            output_keys (list): Optional list to filter output based on output name

        Returns:
            dict: Output in {'output_name' : output_data}
        """
        if not self._infer_created:
            self.create_infer()
        return self._run(input, output_keys)

    def get_performance(self):
        """
        This method returns performance data

        'total_time': Total time taken for run (ms)
        'core_time': Total time taken barring the io copy time (ms)
        'subgraph_time': Total TIDL Subgraphs processing time (ms)
        'read_total': Total DDR Read bytes [X for x86 runs]
        'write_total': Total DDR Write bytes [X for x86 runs]

        Returns:
            dict: performance_name : (performance_value, unit)
        """
        if not self.interpreter:
            return {}
    
        benchmark_dict = self.interpreter.get_TI_benchmark_data()
        subgraph_time = copy_time = 0
        cp_in_time = cp_out_time = 0
        subgraphIds = []
        for stat in benchmark_dict.keys():
            if 'proc_start' in stat:
                value = stat.split("ts:subgraph_")
                value = value[1].split("_proc_start")
                subgraphIds.append(value[0])

        for i in range(len(subgraphIds)):
            subgraph_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_start']
            cp_in_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_start']
            cp_out_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_start']

        copy_time = cp_in_time + cp_out_time
        copy_time = copy_time if len(subgraphIds) == 1 else 0
        total_time = benchmark_dict['ts:run_end'] - benchmark_dict['ts:run_start']
        read_total = benchmark_dict['ddr:read_end'] - benchmark_dict['ddr:read_start']
        write_total = benchmark_dict['ddr:write_end'] - benchmark_dict['ddr:write_start']

        total_time = total_time / 1000000       # Conveting to miliseconds
        copy_time = copy_time / 1000000         # Conveting to miliseconds
        subgraph_time = subgraph_time / 1000000 # Conveting to miliseconds
   
        core_time = total_time - copy_time

        stats = {'total_time':      (total_time,"ms"),
                 'core_time':       (core_time,"ms"),
                 'subgraph_time':   (subgraph_time,"ms"),
                 'read_total':      (read_total, "bytes"),
                 'write_total':     (write_total, "bytes")
                }

        return stats

    def dump_info(self):
        """
        Prints detailed information about the model and its tensors.
        
        This method displays model path, input/output tensor counts, and detailed 
        information about each tensor including name, type, shape, and size.
        """
        print(f"Model Path        = {self.model_path}")
        print(f"Number of Inputs  = {len(self.input_details)}")
        for i, info in enumerate(self.input_details):
            print(f"INPUT [{i}]:")
            print(f"  Name     = {info['name']}")
            print(f"  Type     = {info['dtype']}")
            print(f"  Shape    = {info['shape']}")
            print(f"  Num Dims = {len(info['shape'])}")
            try:
                num_elements = np.prod(info['shape'])
                print(f"  Num Elem = {num_elements}")
            except:
                pass
            
        print(f"Number of Outputs = {len(self.output_details)}")
        for i, info in enumerate(self.output_details):
            print(f"OUTPUT [{i}]:")
            print(f"  Name     = {info['name']}")
            print(f"  Type     = {info['dtype']}")
            print(f"  Shape    = {info['shape']}")
            print(f"  Num Dims = {len(info['shape'])}")
            try:
                num_elements = np.prod(info['shape'])
                print(f"  Num Elem = {num_elements}")
            except:
                pass

    def _create_interpreter(self, options):
        tidl_tools_path = options.get("tidl_tools_path", None) if options else None
        
        if self.tidl_offload == True:
            if not options:
                print("[ERROR] Runtime options not provided.")
                return None
                
            if self.is_import:
                # For model compilation (import)
                delegate_path = os.path.join(tidl_tools_path, 'tidl_model_import_tflite.so') if tidl_tools_path else 'tidl_model_import_tflite.so'
                self.interpreter = tflite.Interpreter(
                    model_path=self.model_path,
                    experimental_delegates=[tflite.load_delegate(delegate_path, options)]
                )
            else:
                # For model inference
                self.interpreter = tflite.Interpreter(
                    model_path=self.model_path,
                    experimental_delegates=[tflite.load_delegate('libtidl_tfl_delegate.so', options)]
                )
        else:
            # CPU execution without TIDL offload
            self.interpreter = tflite.Interpreter(model_path=self.model_path, num_threads=1)
        
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        return self.interpreter
    
    def _run(self, input : dict, output_keys : list = None):
        # Set input tensors
        for input_name, input_data in input.items():
            input_index = None
            for detail in self.input_details:
                if detail['name'] == input_name:
                    input_index = detail['index']
                    break
            
            if input_index is not None:
                # Resize input tensor if needed
                if list(input_data.shape) != list(detail['shape']):
                    self.interpreter.resize_tensor_input(input_index, input_data.shape)
                    self.interpreter.allocate_tensors()
                
                self.interpreter.set_tensor(input_index, input_data)
            else:
                print(f"[WARNING] Input tensor '{input_name}' not found in model")
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensors
        output_dict = {}
        if output_keys is None:
            output_keys = [detail['name'] for detail in self.output_details]
        
        for detail in self.output_details:
            if detail['name'] in output_keys:
                output_dict[detail['name']] = self.interpreter.get_tensor(detail['index'])
        
        return output_dict
