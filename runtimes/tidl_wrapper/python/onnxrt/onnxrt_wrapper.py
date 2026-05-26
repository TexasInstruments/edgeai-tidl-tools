import onnxruntime
import os
import sys
import numpy as np

class ONNXRT:
    """
    This class provides simple apis to import/infer models for TIDL using
    onnxruntime interface
    """

    @staticmethod
    def get_np_type_from_onnx_type(onnx_type):
        onnx_type = onnx_type.strip().lower()
        # [TODO] : Put more types
        if(onnx_type == 'tensor(float)'):
            return np.float32
        elif(onnx_type == 'tensor(int64)'):
            return np.int64
        elif(onnx_type == 'tensor(uint8)'):
            return np.uint8
        elif(onnx_type == 'tensor(int32)'):
            return np.int32
        elif(onnx_type == 'tensor(bool)'):
            return np.bool
        else:
            print("[WARN] Could not determine numpy type from onnx type. Returning float.")
            return np.float32

    def __init__(self, model_path : str, tidl_offload : bool = True):
        """
        Initializes a new ONNXRT object.

        Args:
            model_path (str): Path to onnx model.
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
        self.session_options = onnxruntime.SessionOptions()
        self._set_default_session_options()

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
            print(f"  Name     = {info.name}")
            print(f"  Type     = {info.type}")
            print(f"  Shape    = {info.shape}")
            print(f"  Num Dims = {len(info.shape)}")
            try:
                num_elements = np.prod(info.shape)
                print(f"  Num Elem = {num_elements}")
            except:
                pass
            
        print(f"Number of Outputs = {len(self.output_details)}")
        for i, info in enumerate(self.output_details):
            print(f"OUTPUT [{i}]:")
            print(f"  Name     = {info.name}")
            print(f"  Type     = {info.type}")
            print(f"  Shape    = {info.shape}")
            print(f"  Num Dims = {len(info.shape)}")
            try:
                num_elements = np.prod(info.shape)
                print(f"  Num Elem = {num_elements}")
            except:
                pass

    def disable_onnxruntime_optimization(self):
        """
        Disable internal onnxruntime graph optimization
        """
        self.session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    def _create_interpreter(self, options):
        if self.tidl_offload == True:
            if self.is_import:
                executioner_provider = ['TIDLCompilationProvider', 'CPUExecutionProvider']
            else:
                executioner_provider = ['TIDLExecutionProvider', 'CPUExecutionProvider']
            if not options:
                print("[ERROR] Runtime options not provided.")
                return None
            provider_options = [options, {}]
            self.interpreter = onnxruntime.InferenceSession(self.model_path, providers=executioner_provider, provider_options=provider_options, sess_options=self.session_options)
        else:
            executioner_provider = ['CPUExecutionProvider']
            provider_options = [{}]
            self.interpreter = onnxruntime.InferenceSession(self.model_path, providers=executioner_provider, provider_options=provider_options, sess_options=self.session_options)
        
        self.input_details = self.interpreter.get_inputs()
        self.output_details = self.interpreter.get_outputs()
        return self.interpreter
    
    def _run(self, input : dict, output_keys : list = None):
        outputs = list(self.interpreter.run(None, input))
        output_keys = output_keys or [info.name for info in self.output_details]
        output_dict = {output_key : output for output_key, output in zip(output_keys, outputs)}
        return output_dict
    
    def _set_default_session_options(self):
        self.session_options.log_severity_level = 3
        self.session_options.intra_op_num_threads = 1