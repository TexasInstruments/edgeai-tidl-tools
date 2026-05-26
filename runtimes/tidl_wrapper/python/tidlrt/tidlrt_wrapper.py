import tidlruntime
import os
import sys
import numpy as np

class TIDLRT:
    """
    This class provides simple apis to import/infer models for TIDL using
    onnxruntime interface
    """

    @staticmethod
    def get_np_type_from_tidl_type(tidl_type):
        if(tidl_type == tidlruntime.TIDL_SinglePrecFloat):
            return np.float32
        elif(tidl_type == tidlruntime.TIDL_UnsignedChar):
            return np.uint8
        elif(tidl_type == tidlruntime.TIDL_SignedChar):
            return np.int8
        elif(tidl_type == tidlruntime.TIDL_UnsignedShort):
            return np.uint16
        elif(tidl_type == tidlruntime.TIDL_SignedShort):
            return np.int16
        elif(tidl_type == tidlruntime.TIDL_UnsignedWord):
            return np.uint32
        elif(tidl_type == tidlruntime.TIDL_SignedWord):
            return np.int32
        elif(tidl_type == tidlruntime.TIDL_UnsignedDoubleWord):
            return np.uint64
        elif(tidl_type == tidlruntime.TIDL_SignedDoubleWord):
            return np.int64
        else:
            print("[WARN] Could not determine numpy type from tidl type. Returning float.")
            return np.float32

    def __init__(self, model_path : str):
        """
        Initializes a new TIDLRT object.

        Args:
            model_path (str): Path to model.
        """
        self.is_import = False
        self._import_created = False
        self._infer_created = False
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_path = model_path

        if not self.model_path.strip().endswith('.onnx'):
            raise ValueError("[ERROR] tidlruntime currently only supports onnx models.")

    def create_import(self, options : dict = None):
        """
        Create import session

        Args:
            options (dict): Dictionary containing import options            
        """
        if self._import_created:
            return self.interpreter

        self.is_import = True
        options["inputNetFile"] = self.model_path
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
        'graph_time': Total TIDL graph processing time (ms)
        'read_total': Total DDR Read bytes [X for x86 runs]
        'write_total': Total DDR Write bytes [X for x86 runs]
        'total': Total DDR Read+Write bytes [X for x86 runs]

        Returns:
            dict: performance_name : (performance_value, unit)
        """
        if not self.interpreter:
            return {}
    
        perf = self.interpreter.get_performance()

        proc_time = (perf["ts:proc_end"] - perf["ts:proc_start"])
        cp_in_time = (perf["ts:copy_in_end"] - perf["ts:copy_in_start"])
        cp_out_time = (perf["ts:copy_out_end"] - perf["ts:copy_out_start"])

        total_time = (perf['ts:run_end'] - perf['ts:run_start'])
        read_total = (perf['ddr:read_end'] - perf['ddr:read_start'])
        write_total = (perf['ddr:write_end'] - perf['ddr:write_start'])

        copy_time = cp_in_time + cp_out_time

        # change units
        total_time = total_time/1000000         # Conveting to miliseconds
        copy_time = copy_time/1000000   # Conveting to miliseconds
        proc_time = proc_time/1000000           # Conveting to miliseconds
   
        core_time = total_time - copy_time

        stats = {'total_time':      (total_time,"ms"),
                 'core_time':       (core_time,"ms"),
                 'graph_time':      (proc_time,"ms"),
                 'read_total':      (read_total, "bytes"),
                 'write_total':     (write_total, "bytes"),
                 'total':           (read_total+write_total, "bytes")
                }

        return stats

    def dump_info(self):
        """
        Prints detailed information about the model and its tensors.
        
        This method displays model path, input/output tensor counts, and detailed 
        information about each tensor including name, type, shape, and size.
        """
        #print(f"Model Path        = {self.model_path}")
        print(f"Number of Inputs  = {len(self.input_details)}")
        for i, info in enumerate(self.input_details):
            print(f"INPUT [{i}]:")
            print(f"  Name        = {info.name}")
            print(f"  Type        = {info.type}")
            print(f"  Shape       = {info.shape}")
            print(f"  Num Dims    = {len(info.shape)}")
            try:
                num_elements = np.prod(info.shape)
                print(f"  Num Elem    = {num_elements}")
            except:
                pass
            if hasattr(info, 'pad'):
                print(f"  Pad Channel = {info.pad[0]}")
                print(f"  Pad Top     = {info.pad[1]}")
                print(f"  Pad Bottom  = {info.pad[2]}")
                print(f"  Pad Left    = {info.pad[3]}")
                print(f"  Pad Right   = {info.pad[4]}")


        print(f"Number of Outputs = {len(self.output_details)}")
        for i, info in enumerate(self.output_details):
            print(f"OUTPUT [{i}]:")
            print(f"  Name        = {info.name}")
            print(f"  Type        = {info.type}")
            print(f"  Shape       = {info.shape}")
            print(f"  Num Dims    = {len(info.shape)}")
            try:
                num_elements = np.prod(info.shape)
                print(f"  Num Elem    = {num_elements}")
            except:
                pass
            if hasattr(info, 'pad'):
                print(f"  Pad Channel = {info.pad[0]}")
                print(f"  Pad Top     = {info.pad[1]}")
                print(f"  Pad Bottom  = {info.pad[2]}")
                print(f"  Pad Left    = {info.pad[3]}")
                print(f"  Pad Right   = {info.pad[4]}")


    def _create_interpreter(self, options):
        if not options:
            print("[ERROR] Runtime options not provided.")
            return None

        if self.is_import:
            self.interpreter = tidlruntime.CompileSession(options)
            self.output_details = []
        else:
            self.interpreter = tidlruntime.InferenceSession(options)
            self.output_details = self.interpreter.get_output_details()

        self.input_details = self.interpreter.get_input_details()
        return self.interpreter
    
    def _run(self, input : dict, output_keys : list = None):
        outputs = self.interpreter.run(input)
        output_keys = output_keys or [info.name for info in self.output_details]
        output_dict = {}
        for key in output_keys:
            if key in outputs:
                output_dict[key] = outputs[key]    
        return output_dict