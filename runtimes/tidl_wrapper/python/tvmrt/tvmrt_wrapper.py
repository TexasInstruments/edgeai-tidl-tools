import os
import sys
import numpy as np
import platform
import platform

# set the environment variable before importing TVM so that
# it takes effect while loading and initializing the c++ library.
os.environ["TIDL_RT_PERFSTATS"] = "1"

try:
    import tvm
except ModuleNotFoundError as e:
    print("\n" + "="*80)
    print("ERROR: TVMRT Dependencies Missing")
    print("="*80)
    if platform.machine() != "aarch64":
        print("Please follow the setup steps before running the application.")
    else:
        print(f"Failed to import required module: {e.name if hasattr(e, 'name') else 'tvm dependencies'}")
        print("\nTVMRT requires additional Python packages to function properly.")
        print("Please install the missing dependencies using the following command:")
        print("\n  pip3 install psutil typing_extensions")
        print("\nAfter installation, please re-run your application.")
    print("="*80 + "\n")
    sys.exit(1)

class TVMRT:
    """
    This class provides simple apis to import/infer models for TIDL using
    tvm_runtime interface
    """

    def __init__(self, model_path : str, tidl_offload : bool = True, target_machine : list = ['pc', 'evm']):
        """
        Initializes a new TVMRT object.

        Args:
            model_path (str): Path to onnx model.
            tidl_offload (int): Optional argument to enable/disable tidl offload. Default: True.
            target_machine (list): Platfrom to compile the model for e.g. 'pc', 'evm'
        """
        self.model_path = model_path
        self.tidl_offload = tidl_offload
        self.platform = target_machine
        self.is_import = False
        self._import_created = False
        self._infer_created = False
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.options = None
        # Only onnx model type is supported
        if not model_path.endswith("onnx"):
           raise Exception(f"[ERROR]: Invalid Model Type: {model_path.split('.')[-1]}. Only Onnx model type is supported with TVMRT!!")

    def create_import(self, options : dict = None):
        """
        Create import session

        Args:
            options (dict): Dictionary containing import options
        """
        self.is_import = True
        self.options = options
        self._get_details()
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

        if not hasattr(TVMRT.run_import, 'calib_input_list'):
            TVMRT.run_import.calib_input_list = []
            TVMRT.run_import.frame = 0

        # Compilation with TVM runtime needs list of all the calibration inputs.
        # Creating a list here, and compilation will be done when all the inputs are available
        # This is done to keep in sync with the other runtimes
        TVMRT.run_import.calib_input_list.append(input)
        TVMRT.run_import.frame += 1

        if TVMRT.run_import.frame == self.options["advanced_options:calibration_frames"]:
            self._compile(TVMRT.run_import.calib_input_list)
            # Reset for next import session
            TVMRT.run_import.calib_input_list = []
            TVMRT.run_import.frame = 0
        return []

    def create_infer(self, options : dict = None):
        """
        Create infer session

        Args:
            options (dict): Dictionary containing infer options
        """
        from tvm.contrib import graph_executor as runtime

        self.is_import = False
        self.options = options
        self._get_details()
        if platform.machine() != "aarch64":
            target_machine = 'pc'
        else:
            target_machine = 'evm'

        loaded_json = open(self.options['artifacts_folder'] + f"/deploy_graph.json.{target_machine}").read()
        loaded_lib = tvm.runtime.load_module(self.options['artifacts_folder'] + f"/deploy_lib.so.{target_machine}","so")
        loaded_params = bytearray(open(self.options['artifacts_folder'] + f"/deploy_param.params.{target_machine}", "rb").read())

        # create a runtime executor module
        sess = runtime.create(loaded_json, loaded_lib, tvm.cpu())
        sess.load_params(loaded_params)
        self.interpreter = sess
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
        'num_subgraphs': Total Detected subgraphs

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
                 'write_total':     (write_total, "bytes"),
                 'num_subgraphs':   (len(subgraphIds), "")
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

    @staticmethod
    def _create_npz_file(inputs, npz_path):
        keys = inputs[0].keys()
        stacked_dict = {}
        for key in keys:
            values = [d[key] for d in inputs]
            stacked_dict[key] = np.stack(values, axis=0)
        np.savez_compressed(npz_path, **stacked_dict)

    def _get_details(self):
        self.input_details = []
        self.output_details = []
        import onnxruntime
        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3
        ep_list = ['CPUExecutionProvider']
        interpreter = onnxruntime.InferenceSession(self.model_path, providers=ep_list,
                    provider_options=[{}], sess_options=sess_options)
        self.input_details = interpreter.get_inputs()
        self.output_details = interpreter.get_outputs()
        del interpreter

    def _compile(self, inputs):
        from tvm.contrib import tidl
        import shutil
        # the artifact files that are generated
        deploy_lib = 'deploy_lib.so'
        deploy_graph = 'deploy_graph.json'
        deploy_params = 'deploy_param.params'
        artifact_files = [deploy_lib, deploy_graph, deploy_params]

        for target_machine in self.platform:
            # Running for both PC and EVM in that order - so PC run is assumed done, enable reuse of TIDL artifacts
            if target_machine == 'evm':
                if(os.path.exists(os.path.join(self.options['artifacts_folder'], f'{deploy_lib}.pc'))):
                    print("Reusing TIDL artifacts from x86 compilation for target compilation")
                    os.environ["REUSE_TIDL_ARTIFACTS"] = '1'

            print(f"\n\n******* Compiling for target device -- {target_machine} ***********\n\n")

            if os.getenv("USE_TVMC_MODE"):
                import subprocess
                import yaml
                # TVMC takes the calib data as npz file, create a npz file from the input data
                calib_input = "calib_data.npz"
                self._create_npz_file(inputs, calib_input)

                # Create a config.yaml which has all the compile_options applicable to this model
                config_data = {
                    'compile_options': self.options
                }
                config_file = "config_tvmc.yaml"
                with open(config_file, 'w') as file:
                    yaml.dump(config_data, file, default_flow_style=False)

                cmd = [
                    sys.executable, "-m", "tvm.driver.tvmc", "compile",
                    self.model_path,
                    "--target", "tidl",
                    "--tidl-config", config_file,
                    "--tidl-calibration-input", calib_input,
                    "--enable-tidl-offload", "1" if self.tidl_offload else "0",
                    "--compile-for-device", ("1" if (target_machine == 'evm') else "0"),
                    "--c7x-codegen", str(self.options["advanced_options:c7x_codegen"]),
                    "--output", self.options["artifacts_folder"]
                ]

                print(f"\nRunning TVMC compile command:")
                print(" ".join(cmd))
                print()

                try:
                    subprocess.run(cmd, check=True, capture_output=False, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"TVMC compilation failed with return code {e.returncode}")
                    print("STDOUT:", e.stdout)
                    print("STDERR:", e.stderr)
                    raise Exception("[ERROR]: Model compilation failed")
                # Delete the created calib data file (npz)
                os.remove(calib_input)
                os.remove(config_file)
            else:
                ## TVM - compile_model call does compilation and stores artifacts in artifacts_folder
                status = tidl.compile_model(
                    platform = os.environ["SOC"],
                    compile_for_device = (True if (target_machine == 'evm') else False),
                    enable_tidl_offload = self.tidl_offload,
                    delegate_options = self.options,
                    calibration_input_list = inputs,
                    model_path = self.model_path,
                    input_shape_dict = {inp_d.name: inp_d.shape for inp_d in self.input_details}
                    ### Optional arguments: This API does model to Relay conversion internally, however it can be overridden using already converted IR module and params
                    # mod = mod,   # Input Relay IR module.
                    # params = params   # The parameter dict used by Relay.
                    )
                if status == False:
                    raise Exception("[ERROR]: Model compilation failed")

            for artifact_file in artifact_files:
                path_lib = os.path.join(self.options['artifacts_folder'], artifact_file)
                os.rename(path_lib, f'{path_lib}.{target_machine}')

            os.environ.pop("REUSE_TIDL_ARTIFACTS", None) # Clean up this env variable for next run

    def _run(self, input : dict, output_keys : list = None):
        # feed input data
        for key, value in input.items():
            self.interpreter.set_input(key, value)
        self.interpreter.run()
        output = {}
        output_keys = output_keys or [info.name for info in self.output_details]

        if(len(output_keys) != self.interpreter.get_num_outputs()):
            raise Exception("[ERROR]: Number of output mimatch!")
        #TODO: Make sure if the output is corresponding to the key
        for i in range(self.interpreter.get_num_outputs()):
            output[output_keys[i]] = self.interpreter.get_output(i).asnumpy()
        return output
