import os
import sys
import platform
import argparse
import numpy as np
import yaml
import PIL
from PIL import Image
from typing import Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tidl_wrapper/python')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from dataset_loader.dataset_loader import DatasetLoader
from pre_process.pre_process import PreProcess
from post_process.post_process import PostProcess
import common_utils

def parse_and_validate_config(file : str,
                              soc : str,
                              model_filter : Tuple = None,
                              runtime_filter : Tuple = None):
    """
    Parse and validate a YAML configuration file for model execution.
    
    This function reads a YAML configuration file, validates its structure and content,
    and filters models based on provided criteria (model names, runtime types, and SoC compatibility).
    
    Args:
        file (str): Path to the YAML configuration file
        soc (str): System-on-Chip (SoC) identifier to filter compatible models
        model_filter (Tuple, optional): Tuple of model names to filter from the config
        runtime_filter (Tuple, optional): Tuple of runtime types to filter from the config
    
    Returns:
        dict: Validated and filtered configuration dictionary, or None if validation fails
    """
    if not os.path.exists(file):
        print(f"[ERROR] {file} not found")
        return None

    try:
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"[ERROR] Invalid YAML in config file: {e}")
        return None

    # Check if config file contains model to run
    if 'models' not in config:
        print(f"[ERROR] {file} does not have any models to run")
        return None

    # Determine and validate runtime for each model
    for model, info in config["models"].items():
        if "runtime" not in info:
            print(f"[ERROR] {model} does not contain 'runtime'")
            return None

        if info["runtime"] not in ["onnxrt", "tflitert", "tidlrt", "tvmrt"]:
            print(f"[ERROR] {model} has invalid runtime '{info['runtime']}'. Only 'onnxrt', 'tflitert', 'tidlrt' and 'tvmrt' are supported.")
            return None

    # Filter models by runtime if specified
    if runtime_filter:
        filtered_models = {}
        for model, info in config["models"].items():
            if info["runtime"] in runtime_filter:
                filtered_models[model] = info
        
        if not filtered_models:
            print(f"[ERROR] No models found with specified runtimes filter: {runtime_filter}")
            return None
        
        config["models"] = filtered_models

    # Filter models if arguments provided 
    if model_filter:
        filtered_models = {}
        for model, info in config["models"].items():
            if model in model_filter:
                filtered_models[model] = info
        
        if not filtered_models:
            print(f"[ERROR] No models found with specified model filter: {model_filter}")
            return None

        config["models"] = filtered_models

    # Filter models if soc is provided
    filtered_models = {}
    for model, info in config["models"].items():
        if 'soc' not in info:
            filtered_models[model] = info
            continue

        if isinstance(info["soc"], str):
            temp_list = []
            for i in info["soc"].strip().split(','):
                for j in i.strip().split(' '):
                    if j.strip() != '':
                        temp_list.append(j)
            info["soc"] = temp_list

        for i, s in enumerate(info["soc"]):
            info["soc"][i] = common_utils.get_soc(s)

        if soc in info["soc"]:
            filtered_models[model] = info

    config["models"] = filtered_models

    # Validate model specific parameters
    for model, info in config["models"].items():
        if "path" not in info:
            print(f"[ERROR] {model} does not contain 'path'")
            return None

        if not os.path.isabs(info['path']):
            info['path'] = os.path.abspath(os.path.join(os.path.dirname(file), info['path']))

        if not os.path.isfile(info['path']):
            print(f"[ERROR] {info['path']} is not a file")
            return None

        # Validate file extension matches runtime
        if info["runtime"] == "onnxrt" and not info['path'].endswith('.onnx'):
            print(f"[ERROR] {info['path']} is not a .onnx file but runtime is set to 'onnxrt'")
            return None
        elif info["runtime"] == "tflitert" and not info['path'].endswith('.tflite'):
            print(f"[ERROR] {info['path']} is not a .tflite file but runtime is set to 'tflitert'")
            return None
        if info["runtime"] == "tidlrt" and not info['path'].endswith('.onnx'):
            print(f"[ERROR] 'tidlrt' currently only supports onnx models. {info['path']} is not a .onnx file but runtime is set to 'tidlrt'")
            return None

        if "inputs" not in info:
            print(f"[WARN] {model} does not contain 'inputs', using seeded random")
            info["inputs"] = "random"

        if isinstance(info["inputs"], str):
            temp_list = []
            for i in info["inputs"].strip().split(','):
                for j in i.strip().split(' '):
                    if j.strip() != '':
                        temp_list.append(j)
            info["inputs"] = temp_list
        
        if not isinstance(info["inputs"], list):
            print(f"[ERROR] {model} : 'inputs' can only be a single input or list of inputs")
            return None
        
        for i in range(len(info["inputs"])):
            # Convert each input to a list if not already
            if not isinstance(info["inputs"][i], list):
                info["inputs"][i] = [info["inputs"][i]]

            # Convert to absolute path if input type is not random
            for j in range(len(info["inputs"][i])):
                if info["inputs"][i][j] == 'random':
                    continue
                elif not os.path.isabs(info["inputs"][i][j]):
                    info["inputs"][i][j] = os.path.abspath(os.path.join(os.path.dirname(file),info["inputs"][i][j]))

    # Extra validation for any other potential relative paths
    for model, info in config["models"].items():
        # Process compile options and resolve relative paths for applicable compilation options
        if "compile_options" in info:
            # Resolve relative path for meta_layers_names_list
            if "object_detection:meta_layers_names_list" in info["compile_options"]:
                meta_layers_path = info["compile_options"]["object_detection:meta_layers_names_list"]
                if not os.path.isabs(meta_layers_path):
                    meta_layers_path = os.path.abspath(os.path.join(os.path.dirname(file), meta_layers_path))
                info["compile_options"]["object_detection:meta_layers_names_list"] = meta_layers_path
        
        if "post_process_info" in info:
            # Resolve relative path for labels file for post-processing
            if "labels" in info["post_process_info"]:
                labels_path = info["post_process_info"]["labels"]
                if not os.path.isabs(labels_path):
                    labels_path = os.path.abspath(os.path.join(os.path.dirname(file), labels_path))
                info["post_process_info"]["labels"] = labels_path
        
    # Check global compile options and resolve relative paths for applicable compilation options
    if "compile_options" in config:
        if "object_detection:meta_layers_names_list" in config["compile_options"]:
            meta_layers_path = config["compile_options"]["object_detection:meta_layers_names_list"]
            if not os.path.isabs(meta_layers_path):
                meta_layers_path = os.path.abspath(os.path.join(os.path.dirname(file), meta_layers_path))
            config["compile_options"]["object_detection:meta_layers_names_list"] = meta_layers_path

    return config

def run(config,
        soc,
        compile : bool = False,
        tidl_tools_path : str = None,
        artifacts_base_path : str = None,
        disable_tidl_offload : bool = False,
        verbose : bool = False,
        dump_frames : int = None,
        output_base_path : str = None):
    """
    Run models based on the provided configuration.

    This function processes the configuration, sets up the environment for model execution,
    and runs the specified models either in compilation or inference mode.

    Args:
        config (dict): Run configuration dictionary
        soc (str): System-on-Chip (SoC) identifier
        compile (bool, optional): Whether to run in model compilation mode. Defaults to False.
        tidl_tools_path (str, optional): Path to TIDL tools, required for compilation
        artifacts_base_path (str, optional): Base path for model artifacts
        disable_tidl_offload (bool, optional): Whether to disable TIDL offload. Defaults to False.
        verbose (bool, optional): Whether to enable verbose output. Defaults to False.
        dump_frames (int, optional): Collect and return only the first N frame outputs. Default: all frames.
        output_base_path (str, optional): Base path to save outputs. When None, outputs are not saved to disk.

    Returns:
        tuple: (status, outputs) where status is 0 for success and outputs is a dictionary containing output data for each model
    """
    status = 0

    # Validate tidl_tools_path in case of model compilation
    if (not disable_tidl_offload) and (compile == True):
        if (tidl_tools_path == None):
            print("[ERROR] Please provide tidl_tools_path for model compilation.")
            return -1, None
        else:
            print(f"\nTIDL_TOOLS_PATH={tidl_tools_path}")
    
    # Validate artifacts_base_path in case of tidl offload
    if (not disable_tidl_offload):
        if (artifacts_base_path == None):
            print("[ERROR] Please provide artifacts_base_path.")
            return -1, None

    disable_tidl_offload_orig = disable_tidl_offload

    # Dictionary to save output data in case of inference
    outputs = {}

    # Run the models
    print(f"\nRunning {len(config['models'])} models...")
    for model in config["models"]:
        print(f"\t{model}")
    print("\n")

    for model, info in config["models"].items():

        disable_tidl_offload = disable_tidl_offload_orig

        # Create and cleanup artifacts folder in case of model compilation    
        artifacts_path = os.path.join(artifacts_base_path, model, 'artifacts')
        if (compile == True):
            try:
                os.makedirs(artifacts_path, exist_ok=True)
            except OSError as e:
                print(f"[ERROR][{model}] : Cannot create {artifacts_path} : {e}")
                continue

            for root, dirs, files in os.walk(artifacts_path, topdown=False):
                [os.remove(os.path.join(root, f)) for f in files]
                [os.rmdir(os.path.join(root, d)) for d in dirs]
        
        # Check if model artifacts is present in case of TIDL inference
        elif (compile == False) and (not disable_tidl_offload):
            if not os.path.isdir(artifacts_path):
                print(f"[ERROR][{model}] : Cannot find model-artifacts folder {artifacts_path}")
                continue

        options = {"artifacts_folder" : artifacts_path}

        # Parse compile/infer options from config.yaml
        if compile == True:
            options["tidl_tools_path"] = tidl_tools_path
            options.update(config.get("compile_options", {}))
            options.update(info.get("compile_options", {}))
        else:
            common_infer_options = config.get("infer_options", {})
            model_specific_infer_options = info.get("infer_options", {})
            options.update(config.get("infer_options", {}))
            options.update(info.get("infer_options", {}))

        # Run shape inference for ONNX model compilation
        if compile == True and info["path"].endswith("onnx"):
            import onnx
            print(f"[INFO] [{model}] : Running shape inference")
            onnx.shape_inference.infer_shapes_path(info["path"], info["path"])

        print(f"\nRunning {model} with {info['runtime']} runtime...")

        # Create session based on runtime
        # ONNXRT
        if info["runtime"] == "onnxrt":
            from onnxrt.onnxrt_wrapper import ONNXRT
            import onnxruntime
            session = ONNXRT(model_path=info["path"], tidl_offload=not disable_tidl_offload)

            '''
            Enabling onnxruntime internal optimization by default.
            For vision transformers models set disable_onnx_optimizer to true in config.yaml
            '''
            if ('disable_onnx_optimizer' in info) and (info['disable_onnx_optimizer'] == 1 or info['disable_onnx_optimizer'] == True):
                session.session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

            #In case of not TIDL offload, force to only inference
            if (disable_tidl_offload):
                compile = False

        # TFLITERT
        elif info["runtime"] == "tflitert":
            from tflitert.tflitert_wrapper import TFLiteRT
            session = TFLiteRT(model_path=info["path"], tidl_offload=not disable_tidl_offload)

            # In case of not TIDL offload, force to only inference
            if (disable_tidl_offload):
                compile = False

        # TIDLRT
        elif info["runtime"] == "tidlrt":
            from tidlrt.tidlrt_wrapper import TIDLRT
            session = TIDLRT(model_path=info["path"])
            if (disable_tidl_offload):
                print(f"[WARN] [{model}] : Cannot disable tidl offload when using tidlrt.")

        # TVMRT
        elif info["runtime"] == "tvmrt":
            from tvmrt.tvmrt_wrapper import TVMRT
            from onnxrt.onnxrt_wrapper import ONNXRT
            
            if (disable_tidl_offload) and (compile == False):
                print(f"[ERROR] {model} : TVMRT : Flag 'disable_tidl_offload' is valid only for compilation. First compile model with '--compile --disable_tidl_offload' on PC and then run inference without this flag\n")
                continue

            session = TVMRT(model_path=info["path"], tidl_offload=not disable_tidl_offload)
            if (compile == False) and not os.path.isdir(os.path.join(artifacts_path, "tempDir")):
                disable_tidl_offload = True

            options["soc"] = soc

        # Get number of fames to run for
        if "num_frames" in info:
            num_frames = info["num_frames"]
        else:
            if (compile == True) and ("advanced_options:calibration_frames" in options):
                num_frames = options["advanced_options:calibration_frames"]
            else:
                num_frames = len(info["inputs"])

        # Check number of frames against "advanced_options:calibration_frames" for compilation
        if (compile == True):
            if ("advanced_options:calibration_frames" in options):
                calib_frames = options["advanced_options:calibration_frames"]
                if calib_frames != num_frames:
                    print(f"[WARN] {model} : advanced_options:calibration_frames({calib_frames}) does not match no. of frames({num_frames}). Running for {num_frames} frames.")
                    options["advanced_options:calibration_frames"] = num_frames
            else:
                options["advanced_options:calibration_frames"] = num_frames

        # Check number of frames against inputs available
        if (len(info["inputs"]) == 1):
            info["inputs"] = [info["inputs"][0] for _ in range(num_frames)]

        elif (len(info["inputs"]) > num_frames):
            print(f"[WARN] {model} : No. of inputs and frames do not match. Running only {num_frames} frames")
            info["inputs"] = info["inputs"][:num_frames]

        elif (len(info["inputs"]) < num_frames):
            print(f"[ERROR] {model} : No. of inputs({len(info['inputs'])}) is less than no of. frames({num_frames}).")
            continue
            
        # Print session information if verbose mode is enabled
        if (verbose == True) and (not disable_tidl_offload):
            if (compile == True):
                print(f"\n[INFO][{model}] Compilation Options:")
            else:
                print(f"\n[INFO][{model}] Inference Options:")
            print(options)
            print()
        
        # Initialize import/infer
        if (compile == True):
            session.create_import(options=options)
        else:
            session.create_infer(options=options)

        # Print session information if verbose mode is enabled
        if (verbose == True):
            print(f"\n[INFO][{model}] Session information:")
            session.dump_info()
            print()

        # Initialize variables for tracking total performance across frames
        sum_performance = {}

        # Run across frames
        for i in range(num_frames):

            # Initialize input dataset loader based on type
            input_data = info["inputs"][i]
            dataset_type = None

            if input_data[0] == 'random':
                dataset_type = 'random'
            elif input_data[0].endswith('.npz'):
                dataset_type = 'npz'
            elif input_data[0].endswith('.bin'):
                dataset_type = 'bin'
            elif input_data[0].endswith('.jpg') or input_data[0].endswith('.jpeg') or input_data[0].endswith('.png'):
                dataset_type = 'img'

            if dataset_type == None:
                print(f"[ERROR] [{model}] Frame:{i} : Invalid input type. Only random, *.npz, *.bin, *.jpg, *.jpeg, *.png is allowed")
                continue

            # Create Dataset loader
            dataset_loader = DatasetLoader.create_loader(dataset_type, file_path = input_data)

            # Create post process and pre process in case of image input
            pre_process = None
            post_process = None
            if (dataset_type == 'img'):
                # Create pre processor
                pre_process = PreProcess.create_pre_process('basic', params = info.get('pre_process_info', {}))

                # Create post processor
                if (compile == False):
                    if 'post_process_info' not in info:
                        print("[WARN] post_process_info not defined in model config. Skipping post-processing.")
                    elif ('task_type' not in info['post_process_info']):
                        print("[WARN] task_type not defined in model config's post_process_info. Skipping post-processing.")
                    else:
                        task_type = info['post_process_info']['task_type'].strip().lower()
                        post_process = PostProcess.create_post_process(task_type, params = info['post_process_info'])

            # Fill input data dictionary for session run
            input_dict = {}
            input_images = []
            img_count = 0
            for j in range(len(session.input_details)):
                if info["runtime"] == "onnxrt":
                    # ONNX runtime specific handling
                    name = session.input_details[j].name
                    shape = session.input_details[j].shape
                    dtype = ONNXRT.get_np_type_from_onnx_type(session.input_details[j].type)
                    format = "NCHW"
                    pad = None
                elif info["runtime"] == "tflitert":
                    # TFLite runtime specific handling
                    detail = session.input_details[j]
                    name = detail['name']
                    shape = detail['shape']
                    dtype = detail['dtype']
                    format = "NHWC"
                    pad = None
                elif info["runtime"] == "tvmrt":
                    name = session.input_details[j].name
                    shape = session.input_details[j].shape
                    # Only Onnx models are supported with tvmrt
                    dtype = ONNXRT.get_np_type_from_onnx_type(session.input_details[j].type)
                    format = "NCHW"
                    pad = None
                else:
                    # TIDL runtime specific handling
                    name = session.input_details[j].name
                    shape = session.input_details[j].shape
                    dtype = TIDLRT.get_np_type_from_tidl_type(session.input_details[j].type)
                    format = "NCHW"
                    if (compile == True):
                        pad = None
                    else:
                        pad = session.input_details[j].pad

                # Save original input image for post-processing later
                if (dataset_type == 'img'):
                    if (img_count >= len(input_data)):
                        img_count = 0

                    if format == "NCHW":
                        height, width = shape[-2], shape[-1]
                    else:
                        height, width = shape[-3], shape[-2]
                    input_image = Image.open(input_data[img_count]).convert("RGB").resize((width, height), PIL.Image.LANCZOS)
                    input_images.append((input_data[img_count], input_image))
                    img_count += 1

                # Load the data
                data = dataset_loader.load(shape=shape, dtype=dtype, format=format)

                # Do pre-processing in images only for float inputs
                if pre_process and dtype == np.float32:
                    data = pre_process.process(input=data, format=format)

                # Handle input padding if applicable:
                if pad != None:
                    pad_values = [(0, pad[0]),(pad[1], pad[2]),(pad[3], pad[4])]
                    while len(pad_values) != len(shape):
                        if len(pad_values) < len(shape):
                            pad_values.insert(0, (0,0))
                        elif len(pad_values) > len(shape):
                            pad_values.pop()
                    data = np.pad(data, pad_values, 'constant', constant_values=(0))

                # Add data to input dictionary
                input_dict[name] = data
            
            # Run session
            if (compile == True):
                output = session.run_import(input=input_dict)
            else:
                output = session.run_infer(input=input_dict)

            # Process output data dictionary after session run
            for j in range(len(session.output_details)):
                if info["runtime"] == "onnxrt":
                    name = session.output_details[j].name
                    shape = session.output_details[j].shape
                    dtype = ONNXRT.get_np_type_from_onnx_type(session.output_details[j].type)
                    pad = None
                elif info["runtime"] == "tflitert":
                    detail = session.output_details[j]
                    name = detail['name']
                    shape = detail['shape']
                    dtype = detail['dtype']
                    pad = None
                elif info["runtime"] == "tvmrt":
                    name = session.output_details[j].name
                    shape = session.output_details[j].shape
                    dtype = ONNXRT.get_np_type_from_onnx_type(session.output_details[j].type)
                    pad = None
                else:
                    name = session.output_details[j].name
                    shape = session.output_details[j].shape
                    dtype = TIDLRT.get_np_type_from_tidl_type(session.output_details[j].type)
                    pad = session.output_details[j].pad

                if name not in output:
                    continue

                # Handle output padding if applicable:
                if pad != None:
                    pad_values = [(0, pad[0]),(pad[1], pad[2]),(pad[3], pad[4])]
                    while len(pad_values) != len(shape):
                        if len(pad_values) < len(shape):
                            pad_values.insert(0, (0,0))
                        elif len(pad_values) > len(shape):
                            pad_values.pop()

                    # Create slicing for removing padding
                    slices = []
                    for dim, (pad_before, pad_after) in enumerate(pad_values):
                        if pad_before > 0 or pad_after > 0:
                            slices.append(slice(pad_before, output[name].shape[dim] - pad_after if pad_after > 0 else None))
                        else:
                            slices.append(slice(None))

                    # Apply slicing to remove padding
                    output[name] = output[name][tuple(slices)]
            
            # Collect performance data for inference
            if (compile == False):
                    performance = session.get_performance()
                    for perf_key, (perf_val, perf_unit) in performance.items():

                        # For no-offload, only log total time
                        if disable_tidl_offload and perf_key != "total_time":
                            continue

                        if perf_key in sum_performance:
                            sum_performance[perf_key][0] += perf_val
                        else:
                            sum_performance[perf_key] = [perf_val, perf_unit]

            if model not in outputs:
                outputs[model] = []

            # Store output binaries
            if dump_frames is None or len(outputs[model]) < dump_frames:
                frame_output_binary = {}
                try:
                    for output_name, output_data in output.items():
                        output_data = np.array(output_data, dtype=np.float32)
                        frame_output_binary[output_name] = output_data
                except:
                    pass

                # Post process and store output images
                frame_post_proc_output = {}
                if (compile == False) and (post_process):
                    for j in range(len(input_images)):
                        metadata, post_processed_image = post_process.process(input_images[j][1], list(output.values()), j)
                        image_name = os.path.basename(input_images[j][0]).strip().split('.')[0]
                        frame_post_proc_output[image_name] = (metadata, post_processed_image)

                outputs[model].append((frame_output_binary, frame_post_proc_output))

        # Print average performance metrics after processing all frames
        if len(sum_performance) > 0:
            print("\n" + "="*80)
            print(f"Average performance metrics for {model} across {num_frames} frames:")
            print("-"*80)
            max_key_length = max(len(str(key)) for key in sum_performance.keys())
            for perf_key, (perf_val, perf_unit)  in sum_performance.items():
                perf_val = perf_val/num_frames
                print(f"{str(perf_key):<{max_key_length+2}}: {perf_val:.2f} {perf_unit}")
            print("="*80 + "\n")

        # Save outputs for this model if applicavle
        if output_base_path is not None and compile == False and model in outputs and len(outputs[model]) > 0:
            if disable_tidl_offload:
                output_path = os.path.join(output_base_path, model, "no_offload")
            else:
                output_path = os.path.join(output_base_path, model, "offload")

            for i, data in enumerate(outputs[model]):
                path = os.path.join(output_path, f"frame_{i+1}")
                try:
                    os.makedirs(path, exist_ok=True)
                except OSError as e:
                    print(f"[ERROR] [{model}] Frame:{i} : Cannot create directory for saving output {path} : {e}")
                    continue

                output_binaries = data[0]
                post_proc_data = data[1]

                for name, binary in output_binaries.items():
                    out_bin_file = f"{name}.bin".replace('/', '_')
                    binary.tofile(os.path.join(path, out_bin_file))

                for name, frame_data in post_proc_data.items():
                    metadata, image = frame_data
                    image.save(os.path.join(path, f"{name}.jpg"), "JPEG")
                    with open(os.path.join(path, f"{name}.txt"), 'w+') as f:
                        f.write(metadata)

            print(f"Outputs saved: {output_path}")

    return status, outputs

def main():
    """
    Main entry point for the basic example script.
    
    This function parses command line arguments, sets up the environment,
    and calls the run function with the appropriate parameters.
    It also handles saving the outputs to the specified directory.
    
    Returns:
        None
    """
    TIDL_TOOLS_PATH = os.getenv('TIDL_TOOLS_PATH')
    CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.yaml"))
    ARTIFACTS_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model-artifacts"))
    OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "outputs")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Runtime basic example with TIDL')
    parser.add_argument('-c', '--compile', action='store_true', help='Run in model compilation mode')
    parser.add_argument('-i', '--infer', action='store_true', help='Run in inference mode')
    parser.add_argument('-d','--disable_tidl_offload', action='store_true', help='Disable offload to TIDL')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-x', '--config', type=str, help='Path to config.yaml file. Default: <script_dir>/config.yaml')
    parser.add_argument('-m', '--models', nargs='*', type=str, help='Filter model keys to run from config file. Default: None')
    parser.add_argument('-r', '--runtimes', nargs='*', type=str, choices=['onnxrt', 'tflitert', 'tidlrt', 'tvmrt'],
                        help='Filter by runtime types. Default: None (run all runtimes)')
    parser.add_argument('--dump-frames', type=int, default=None, metavar='N', help='Collect and save only the first N frame outputs. Default: all frames')
    args = parser.parse_args()

    # Update config file path if provided
    if args.config:
        CONFIG_FILE = os.path.abspath(args.config)

    # Check for SOC constraints
    SOC = os.environ.get("SOC")
    if (SOC == None):
        print("[ERROR] Please set SOC environment variable")
        sys.exit(-1)

    SOC = common_utils.get_soc(SOC)
    if(SOC == "AM62"):
        args.disable_tidl_offload = True
        args.compile = False

    # Check for pltform constraints
    if platform.machine() == "aarch64" and args.compile == True:
        print("[ERROR] aarch64 currently does not support compilation. Please run compilation on x86 machine.")
        sys.exit(-1)

    # Validate parameters
    if args.compile and args.infer:
        print("[WARN] Cannot compile and infer at the same time. Running only compile.")
        args.compile = True
    if not args.infer and not args.compile:
        args.infer = True

    # Parse and validate config file
    config = parse_and_validate_config(
        file=CONFIG_FILE,
        soc=SOC,
        model_filter=args.models,
        runtime_filter=args.runtimes
    )
    
    if config is None:
        sys.exit(-1)
        
    status, outputs = run(
        config=config,
        soc=SOC,
        compile=args.compile,
        tidl_tools_path=TIDL_TOOLS_PATH,
        artifacts_base_path=ARTIFACTS_BASE,
        disable_tidl_offload=args.disable_tidl_offload,
        verbose=args.verbose,
        dump_frames=args.dump_frames,
        output_base_path=OUTPUT_BASE if not args.compile else None
    )

    if (status != 0):
        sys.exit(status)

if __name__ == "__main__":
    main()
