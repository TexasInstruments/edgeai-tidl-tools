# import onnxruntime as rt
import time
import os
import sys
import numpy as np
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import argparse
import re
import multiprocessing
import platform
import shutil

# set the environment variable before importing TVM so that
# it takes effect while loading and initializing the C++ library.
os.environ["TIDL_RT_PERFSTATS"] = "1"
import tvm
from tvm.contrib import graph_executor as runtime

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

sys.path.append(parent)
from common_utils import *
from model_configs import *

from config_utils import postprocess_utils as formatter_transform

mutex_lock = multiprocessing.Lock()

model_optimizer_found = False
if platform.machine() != "aarch64":
    try:
        from osrt_model_tools.onnx_tools.tidl_onnx_model_optimizer import optimize

        model_optimizer_found = True
    except ModuleNotFoundError as e:
        print("Skipping import of model optimizer")

required_options = {
    "tidl_tools_path": tidl_tools_path,
    "artifacts_folder": artifacts_folder,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--compile", action="store_true", help="Run in Model compilation mode"
)
parser.add_argument(
    "-d", "--disable_offload", action="store_true", help="Disable offload to TIDL"
)
parser.add_argument(
    "-z", "--run_model_zoo", action="store_true", help="Run model zoo models"
)
parser.add_argument(
    "-o",
    "--graph_optimize",
    action="store_true",
    help="Run ONNX model optimization thourgh onnx-graph-surgeon-tidl",
)
parser.add_argument(
    "-m",
    "--models",
    action="append",
    default=[],
    help="Model name to be added to the list to run",
)
parser.add_argument(
    "-n", "--ncpus", type=int, default=None, help="Number of threads to spawn"
)
parser.add_argument(
    "-cp",
    "--compile_for_platform",
    action="store",
    nargs='*',
    choices=['pc', 'evm'],
    default=['pc', 'evm'],
    help="Add a platform to compile the model for e.g. 'pc', 'evm'",
)

args = parser.parse_args()

calib_images = [
    "../../../test_data/airshow.jpg",
    "../../../test_data/ADE_val_00001801.jpg",
]
class_test_images = ["../../../test_data/airshow.jpg"]
od_test_images = ["../../../test_data/ADE_val_00001801.jpg"]
seg_test_images = ["../../../test_data/ADE_val_00001801.jpg"]

# Initialize semaphore for multi-threading
sem = multiprocessing.Semaphore(0)
if platform.machine() == "aarch64":
    ncpus = 1
else:
    if args.ncpus and args.ncpus > 0 and args.ncpus < os.cpu_count():
        ncpus = args.ncpus
    else:
        ncpus = os.cpu_count()

idx = 0
nthreads = 0
run_count = 0

if "SOC" in os.environ:
    SOC = os.environ["SOC"]
else:
    print("Please export SOC var to proceed")
    exit(-1)

# Enforce compilation on x86 only
if platform.machine() == "aarch64" and args.compile == True:
    print(
        "Compilation of models is only supported on x86 machine \n\
        Please do the compilation on PC and copy artifacts for running on TIDL devices "
    )
    exit(-1)

# Disable compilation and offload for AM62 (ARM only analytics)
if SOC == "am62":
    args.disable_offload = True
    args.compile = False

if args.compile == True and tidl_tools_path == None:
    print("TIDL_TOOLS_PATH is not set" )
    exit(-1)

def get_benchmark_output(sess):
    '''
    Returns benchmark data

    :param interpreter: Runtime session
    :return: Copy time
    :return: Processing time
    :return: Total time
    :return: Total ddr bandwidth
    '''
    benchmark_dict = sess.get_TI_benchmark_data()
    proc_time = copy_time = 0
    cp_in_time = cp_out_time = 0
    totaltime = 0
    subgraphIds = []
    for stat in benchmark_dict.keys():
        if "proc_start" in stat:
            value = stat.split("ts:subgraph_")
            value = value[1].split("_proc_start")
            subgraphIds.append(value[0])

    for i in range(len(subgraphIds)):
        proc_time += (
            benchmark_dict["ts:subgraph_" + str(subgraphIds[i]) + "_proc_end"]
            - benchmark_dict["ts:subgraph_" + str(subgraphIds[i]) + "_proc_start"]
        )
        cp_in_time += (
            benchmark_dict["ts:subgraph_" + str(subgraphIds[i]) + "_copy_in_end"]
            - benchmark_dict["ts:subgraph_" + str(subgraphIds[i]) + "_copy_in_start"]
        )
        cp_out_time += (
            benchmark_dict["ts:subgraph_" + str(subgraphIds[i]) + "_copy_out_end"]
            - benchmark_dict["ts:subgraph_" + str(subgraphIds[i]) + "_copy_out_start"]
        )
        copy_time += cp_in_time + cp_out_time
    copy_time = copy_time if len(subgraphIds) == 1 else 0
    totaltime = benchmark_dict["ts:run_end"] - benchmark_dict["ts:run_start"]

    ddr_read_total = benchmark_dict['ddr:read_end'] - benchmark_dict['ddr:read_start']
    ddr_write_total = benchmark_dict['ddr:write_end'] - benchmark_dict['ddr:write_start']

    ddr_bw = ddr_read_total + ddr_write_total

    return copy_time, proc_time, totaltime, ddr_bw

def infer_image(sess, input_dict):
    '''
    Invoke the runtime session

    :param sess: Runtime session
    :param input_dict: Dictionary with input name and data
    :return: Output tensors
    :return: Total Processing time
    :return: Subgraphs Processing time
    '''

    # Invoke session for inference
    start_time = time.time()

    # feed input data
    for key, value in input_dict.items():
        sess.set_input(key, value)
    sess.run()
    output = []
    for i in range(sess.get_num_outputs()):
        output.append(sess.get_output(i).asnumpy())

    stop_time = time.time()
    infer_time = stop_time - start_time

    copy_time, sub_graphs_proc_time, totaltime, ddr_bw = get_benchmark_output(sess)
    proc_time = totaltime - copy_time

    return output, proc_time, sub_graphs_proc_time, ddr_bw

def run_model(model, mIdx):
    '''
    Run a single model

    :param model: Name of the model
    :param mIdx: Run number
    '''
    try:
        print("\nRunning_Model : ", model, " \n")
        if platform.machine() != "aarch64":
            mutex_lock.acquire()
            download_model(models_configs, model)
            mutex_lock.release()

        config = models_configs[model]

        ### TVM - determine model framework (onnx)
        model_type = os.path.splitext(config["session"]["model_path"])[1][1:]
        if model_type not in ['onnx']:
            raise Exception(f"ERROR processing model - {model} - Only onnx models are supported")

        # Run graph optimization
        if args.graph_optimize:
            if model_optimizer_found:
                if (args.compile or args.disable_offload) and (
                    platform.machine() != "aarch64"
                ):
                    copy_path = config["session"]["model_path"][:-5] + "_org.onnx"
                    # Check if copy path exists and prompt for permission to overwrite
                    if os.path.isfile(copy_path):
                        overwrite_permission = input(
                            f"\033[96mThe file {copy_path} exists, do you want to overwrite? [Y/n] \033[00m"
                        )
                        if overwrite_permission != "Y":
                            print("Aborting run...")
                            sys.exit(-1)
                        else:
                            print(
                                f"\033[93m[WARNING] File {copy_path} will be overwritten\033[00m"
                            )

                    shutil.copy2(config["session"]["model_path"], copy_path)
                    print(
                        f"\033[93mOptimization Enabled: Moving {config['model_path']} to {copy_path} before overwriting by optimization\033[00m"
                    )
                    optimize(
                        model=config["session"]["model_path"], out_model=config["session"]["model_path"]
                    )
                else:
                    print(
                        "Model optimization is only supported in compilation or disabled offload mode on x86 machines"
                    )
            else:
                print("Model optimizer not found, -o flag has no effect")

        # Set input images
        config = models_configs[model]
        if config["task_type"] == "classification":
            test_images = class_test_images
        elif config["task_type"] == "detection":
            test_images = od_test_images
        elif config["task_type"] == "segmentation":
            test_images = seg_test_images
        
        # Set delegate options 
        delegate_options = {}
        delegate_options.update(required_options)
        delegate_options.update(optional_options)
        delegate_options.update(config.get("runtime_options", {}))

        if config["task_type"] == "detection":
            delegate_options["object_detection:meta_layers_names_list"] = config["session"].get("meta_layers_names_list", "")
            delegate_options["object_detection:meta_arch_type"] = config["session"].get("meta_arch_type", -1)

        input_details = get_tensor_details(model_type, 'input', config["session"]["model_path"])
        output_details = get_tensor_details(model_type, 'output', config["session"]["model_path"])
        # Adding input_details and output_details to configuration
        config["session"]["input_details"] = input_details
        config["session"]["output_details"] = output_details

        num_frames = config["extra_info"]["num_images"]

        # Set the formatter for post-processing
        if "postprocess" in config and "formatter" in config["postprocess"]:
            formatter = config["postprocess"]["formatter"]
            if isinstance(formatter, str):
                formatter_name = formatter
                formatter = getattr(formatter_transform, formatter_name)()
            elif isinstance(formatter, dict) and "type" in formatter:
                formatter_name = formatter.pop("type")
                formatter = getattr(formatter_transform, formatter_name)(**formatter)
            config["postprocess"]["formatter"] = formatter
        
        ## Shape inference for ONNX models
        if args.compile == True:
            #### Shape inference - required for ONNX models #######
            if model_type == 'onnx':
                import onnx
                mutex_lock.acquire()
                log = f'\nRunning shape inference on model {config["session"]["model_path"]} \n'
                print(log)
                onnx.shape_inference.infer_shapes_path(
                    config["session"]["model_path"], config["session"]["model_path"]
                )
                mutex_lock.release()
        else:
            # For inference, use current platform
            if platform.machine() != "aarch64":
                args.compile_for_platform = ['pc']
            else:
                args.compile_for_platform = ['evm']
        
        ####### Run for all platforms sequentially ############
        delegate_options_orig = copy.deepcopy(delegate_options)

        for device_platform in args.compile_for_platform:
            delegate_options = copy.deepcopy(delegate_options_orig)
            ### TVM - different artifact directories for x86 and target device
            artifacts_folder_pc = delegate_options["artifacts_folder"] + "/" + model + "/artifacts"
            artifacts_folder_evm = delegate_options["artifacts_folder"] + "/" + model + "_device/artifacts"
            if device_platform == "pc":
                delegate_options["artifacts_folder"] = artifacts_folder_pc
            else:
                delegate_options["artifacts_folder"] = artifacts_folder_evm
            
            # Create/Cleanup artifacts_folder
            if args.compile:
                print(f"\n\n******* Compiling for device platform -- {device_platform} ***********\n\n")
                os.makedirs(delegate_options["artifacts_folder"], exist_ok=True)                
                if not os.environ.get("REUSE_TIDL_ARTIFACTS"): # Do not clean if want to reuse TIDL artifacts
                    for root, dirs, files in os.walk(
                        delegate_options["artifacts_folder"], topdown=False
                    ):
                        [os.remove(os.path.join(root, f)) for f in files]
                        [os.rmdir(os.path.join(root, d)) for d in dirs]
            
            # Reuse PC TIDL artifacts for EVM
            if args.compile_for_platform == ['pc', 'evm']:
                if device_platform == 'evm':
                    # Running for both PC and EVM in that order - so PC run is assumed done
                    try:
                        shutil.copytree(artifacts_folder_pc, artifacts_folder_evm, dirs_exist_ok=True)
                        os.environ["REUSE_TIDL_ARTIFACTS"] = '1'
                    except shutil.Error as e:
                        print(f"Error copying folder {e}")
                        print("Not reusing TIDL artifacts from PC run")
                else:
                    # Avoid carrying over the env variable set from previous model
                    os.environ.pop("REUSE_TIDL_ARTIFACTS", None)
            
            ## Compilation
            if args.compile:
                input_image = calib_images
                from tvm.contrib import tidl            
                ############ Model compilation call #################
                
                # Default values for no TIDL offload
                calib_data_list = []

                if not args.disable_offload:
                    # Offload to TIDL enabled - perform 2 actions
                        # 1. Create input list for TIDL calibration
                        # 2. Set enable_tidl_offload flag to True
                    
                    # 1. Create input list for TIDL calibration
                    if num_frames > delegate_options["advanced_options:calibration_frames"]:
                        num_frames = delegate_options["advanced_options:calibration_frames"]
                    for i in range(num_frames):
                        input_images = append_inputs_for_batch(i, input_image, input_details)        
                        input_dict, _  = preprocess_input(input_images, config, input_details, model_type)
                        calib_data_list.append(input_dict)
                ### TVM - compile_model call does compilation and stores artifacts in artifacts_folder
                status = tidl.compile_model(
                                            platform = os.environ["SOC"],
                                            compile_for_device = (True if (device_platform == 'evm') else False),
                                            enable_tidl_offload = (not args.disable_offload),
                                            delegate_options = delegate_options,
                                            calibration_input_list = calib_data_list,
                                            model_path = config["session"]["model_path"],
                                            input_shape_dict = {inp_d['name']: inp_d['shape'] for inp_d in input_details}
                                            ### Optional arguments: This API does model to Relay conversion internally, however it can be overridden using already converted IR module and params 
                                            # mod = mod,   # Input Relay IR module.
                                            # params = params   # The parameter dict used by Relay.
                                            )
                if status == False:
                    raise Exception(f"Model - {model} - compilation failed")
            else:   ## Model inference
                input_image = test_images

                loaded_json = open(delegate_options['artifacts_folder'] + "/deploy_graph.json").read()
                loaded_lib = tvm.runtime.load_module(delegate_options['artifacts_folder'] + "/deploy_lib.so","so")
                loaded_params = bytearray(open(delegate_options['artifacts_folder'] + "/deploy_param.params", "rb").read())

                # create a runtime executor module
                sess = runtime.create(loaded_json, loaded_lib, tvm.cpu())
                sess.load_params(loaded_params)
                
                total_proc_time = 0
                sub_graphs_time = 0
                ddr_bw_total = 0
                for i in range(num_frames):
                    input_images = append_inputs_for_batch(i, input_image, input_details)        
                    input_dict, imgs = preprocess_input(input_images, config, input_details, model_type)
                    output, proc_time, sub_graph_time, ddr_bw = infer_image(sess, input_dict)

                    total_proc_time = total_proc_time + proc_time
                    sub_graphs_time = sub_graphs_time + sub_graph_time
                    ddr_bw_total = ddr_bw_total + ddr_bw

                total_proc_time = total_proc_time / 1000000  # Conveting to miliseconds
                sub_graphs_time = sub_graphs_time / 1000000  # Conveting to miliseconds
                ddr_bw_total = ddr_bw_total / 1000000        # Conveting to MB/s

                # Averaging out for number of frames
                total_proc_time = total_proc_time / num_frames
                sub_graphs_time = sub_graphs_time / num_frames
                ddr_bw_total = int(ddr_bw_total / num_frames)

                # Post-Processing for inference
                output_image_file_name = "py_out_" + model + "_" + os.path.basename(input_image[i % len(input_image)])
                output_bin_file_name = output_image_file_name.replace(".jpg", "") + ".bin"
                if args.compile == False:
                    images = []
                    output_tensors = []
                    batch = input_details[0]['shape'][0]
                    if config["task_type"] == "classification":
                        for j in range(batch):
                            classes, image = get_class_labels(output[0][j], imgs[j])
                            print("\n", classes)
                            images.append(image)
                            output_tensors.append(
                                np.array(output[0][j], dtype=np.float32).flatten()
                            )
                    elif config["task_type"] == "detection":
                        for j in range(batch):
                            classes, image = det_box_overlay(
                                output,
                                imgs[j],
                                config["extra_info"]["od_type"],
                                config["extra_info"]["framework"],
                            )
                            images.append(image)
                            output_np = np.array([], dtype=np.float32)
                            for tensor in output:
                                output_np = np.concatenate(
                                    (output_np, np.array(tensor, dtype=np.float32).flatten())
                                )
                            output_tensors.append(output_np)
                    elif config["task_type"] == "segmentation":
                        for j in range(batch):
                            imgs[j] = imgs[j].resize(
                                (output[0][j].shape[-1], output[0][j].shape[-2]), PIL.Image.LANCZOS
                            )
                            classes, image = seg_mask_overlay(output[0][j], imgs[j])
                            images.append(image)
                            output_tensors.append(
                                np.array(output[0][j], dtype=np.float32).flatten()
                            )
                    else:
                        print("\nInvalid task type ", config["task_type"])

                    # Save the output images and output tensors
                    for j in range(batch):
                        output_image_file_name = "py_out_" + model + "_" + os.path.basename(input_images[j])
                        print("\nSaving image to ", output_images_folder)
                        if not os.path.exists(output_images_folder):
                            os.makedirs(output_images_folder)
                        images[j].save(output_images_folder + output_image_file_name, "JPEG")
                        print("\nSaving output tensor to ", output_binary_folder)
                        if not os.path.exists(output_binary_folder):
                            os.makedirs(output_binary_folder)
                        output_bin_file_name = output_image_file_name.replace(".jpg", "") + ".bin"
                        output_tensors[j].tofile(output_binary_folder + output_bin_file_name)

            # Generate param.yaml after model compilation
            if args.compile or args.disable_offload:
                gen_param_yaml(
                    delegate_options["artifacts_folder"], config, model_type
                )

        if args.compile == True:
            log = f"\n \nCompleted_Model : {mIdx+1:5d}, Name : {model:50s}\n\n"
        else:
            log = f"\n \nCompleted_Model : {mIdx+1:5d}, Name : {model:50s}, Total time : {total_proc_time:10.2f}, Offload Time : {sub_graphs_time:10.2f} , DDR RW MBs : {ddr_bw_total}, Output Image File : {output_image_file_name}, Output Bin File : {output_bin_file_name}\n \n "  # {classes} \n \n'
        print(log)
    except Exception as e:
        print(f"Error processing model {model}: {str(e)}")
    finally:
        if ncpus > 1:
            sem.release()


if len(args.models) > 0:
    models = args.models
else:
    models = ["cl-tvm-ort-resnet18-v1",      # CL ONNX model
              "od-tvm-ort-ssd-lite_mobilenetv2_fpn", # OD ONNX model
              "cl-tvm-ort-resnet18-v1_c7x"     # Example model with one layer forced to generate C7x code (using "deny_list" option)
              ]   
    if SOC not in ("am62a", "am67a"):
        models.append("ss-tvm-ort-deeplabv3lite_mobilenetv2") # SS ONNX model

log = f"\nRunning {len(models)} Models - {models}\n"
print(log)

def join_one(nthreads):
    '''
    Join the thread

    :param nthreads: Thread count
    '''
    global run_count
    sem.acquire()
    run_count = run_count + 1
    return nthreads - 1

def spawn_one(models, idx, nthreads):
    '''
    Spawn a process

    :param models: Name of the model to run
    :param idx: Index
    :param nthreads: Thread count
    '''
    p = multiprocessing.Process(
        target=run_model,
        args=(
            models,
            idx,
        ),
    )
    p.start()
    return idx + 1, nthreads + 1


# Run the models using multi-processing if possible
if ncpus > 1:
    for t in range(min(len(models), ncpus)):
        idx, nthreads = spawn_one(models[idx], idx, nthreads)

    while idx < len(models):
        nthreads = join_one(nthreads)
        idx, nthreads = spawn_one(models[idx], idx, nthreads)

    for n in range(nthreads):
        nthreads = join_one(nthreads)
else:
    for mIdx, model in enumerate(models):
        run_model(model, mIdx)
