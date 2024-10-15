import tflite_runtime.interpreter as tflite
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

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

sys.path.append(parent)
from common_utils import *
from model_configs import *
from common import postprocess_utils as formatter_transform

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
    "-m",
    "--models",
    action="append",
    default=[],
    help="Model name to be added to the list to run",
)
parser.add_argument(
    "-n", "--ncpus", type=int, default=None, help="Number of threads to spawn"
)
args = parser.parse_args()
os.environ["TIDL_RT_PERFSTATS"] = "1"

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

def get_benchmark_output(interpreter):
    '''
    Returns benchmark data

    :param interpreter: Runtime session
    :return: Copy time
    :return: Total time
    :return: Processing time
    :return: Write time
    :return: Read time
    '''
    benchmark_dict = interpreter.get_TI_benchmark_data()
    proc_time = copy_time = 0
    cp_in_time = cp_out_time = 0
    subgraphIds = []
    for stat in benchmark_dict.keys():
        if "proc_start" in stat:
            subgraphIds.append(int(re.sub("[^0-9]", "", stat)))
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

    write_total = benchmark_dict["ddr:read_end"] - benchmark_dict["ddr:read_start"]
    read_total = benchmark_dict["ddr:write_end"] - benchmark_dict["ddr:write_start"]
    totaltime = benchmark_dict["ts:run_end"] - benchmark_dict["ts:run_start"]

    copy_time = copy_time if len(subgraphIds) == 1 else 0
    return copy_time, totaltime, proc_time, write_total / 1000000, read_total / 1000000


def infer_image(interpreter, image_files, config):
    '''
    Invoke the runtime session

    :param interpreter: Runtime session
    :param image_files: List of input image filename
    :param config: Configuration dictionary
    :return: Input Images
    :return: Output tensors
    :return: Total Processing time
    :return: Subgraphs Processing time
    :return: DDR write time
    :return: DDR read time
    :return: Height of input tensor
    :return: Width of input tensor
    '''

    # Get input details from the session
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model = input_details[0]["dtype"] == np.float32
    batch = input_details[0]["shape"][0]
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]
    channel = input_details[0]["shape"][3]
    new_height = height
    new_width = width
    imgs = []
    shape = [batch, new_height, new_width, channel]

    # Prepare the input data
    input_data = np.zeros(shape)
    for i in range(batch):
        imgs.append(
            Image.open(image_files[i])
            .convert("RGB")
            .resize((new_width, new_height), PIL.Image.LANCZOS)
        )
        temp_input_data = np.expand_dims(imgs[i], axis=0)
        input_data[i] = temp_input_data[0]
    if floating_model:
        input_data = np.float32(input_data)
        for mean, scale, ch in zip(
            config["session"]["input_mean"],
            config["session"]["input_scale"],
            range(input_data.shape[3]),
        ):
            input_data[:, :, :, ch] = (input_data[:, :, :, ch] - mean) * scale
    else:
        input_data = np.uint8(input_data)
        config["session"]["input_mean"] = [0, 0, 0]
        config["session"]["input_scale"] = [1, 1, 1]

    # Allocate and set tensors
    interpreter.resize_tensor_input(input_details[0]["index"], [batch, new_height, new_width, channel])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], input_data)

    # Invoke the session
    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    if SOC != "am62":
        (
            copy_time,
            proc_time,
            sub_graphs_proc_time,
            ddr_write,
            ddr_read,
        ) = get_benchmark_output(interpreter)
        proc_time = proc_time - copy_time
    else:
        copy_time = proc_time = sub_graphs_proc_time = ddr_write = ddr_read = 0
        proc_time = (stop_time - start_time) * 1000000000

    # Get output tensors
    outputs = [
        interpreter.get_tensor(output_detail["index"])
        for output_detail in output_details
    ]

    return (
        imgs,
        outputs,
        proc_time,
        sub_graphs_proc_time,
        ddr_write,
        ddr_read,
        new_height,
        new_width,
    )


def run_model(model, mIdx):
    '''
    Run a single model

    :param model: Name of the model
    :param mIdx: Run number
    '''
    print("\nRunning_Model : ", model)
    if platform.machine() != "aarch64":
        download_model(models_configs, model)

    config = models_configs[model]

    # Set input images
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
    if "optional_options" in config:
        delegate_options.update(config["optional_options"])

    delegate_options["artifacts_folder"] = (
        delegate_options["artifacts_folder"] + "/" + model + "/artifacts"
    )

    if config["task_type"] == "detection":
        delegate_options["object_detection:meta_layers_names_list"] = config["session"].get("meta_layers_names_list", "")
        delegate_options["object_detection:meta_arch_type"] = config["session"].get("meta_arch_type", -1)

    if ("object_detection:confidence_threshold" in config and "object_detection:top_k" in config):
        delegate_options["object_detection:confidence_threshold"] = config["object_detection:confidence_threshold"]
        delegate_options["object_detection:top_k"] = config["object_detection:top_k"]

    # Create/Cleanup artifacts_folder
    if args.compile or args.disable_offload:
        os.makedirs(delegate_options["artifacts_folder"], exist_ok=True)
        for root, dirs, files in os.walk(
            delegate_options["artifacts_folder"], topdown=False
        ):
            [os.remove(os.path.join(root, f)) for f in files]
            [os.rmdir(os.path.join(root, d)) for d in dirs]

    if args.compile == True:
        input_image = calib_images
    else:
        input_image = test_images

    numFrames = config["extra_info"]["num_images"]
    if args.compile:
        if numFrames > delegate_options["advanced_options:calibration_frames"]:
            numFrames = delegate_options["advanced_options:calibration_frames"]

    # Create the Inference Session
    if args.disable_offload:
        interpreter = tflite.Interpreter(
            model_path=config["session"]["model_path"], num_threads=2
        )
    elif args.compile:
        interpreter = tflite.Interpreter(
            model_path=config["session"]["model_path"],
            experimental_delegates=[
                tflite.load_delegate(
                    os.path.join(tidl_tools_path, "tidl_model_import_tflite.so"),
                    delegate_options,
                )
            ],
        )
    else:
        interpreter = tflite.Interpreter(
            model_path=config["session"]["model_path"],
            experimental_delegates=[
                tflite.load_delegate("libtidl_tfl_delegate.so", delegate_options)
            ],
        )

    # Adding input_details and output_details to configuration
    input_details = interpreter.get_input_details()
    input_name = input_details[0]["name"]
    type = "tensor(float)"
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]
    channel = input_details[0]["shape"][3]
    batch = input_details[0]["shape"][0]
    shape = [int(batch), int(channel), int(height), int(width)]
    input_details = {"name": input_name, "shape": shape, "type": type}

    output_details = interpreter.get_output_details()
    output_name = output_details[0]["name"]
    type = "tensor(float)"
    num_class = output_details[0]["shape"][1]
    batch = output_details[0]["shape"][0]
    shape = [int(batch), int(num_class)]
    output_details = {"name": input_name, "shape": shape, "type": type}

    config["session"]["input_details"] = [input_details]
    config["session"]["output_details"] = [output_details]

    # Set the formatter for post-processing
    if "formatter" in config["postprocess"]:
        formatter = config["postprocess"]["formatter"]
        if isinstance(formatter, str):
            formatter_name = formatter
            formatter = getattr(formatter_transform, formatter_name)()
        elif isinstance(formatter, dict) and "type" in formatter:
            formatter_name = formatter.pop("type")
            formatter = getattr(formatter_transform, formatter_name)(**formatter)
        config["postprocess"]["formatter"] = formatter

    for i in range(numFrames):
        start_index = i % len(input_image)
        input_details = interpreter.get_input_details()
        batch = input_details[0]["shape"][0]
        input_images = []

        # For batch processing different images are needed for a single input
        for j in range(batch):
            input_images.append(input_image[(start_index + j) % len(input_image)])
        
        # Invoke the session
        (
            imgs,
            output,
            proc_time,
            sub_graph_time,
            ddr_write,
            ddr_read,
            new_height,
            new_width,
        ) = infer_image(
            interpreter, input_images, config
        )

        total_proc_time = (
            total_proc_time + proc_time
            if ("total_proc_time" in locals())
            else proc_time
        )
        sub_graphs_time = (
            sub_graphs_time + sub_graph_time
            if ("sub_graphs_time" in locals())
            else sub_graph_time
        )
        total_ddr_write = (
            total_ddr_write + ddr_write
            if ("total_ddr_write" in locals())
            else ddr_write
        )
        total_ddr_read = (
            total_ddr_read + ddr_read if ("total_ddr_read" in locals()) else ddr_read
        )
    total_proc_time = total_proc_time / 1000000
    sub_graphs_time = sub_graphs_time / 1000000

    # Post-Processing for inference
    output_image_file_name = "py_out_" + model + "_" + os.path.basename(input_image[i % len(input_image)])
    output_bin_file_name = output_image_file_name.replace(".jpg", "") + ".bin"
    if args.compile == False:
        images = []
        output_tensors = []
        if config["task_type"] == "classification":
            for j in range(batch):
                classes, image = get_class_labels(output[0][j], imgs[j])
                images.append(image)
                output_tensors.append(
                    np.array(output[0][j], dtype=np.float32).flatten()
                )
                print("\n", classes)
        elif config["task_type"] == "detection":
            for j in range(batch):
                classes, image = det_box_overlay(
                    output, imgs[j], config["extra_info"]["od_type"]
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
            delegate_options["artifacts_folder"], config, int(new_height), int(new_width)
        )

    log = f"\n \nCompleted_Model : {mIdx+1:5d}, Name : {model:50s}, Total time : {total_proc_time/(i+1):10.2f}, Offload Time : {sub_graphs_time/(i+1):10.2f} , DDR RW MBs : {(total_ddr_write+total_ddr_read)/(i+1):10.2f}, Output Image File : {output_image_file_name}, Output Bin File : {output_bin_file_name}\n \n "  # {classes} \n \n'
    print(log)
    if ncpus > 1:
        sem.release()

if len(args.models) > 0:
    models = args.models
else:
    models = [
        "cl-tfl-mobilenet_v1_1.0_224",
        "ss-tfl-deeplabv3_mnv2_ade20k_float",
        "od-tfl-ssd_mobilenet_v2_300_float",
        "od-tfl-ssdlite_mobiledet_dsp_320x320_coco",
    ]
    if SOC == "am69a":
        # Model to demonstrate multi core parallel batch processing
        models.append("cl-tfl-mobilenetv2_4batch")

        models.append("ss-tfl-deeplabv3_mnv2_ade20k_float_low_latency")

if args.run_model_zoo:
    models = [
        "cl-0000_tflitert_imagenet1k_mlperf_mobilenet_v1_1.0_224_tflite",
        "od-2020_tflitert_coco_tf1-models_ssdlite_mobiledet_dsp_320x320_coco_20200519_tflite",
        "ss-2580_tflitert_ade20k32_mlperf_deeplabv3_mnv2_ade20k32_float_tflite",
    ]
log = f"Running {len(models)} Models - {models}\n"
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
