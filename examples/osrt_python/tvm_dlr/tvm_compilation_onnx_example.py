import os
import sys
import argparse

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

sys.path.append(parent)
from common_utils import *
from model_configs import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-offload", dest="offload", action="store_false", help="do not offload to TIDL"
)
parser.add_argument(
    "--num_bits",
    dest="num_bits",
    default=8,
    choices=[8, 16, 32],
    help="number of bits used for quantization (use 32 for float-mode TIDL subgraphs)",
)
parser.add_argument(
    "--num_subgraphs",
    dest="num_subgraphs_max",
    default=16,
    type=int,
    help="maximum number of TIDL subgraphs for offload (actual number of subgraphs may be less that this)",
)
parser.add_argument(
    "--pc-inference",
    dest="device",
    action="store_false",
    help="compile for inference on PC",
)
parser.add_argument(
    "--num_calib_images",
    dest="calib_iters",
    default=4,
    type=int,
    help="number of images to use for calibration",
)
parser.add_argument(
    "-z", "--run_model_zoo", action="store_true", help="Run model zoo models"
)
args = parser.parse_args()

model_id = "cl-dlr-onnx_mobilenetv2"
if args.run_model_zoo:
    model_id = "cl-3090_tvmdlr_imagenet1k_torchvision_mobilenet_v2_tv_onnx"
download_model(models_configs, model_id)

# model specifics
model_path = models_configs[model_id]["session"]["model_path"]
model_input_name = "input.1Net_IN"
model_input_shape = (1, 3, 224, 224)
model_input_dtype = "float32"
model_layout = "NCHW"
model_output_directory = artifacts_folder + model_id

# TIDL compiler specifics
# We are compiling the model for J7 device using
# a compiler distributed with SDK 7.0
DEVICE = os.environ["SOC"]
SDK_VERSION = (7, 0)

# convert the model to relay IR format
import onnx
from tvm import relay

print(model_path)
onnx_model = onnx.load(model_path)
mod, params = relay.frontend.from_onnx(
    onnx_model, shape={model_input_name: model_input_shape}
)

if args.device:
    build_target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
    cross_cc_args = {
        "cc": os.path.join(
            os.environ["ARM64_GCC_PATH"], "bin", "aarch64-none-linux-gnu-gcc"
        )
    }
    model_output_directory = model_output_directory + "_device"
else:
    build_target = "llvm"
    cross_cc_args = {}
model_output_directory = model_output_directory + "/artifacts"
# image preprocessing for calibration
def preprocess_for_onnx_mobilenetv2(image_path):
    import cv2
    import numpy as np

    # read the image using openCV
    img = cv2.imread(image_path)

    # convert to RGB
    img = img[:, :, ::-1]

    # Most of the onnx models are trained using
    # 224x224 images. The general rule of thumb
    # is to scale the input image while preserving
    # the original aspect ratio so that the
    # short edge is 256 pixels, and then
    # center-crop the scaled image to 224x224
    orig_height, orig_width, _ = img.shape
    short_edge = min(img.shape[:2])
    new_height = (orig_height * 256) // short_edge
    new_width = (orig_width * 256) // short_edge
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    startx = new_width // 2 - (224 // 2)
    starty = new_height // 2 - (224 // 2)
    img = img[starty : starty + 224, startx : startx + 224]

    # apply scaling and mean subtraction.
    # if your model is built with an input
    # normalization layer, then you might
    # need to skip this
    img = img.astype("uint8")
    # for mean, scale, ch in zip([123.675, 116.28, 103.53], [0.017125, 0.017507, 0.017429], range(img.shape[2])):
    #        img[:,:,ch] = ((img.astype('float32')[:,:,ch] - mean) * scale)

    # convert HWC to NCHW
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    config = models_configs[model_id]
    # "download_model" function does pre-processing to absorb actual mean/scale into the model, 
    # so update mean/scale to 0/1 respectively before generating yaml file for inference
    config["session"]["input_mean"] = [0, 0, 0]
    config["session"]["input_scale"] = [1, 1, 1]

    gen_param_yaml(model_output_directory, config, 224, 224)
    return img


# create the directory if not present
# clear the directory
os.makedirs(model_output_directory, exist_ok=True)
for root, dirs, files in os.walk(model_output_directory, topdown=False):
    [os.remove(os.path.join(root, f)) for f in files]
    [os.rmdir(os.path.join(root, d)) for d in dirs]

if args.offload:
    from tvm.relay.backend.contrib import tidl

    assert args.num_bits in [8, 16, 32]
    assert args.num_subgraphs_max <= 16

    # Use advanced calibration for 8-bit quantization
    # Use simple calibration for 16-bit quantization and float-mode
    advanced_options = {
        8: {
            "calibration_iterations": 10,
            # below options are set to default values, include here for reference
            "quantization_scale_type": 0,
            "high_resolution_optimization": 0,
            "pre_batchnorm_fold": 1,
            # below options are only overwritable at accuracy level 9, otherwise ignored
            "activation_clipping": 1,
            "weight_clipping": 1,
            "bias_calibration": 1,
            "channel_wise_quantization": 0,
        },
        16: {
            "calibration_iterations": 1,
        },
        32: {
            "calibration_iterations": 1,
        },
    }
    calib_files = [
        "../../../test_data/airshow.jpg",
        "../../../test_data/ADE_val_00001801.jpg",
    ]

    calib_images = list(map(preprocess_for_onnx_mobilenetv2, calib_files))
    calib_input_list = [{model_input_name: img} for img in calib_images]

    # Create the TIDL compiler with appropriate parameters
    compiler = tidl.TIDLCompiler(
        DEVICE,
        SDK_VERSION,
        tidl_tools_path=os.environ["TIDL_TOOLS_PATH"],
        artifacts_folder=model_output_directory,
        tensor_bits=args.num_bits,
        debug_level=0,
        max_num_subgraphs=args.num_subgraphs_max,
        c7x_codegen=0,
        accuracy_level=(1 if args.num_bits == 8 else 0),
        advanced_options=advanced_options[args.num_bits],
    )

    # partition the graph into TIDL operations and TVM operations
    mod, status = compiler.enable(mod, params, calib_input_list)

    # build the relay module into deployables
    with tidl.build_config(tidl_compiler=compiler):
        graph, lib, params = relay.build_module.build(
            mod, target=build_target, params=params
        )

    # remove nodes / params not needed for inference
    tidl.remove_tidl_params(params)
else:
    import tvm

    # build the relay module into deployables
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, target=build_target, params=params)

# save the deployables
path_lib = os.path.join(model_output_directory, "deploy_lib.so")
path_graph = os.path.join(model_output_directory, "deploy_graph.json")
path_params = os.path.join(model_output_directory, "deploy_params.params")

lib.export_library(path_lib, **cross_cc_args)
with open(path_graph, "w") as fo:
    fo.write(graph)
with open(path_params, "wb") as fo:
    fo.write(relay.save_param_dict(params))
