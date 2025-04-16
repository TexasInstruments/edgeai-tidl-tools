# Copyright (c) 2018-2024, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import platform
import numpy as np
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import yaml
import shutil
import json
from config_utils import *
import copy


if platform.machine() == "aarch64":
    numImages = 100
else:
    import requests
    import onnx

    numImages = 3

    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(os.path.dirname(current))

    sys.path.append(parent)
    try:
        from osrt_model_tools.tflite_tools import tflite_model_opt as tflOpt
    except ImportError:
        pass
    try:
        from osrt_model_tools.onnx_tools.tidl_onnx_model_utils import (
            onnx_model_opt as onnxOpt,
        )
    except ImportError:
        pass

    from caffe2onnx.src.load_save_model import loadcaffemodel, saveonnxmodel
    from caffe2onnx.src.caffe2onnx import Caffe2Onnx
    from caffe2onnx.src.args_parser import parse_args
    from caffe2onnx.src.utils import freeze

artifacts_folder = "../../../model-artifacts"
output_images_folder = "../../../output_images/"
output_binary_folder = "../../../output_binaries/"

tensor_bits = 8
debug_level = 0
max_num_subgraphs = 16
accuracy_level = 1
calibration_frames = 2
calibration_iterations = 5
output_feature_16bit_names_list = ""  # "conv1_2, fire9/concat_1"
params_16bit_names_list = ""  # "fire3/squeeze1x1_2"
mixed_precision_factor = -1
quantization_scale_type = 0
high_resolution_optimization = 0
pre_batchnorm_fold = 1
inference_mode = 0
num_cores = 1
c7x_codegen = 0
ti_internal_nc_flag = 1601

data_convert = 3
SOC = os.environ["SOC"]
if quantization_scale_type == 3:
    data_convert = 0

# set to default accuracy_level 1
activation_clipping = 1
weight_clipping = 1
bias_calibration = 1
channel_wise_quantization = 0

tidl_tools_path = os.environ.get("TIDL_TOOLS_PATH")
# custom_layers_list_name = "125"
# enable_custom_layers = 1
optional_options = {
    # "priority":0,
    # delay in ms
    # "max_pre_empt_delay":10
    "platform": "J7",
    "version": "7.2",
    "tensor_bits": tensor_bits,
    "debug_level": debug_level,
    "max_num_subgraphs": max_num_subgraphs,
    "deny_list": "",  # "MaxPool"
    "deny_list:layer_type": "",
    "deny_list:layer_name": "",
    "model_type": "",  # OD
    "accuracy_level": accuracy_level,
    "advanced_options:calibration_frames": calibration_frames,
    "advanced_options:calibration_iterations": calibration_iterations,
    "advanced_options:output_feature_16bit_names_list": output_feature_16bit_names_list,
    "advanced_options:params_16bit_names_list": params_16bit_names_list,
    "advanced_options:mixed_precision_factor": mixed_precision_factor,
    "advanced_options:quantization_scale_type": quantization_scale_type,
    # "advanced_options:enable_custom_layers":enable_custom_layers,
    # "advanced_options:custom_layers_list_name":custom_layers_list_name
    # "object_detection:meta_layers_names_list" : meta_layers_names_list,  -- read from models_configs dictionary below
    # "object_detection:meta_arch_type" : meta_arch_type,                  -- read from models_configs dictionary below
    "advanced_options:high_resolution_optimization": high_resolution_optimization,
    "advanced_options:pre_batchnorm_fold": pre_batchnorm_fold,
    "ti_internal_nc_flag": ti_internal_nc_flag,
    # below options will be read only if accuracy_level = 9, else will be discarded.... for accuracy_level = 0/1, these are preset internally
    "advanced_options:activation_clipping": activation_clipping,
    "advanced_options:weight_clipping": weight_clipping,
    "advanced_options:bias_calibration": bias_calibration,
    "advanced_options:add_data_convert_ops": data_convert,
    "advanced_options:channel_wise_quantization": channel_wise_quantization,
    # Advanced options for SOC 'am69a'
    "advanced_options:inference_mode": inference_mode,
    "advanced_options:num_cores": num_cores,
    # TVM specific options
    'advanced_options:c7x_codegen': c7x_codegen
}

modelzoo_path = "../../../../../../jacinto-ai-modelzoo/models"
modelforest_path = "../../../../../../jacinto-ai-modelforest/models"

lables = "../../../test_data/labels.txt"
models_base_path = "../../../models/public/"
model_artifacts_base_path = "../../../model-artifacts/"


def get_dataset_info(task_type, num_classes):
        categories = [dict(id=catagory_id+1, supercategory=task_type, name=f"category_{catagory_id+1}") for catagory_id in range(num_classes)]
        dataset_info = dict(info=dict(description=f'{task_type} dataset'),
                            categories=categories,
                            color_map=get_color_palette(num_classes))
        return dataset_info

def append_inputs_for_batch(frame_idx, input_image_list, input_details):
    start_index = frame_idx % len(input_image_list)
    batch = input_details[0]['shape'][0]

    input_images = []
    # For batch processing different images are needed for a single input
    for j in range(batch):
        input_images.append(input_image_list[(start_index + j) % len(input_image_list)])
    return input_images

def preprocess_input(image_files, config, input_details, model_type):
    input_dict = {}
    for input_detail in input_details:
        input_name = input_detail['name']
        floating_model = input_detail['type'] == "tensor(float)"
        if model_type == 'tflite':
            floating_model = str(input_detail['type'].__name__) == "float32"
        # Map N, C, H, W to correct values based on framework
        batch = input_detail['shape'][0]
        channel = input_detail['shape'][1]
        height = input_detail['shape'][2]
        width = input_detail['shape'][3]
        if model_type == "tflite":
            batch = input_detail['shape'][0]
            height = input_detail['shape'][1]
            width = input_detail['shape'][2]
            channel = input_detail['shape'][3]

        imgs = []
        # Prepare data assuming NCHW format
        shape = [batch, channel, height, width]
        input_data = np.zeros(shape)
        for i in range(batch):
            imgs.append(
                Image.open(image_files[i])
                .convert("RGB")
                .resize((width, height), PIL.Image.LANCZOS)
            )
            temp_input_data = np.expand_dims(imgs[i], axis=0)
            temp_input_data = np.transpose(temp_input_data, (0, 3, 1, 2))
            input_data[i] = temp_input_data[0]
        if floating_model:
            input_data = np.float32(input_data)
            for mean, scale, ch in zip(
                config["session"]["input_mean"],
                config["session"]["input_scale"],
                range(input_data.shape[1]),
            ):
                input_data[:, ch, :, :] = (input_data[:, ch, :, :] - mean) * scale
        else:
            input_data = np.uint8(input_data)
            config["session"]["input_mean"] = [0, 0, 0]
            config["session"]["input_scale"] = [1, 1, 1]
        
        # Update to NHWC after data preparation for TfLite
        if model_type == "tflite":
            input_data = np.transpose(input_data, (0, 2, 3, 1))
        input_dict[input_name] = input_data
    return input_dict, imgs

def get_tensor_details(model_type, details_type, model_path):
    if details_type not in ['input', 'output']:
        details_type = 'input'
        print("Tensor details type not specified correctly - assuming 'input' ")
    
    if model_type == 'onnx':
        import onnxruntime
        sess_options = onnxruntime.SessionOptions()
        ep_list = ['CPUExecutionProvider']
        interpreter = onnxruntime.InferenceSession(model_path, providers=ep_list,
                        provider_options=[{}], sess_options=sess_options)
        if details_type == 'input':
            model_tensor_details = interpreter.get_inputs()
        else:
            model_tensor_details = interpreter.get_outputs()
        del interpreter
    elif model_type == 'tflite':
        import tflite_runtime.interpreter as tflitert_interpreter
        interpreter = tflitert_interpreter.Interpreter(model_path)
        if details_type == 'input':
            model_tensor_details = interpreter.get_input_details()
        else:
            model_tensor_details = interpreter.get_output_details()
        del interpreter

    # Maps property name from framework to a common property name, e.g. 'dtype' in tflite is mapped to 'type'
    properties_mapping = {'name':'name', 'shape':'shape', 'dtype':'type', 'type':'type', 'index':'index'}
    tensor_details = []
    for tensor_d in model_tensor_details:
        tensor_dict = {}
        for p_key, p_val in properties_mapping.items():
            if (model_type == 'onnx' and hasattr(tensor_d, p_key)):
                tensor_d_val = getattr(tensor_d, p_key)
                if p_key == 'shape':
                    tensor_d_val = list(tensor_d_val)
                tensor_dict[p_val] = tensor_d_val
            elif (model_type == 'tflite' and p_key in tensor_d):
                tensor_d_val = tensor_d[p_key]
                if p_key == 'shape':
                    tensor_d_val = [int(val) for val in tensor_d_val]
                tensor_dict[p_val] = tensor_d_val

        tensor_dict_to_append = {"name": tensor_dict['name'], "shape": tensor_dict['shape'], "type": tensor_dict['type']}
        tensor_details.append(tensor_dict_to_append)
    return tensor_details


def gen_param_yaml(artifacts_folder_path, config_orig, model_type):
    config = copy.deepcopy(config_orig)
    layout = config["preprocess"]["data_layout"]
    if model_type == "tflite":
        layout = "NHWC"
        config["preprocess"]["data_layout"] = layout

    model_path = config["session"]["model_path"]
    model_name = model_path.split("/")[-1]
    config["preprocess"]["add_flip_image"] = False
    config["session"]["artifacts_folder"] = "artifacts"
    config["session"]["model_path"] = "model/" + model_name
    config["session"]["input_data_layout"] = layout
    config["session"]["target_device"] = SOC.upper()

    if config["task_type"] == "detection":
        if config["extra_info"]["label_offset_type"] == "80to90":
            config["postprocess"]["label_offset_pred"] = coco_det_label_offset_80to90(
                label_offset=config["extra_info"]["label_offset"]
            )
        else:
            config["postprocess"]["label_offset_pred"] = coco_det_label_offset_90to90(
                label_offset=config["extra_info"]["label_offset"]
            )

    if isinstance(config["preprocess"]["crop"], int):
        config["preprocess"]["crop"] = (
            config["preprocess"]["crop"],
            config["preprocess"]["crop"],
        )
    if isinstance(config["preprocess"]["resize"], int):
        config["preprocess"]["resize"] = (
            config["preprocess"]["resize"],
            config["preprocess"]["resize"],
        )
    
    for detail in config["session"]["input_details"]:
        detail['type'] = str(detail['type'])

    for detail in config["session"]["output_details"]:
        detail['type'] = str(detail['type'])

    param_dict = pretty_object(config)
    param_dict.pop("source")
    param_dict.pop("extra_info")
    dataset_info = get_dataset_info(config["task_type"], config["extra_info"]["num_classes"])

    artifacts_model_path = "/".join(artifacts_folder_path.split("/")[:-1])
    artifacts_model_path_yaml = os.path.join(artifacts_model_path, "param.yaml")
    with open(artifacts_model_path_yaml, "w") as file:
        yaml.safe_dump(param_dict, file, sort_keys=False)
    dataset_path_yaml = os.path.join(artifacts_model_path, "dataset.yaml")
    with open(dataset_path_yaml, "w") as dataset_fp:
        yaml.safe_dump(dataset_info, dataset_fp, sort_keys=False)
    del config

headers = {
    "User-Agent": "My User Agent 1.0",
    "From": "aid@ti.com",  # This is another valid field
}


def get_url_from_link_file(url):
    if url.endswith(".link"):
        r = requests.get(url, allow_redirects=True, headers=headers)
        url = r.content.rstrip()
    return url


def download_model(models_configs, model_name):

    model_artifacts_path = model_artifacts_base_path + model_name + "/model"
    if not os.path.isdir(model_artifacts_path):
        os.makedirs(model_artifacts_path)

    if model_name in models_configs.keys():
        model_path = models_configs[model_name]["session"]["model_path"]
        if "source" in models_configs[model_name].keys():
            model_source = models_configs[model_name]["source"]
            if not os.path.isfile(model_path):
                # Check whether the specified path exists or not
                if not os.path.exists(os.path.dirname(model_path)):
                    # Create a new directory because it does not exist
                    os.makedirs(os.path.dirname(model_path))
                if (
                    "original_model_type"
                    in models_configs[model_name]["extra_info"].keys()
                ) and models_configs[model_name]["extra_info"][
                    "original_model_type"
                ] == "caffe":
                    print("Downloading  ", model_source["prototext"])
                    r = requests.get(
                        get_url_from_link_file(model_source["model_url"]),
                        allow_redirects=True,
                        headers=headers,
                    )
                    open(model_source["prototext"], "wb").write(r.content)
                    print("Downloading  ", model_source["caffe_model"])
                    r = requests.get(
                        get_url_from_link_file(model_source["caffe_model_url"]),
                        allow_redirects=True,
                        headers=headers,
                    )
                    open(model_source["caffe_model"], "wb").write(r.content)

                    graph, params = loadcaffemodel(
                        model_source["prototext"], model_source["caffe_model"]
                    )
                    c2o = Caffe2Onnx(graph, params, model_path)
                    onnxmodel = c2o.createOnnxModel()
                    freeze(onnxmodel)
                    saveonnxmodel(onnxmodel, model_path)

                else:
                    print("Downloading  ", model_path)
                    r = requests.get(
                        get_url_from_link_file(model_source["model_url"]),
                        allow_redirects=True,
                        headers=headers,
                    )
                    open(model_path, "wb").write(r.content)

                filename = os.path.splitext(model_path)
                abs_path = os.path.realpath(model_path)
                input_mean = models_configs[model_name]["session"]["input_mean"]
                input_scale = models_configs[model_name]["session"]["input_scale"]

                if models_configs[model_name]["session"]["input_optimization"] == True:
                    if filename[-1] == ".onnx":
                        onnxOpt.tidlOnnxModelOptimize(
                            abs_path, abs_path, input_scale, input_mean
                        )
                    elif filename[-1] == ".tflite":
                        tflOpt.tidlTfliteModelOptimize(
                            abs_path, abs_path, input_scale, input_mean
                        )
                if (filename[-1] == ".onnx") and (
                    models_configs[model_name]["source"]["infer_shape"] == True
                ):
                    onnx.shape_inference.infer_shapes_path(model_path, model_path)

            if "meta_layers_names_list" in models_configs[model_name]["session"].keys():
                meta_layers_names_list = models_configs[model_name]["session"][
                    "meta_layers_names_list"
                ]
                if meta_layers_names_list is not None and not os.path.isfile(
                    meta_layers_names_list
                ):
                    print("Downloading  ", meta_layers_names_list)
                    r = requests.get(
                        get_url_from_link_file(model_source["meta_arch_url"]),
                        allow_redirects=True,
                        headers=headers,
                    )
                    open(meta_layers_names_list, "wb").write(r.content)
                shutil.copy(meta_layers_names_list, model_artifacts_path)
        shutil.copy(model_path, model_artifacts_path)
    else:
        print(
            f"{model_name} ot found in availbale list of model configs - {models_configs.keys()}"
        )


def load_labels(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]


def get_class_labels(output, org_image_rgb):
    output = np.squeeze(np.float32(output))
    source_img = org_image_rgb.convert("RGBA")
    draw = ImageDraw.Draw(source_img)

    outputoffset = 0 if (output.shape[0] == 1001) else 1
    top_k = output.argsort()[-5:][::-1]
    labels = load_labels(lables)
    for j, k in enumerate(top_k):
        curr_class = f"\n  {j}  {output[k]:08.6f}  {labels[k+outputoffset]} \n"
        classes = classes + curr_class if ("classes" in locals()) else curr_class
    draw.text((0, 0), classes, fill="red")
    source_img = source_img.convert("RGB")
    classes = classes.replace("\n", ",")
    return (classes, source_img)


colors_list = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 64, 0),
    (64, 255, 0),
    (64, 0, 255),
    (255, 255, 64),
    (64, 255, 255),
    (255, 64, 255),
    (196, 128, 0),
    (128, 196, 0),
    (128, 0, 196),
    (196, 196, 128),
    (128, 196, 196),
    (196, 128, 196),
    (64, 128, 0),
    (128, 64, 0),
    (128, 0, 64),
    (196, 0, 0),
    (196, 64, 64),
    (64, 196, 64),
    (64, 255, 64),
    (64, 64, 255),
    (255, 64, 64),
    (128, 255, 128),
    (128, 128, 255),
    (255, 128, 128),
    (196, 64, 196),
    (196, 196, 64),
    (64, 196, 196),
    (196, 255, 196),
    (196, 196, 255),
    (196, 196, 128),
]


def mask_transform(inp):
    colors = np.asarray(colors_list)
    inp = np.squeeze(inp)
    colorimg = np.zeros((inp.shape[0], inp.shape[1], 3), dtype=np.float32)
    height, width = inp.shape
    inp = np.rint(inp)
    inp = inp.astype(np.uint8)
    for y in range(height):
        for x in range(width):
            if inp[y][x] < 22:
                colorimg[y][x] = colors[inp[y][x]]
    inp = colorimg.astype(np.uint8)
    return inp


def RGB2YUV(rgb):
    m = np.array(
        [
            [0.29900, -0.16874, 0.50000],
            [0.58700, -0.33126, -0.41869],
            [0.11400, 0.50000, -0.08131],
        ]
    )
    yuv = np.dot(rgb, m)
    yuv[:, :, 1:] += 128.0
    rgb = np.clip(yuv, 0.0, 255.0)
    return yuv


def YUV2RGB(yuv):
    m = np.array(
        [
            [1.0, 1.0, 1.0],
            [-0.000007154783816076815, -0.3441331386566162, 2.0320025777816772],
            [1.14019975662231445, -0.5811380310058594, 0.00001542569043522235],
        ]
    )
    yuv[:, :, 1:] -= 128.0
    rgb = np.dot(yuv, m)
    rgb = np.clip(rgb, 0.0, 255.0)

    return rgb


def seg_mask_overlay(output_data, org_image_rgb):
    classes = ""
    output_data = np.squeeze(output_data)
    if output_data.ndim > 2:
        output_data = output_data.argmax(axis=2)
        # output_data = output_data.argmax(axis=0) #segformer
    output_data = np.squeeze(output_data)
    mask_image_rgb = mask_transform(output_data)
    org_image = RGB2YUV(org_image_rgb)
    mask_image = RGB2YUV(mask_image_rgb)

    org_image[:, :, 1] = mask_image[:, :, 1]
    org_image[:, :, 2] = mask_image[:, :, 2]
    blend_image = YUV2RGB(org_image)
    blend_image = blend_image.astype(np.uint8)
    blend_image = Image.fromarray(blend_image).convert("RGB")

    return (classes, blend_image)


def det_box_overlay(outputs, org_image_rgb, od_type, framework=None):
    classes = ""
    source_img = org_image_rgb.convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    # mmdet
    if framework == "MMDetection":
        outputs = [np.squeeze(output_i) for output_i in outputs]
        if len(outputs[0].shape) == 2:
            num_boxes = int(outputs[0].shape[0])
            for i in range(num_boxes):
                if outputs[0][i][4] > 0.3:
                    xmin = outputs[0][i][0]
                    ymin = outputs[0][i][1]
                    xmax = outputs[0][i][2]
                    ymax = outputs[0][i][3]
                    print(outputs[1][i])
                    draw.rectangle(
                        ((int(xmin), int(ymin)), (int(xmax), int(ymax))),
                        outline=colors_list[int(outputs[1][i]) % len(colors_list)],
                        width=2,
                    )
        elif len(outputs[0].shape) == 1:
            num_boxes = 1
            for i in range(num_boxes):
                if outputs[i][4] > 0.3:
                    xmin = outputs[i][0]
                    ymin = outputs[i][1]
                    xmax = outputs[i][2]
                    ymax = outputs[i][3]
                    draw.rectangle(
                        ((int(xmin), int(ymin)), (int(xmax), int(ymax))),
                        outline=colors_list[int(outputs[1]) % len(colors_list)],
                        width=2,
                    )
    # SSD
    elif od_type == "SSD":
        outputs = [np.squeeze(output_i) for output_i in outputs]
        num_boxes = int(outputs[0].shape[0])
        for i in range(num_boxes):
            if outputs[2][i] > 0.3:
                xmin = outputs[0][i][0]
                ymin = outputs[0][i][1]
                xmax = outputs[0][i][2]
                ymax = outputs[0][i][3]
                draw.rectangle(
                    (
                        (int(xmin * source_img.width), int(ymin * source_img.height)),
                        (int(xmax * source_img.width), int(ymax * source_img.height)),
                    ),
                    outline=colors_list[int(outputs[1][i]) % len(colors_list)],
                    width=2,
                )
    # yolov5
    elif od_type == "YoloV5":
        outputs = [np.squeeze(output_i) for output_i in outputs]
        num_boxes = int(outputs[0].shape[0])
        for i in range(num_boxes):
            if outputs[0][i][4] > 0.3:
                xmin = outputs[0][i][0]
                ymin = outputs[0][i][1]
                xmax = outputs[0][i][2]
                ymax = outputs[0][i][3]
                draw.rectangle(
                    ((int(xmin), int(ymin)), (int(xmax), int(ymax))),
                    outline=colors_list[int(outputs[0][i][5]) % len(colors_list)],
                    width=2,
                )

    elif (
        od_type == "HasDetectionPostProcLayer"
    ):  # model has detection post processing layer
        for i in range(int(outputs[3][0])):
            if outputs[2][0][i] > 0.1:
                ymin = outputs[0][0][i][0]
                xmin = outputs[0][0][i][1]
                ymax = outputs[0][0][i][2]
                xmax = outputs[0][0][i][3]
                draw.rectangle(
                    (
                        (int(xmin * source_img.width), int(ymin * source_img.height)),
                        (int(xmax * source_img.width), int(ymax * source_img.height)),
                    ),
                    outline=colors_list[int(outputs[1][0][i]) % len(colors_list)],
                    width=2,
                )
    elif (
        od_type == "EfficientDetLite"
    ):  # model does not have detection post processing layer
        for i in range(int(outputs[0].shape[1])):
            if outputs[0][0][i][5] > 0.3:
                ymin = outputs[0][0][i][1]
                xmin = outputs[0][0][i][2]
                ymax = outputs[0][0][i][3]
                xmax = outputs[0][0][i][4]
                print(outputs[0][0][i][6])
                draw.rectangle(
                    ((int(xmin), int(ymin)), (int(xmax), int(ymax))),
                    outline=colors_list[int(outputs[0][0][i][6]) % len(colors_list)],
                    width=2,
                )

    source_img = source_img.convert("RGB")
    return (classes, source_img)
