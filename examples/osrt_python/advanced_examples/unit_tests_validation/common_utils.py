import os
import sys
import platform
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import yaml
import shutil

if platform.machine() == 'aarch64':
    numImages = 100
else : 
    import requests
    import onnx
    numImages = 1

artifacts_folder = '../model-artifacts/'
output_images_folder = '../output_images/'


tensor_bits = 32  ### This script is intended to run floating point
debug_level = 0
max_num_subgraphs = 16
accuracy_level = 1
calibration_frames = 1
calibration_iterations = 1
output_feature_16bit_names_list = ""#"conv1_2, fire9/concat_1"
params_16bit_names_list = "" #"fire3/squeeze1x1_2"
mixed_precision_factor = -1

quantization_scale_type = 0
high_resolution_optimization = 0
pre_batchnorm_fold = 1
ti_internal_nc_flag = 1601

#set to default accuracy_level 1
activation_clipping = 1
weight_clipping = 1
bias_calibration = 1
channel_wise_quantization = 0

tidl_tools_path = os.environ["TIDL_TOOLS_PATH"]

optional_options = {
# "priority":0,
#delay in ms
# "max_pre_empt_delay":10
"platform":"J7",
"version":"7.2",
"tensor_bits":tensor_bits,
"debug_level":debug_level,
"max_num_subgraphs":max_num_subgraphs,
"deny_list":"", #"MaxPool",
"accuracy_level":accuracy_level,
"advanced_options:calibration_frames": calibration_frames,
"advanced_options:calibration_iterations": calibration_iterations,
"advanced_options:output_feature_16bit_names_list" : output_feature_16bit_names_list,
"advanced_options:params_16bit_names_list" : params_16bit_names_list,
"advanced_options:mixed_precision_factor" :  mixed_precision_factor,
"advanced_options:quantization_scale_type": quantization_scale_type,
#"object_detection:meta_layers_names_list" : meta_layers_names_list,  -- read from models_configs dictionary below
#"object_detection:meta_arch_type" : meta_arch_type,                  -- read from models_configs dictionary below
"advanced_options:high_resolution_optimization": high_resolution_optimization,
"advanced_options:pre_batchnorm_fold" : pre_batchnorm_fold,
"ti_internal_nc_flag" : ti_internal_nc_flag,
# below options will be read only if accuracy_level = 9, else will be discarded.... for accuracy_level = 0/1, these are preset internally
"advanced_options:activation_clipping" : activation_clipping,
"advanced_options:weight_clipping" : weight_clipping,
"advanced_options:bias_calibration" : bias_calibration,
"advanced_options:add_data_convert_ops" : 0,
"advanced_options:channel_wise_quantization" : channel_wise_quantization
}

models_base_path = '../unit_test_models'

models_configs = {
    ## tflite
    'add_const' : {
        'model_path' : os.path.join(models_base_path, 'add_const.tflite'),
        'source' : {'model_url': 'dummy', 'opt': True}, # URL irrelavant if model present in specified path
        'num_images' : numImages,
        'model_type': 'classification'
    },
    ##onnx
    'add_eltwise' : {  
        'model_path' : os.path.join(models_base_path, 'add_eltwise.onnx'),
        'source' : {'model_url': 'dummy', 'opt': True}, # URL irrelavant if model present in specified path
        'num_images' : numImages,
        'model_type': 'classification'
    },
}

