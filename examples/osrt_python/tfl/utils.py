import os
import platform
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

if platform.machine() == 'aarch64':
    numImages = 100
else : 
    import requests
    numImages = 3

models = {
    'mobilenet_v1_1.0_224.tflite' : 'https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_224/1/default/1?lite-format=tflite',
    'deeplabv3_mnv2_ade20k_float.tflite' : 'https://github.com/mlcommons/mobile_models/blob/main/v0_7/tflite/deeplabv3_mnv2_ade20k_float.tflite?raw=true',
    'ssd_mobilenet_v2_300_float.tflite' : 'https://github.com/mlcommons/mobile_models/blob/main/v0_7/tflite/ssd_mobilenet_v2_300_float.tflite?raw=true'
}

tensor_bits = 8
debug_level = 0
max_num_subgraphs = 16
accuracy_level = 1
calibration_frames = 2
calibration_iterations = 7
output_feature_16bit_names_list = ""#"MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6, MobilenetV1/Logits/AvgPool_1a/AvgPool"
params_16bit_names_list = ""#"MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6"

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

artifacts_folder = '../../../model-artifacts/tfl/'

required_options = {
"tidl_tools_path":tidl_tools_path,
"artifacts_folder":artifacts_folder
}
optional_options = {
"platform":"J7",
"version":"7.2",
"tensor_bits":tensor_bits,
"debug_level":debug_level,
"max_num_subgraphs":max_num_subgraphs,
"deny_list":"",
"accuracy_level":accuracy_level,
"advanced_options:calibration_frames": calibration_frames,
"advanced_options:calibration_iterations": calibration_iterations,
"advanced_options:output_feature_16bit_names_list" : output_feature_16bit_names_list,
"advanced_options:params_16bit_names_list" : params_16bit_names_list,
"advanced_options:quantization_scale_type": quantization_scale_type,
#"object_detection:meta_layers_names_list" : meta_layers_names_list,  -- read from models_configs dictionary below
#"object_detection:meta_arch_type" : meta_arch_type,                  -- read from models_configs dictionary below
#"object_detection:confidence_threshold" : 0.3,
#"object_detection:nms_threshold" : 0.5,
#"object_detection:top_k" : 10,
#"object_detection:keep_top_k" : 10,
"advanced_options:high_resolution_optimization": high_resolution_optimization,
"advanced_options:pre_batchnorm_fold" : pre_batchnorm_fold,
#"ti_internal_nc_flag" : ti_internal_nc_flag,
# below options will be read only if accuracy_level = 9, else will be discarded.... for accuracy_level = 0/1, these are preset internally
"advanced_options:activation_clipping" : activation_clipping,
"advanced_options:weight_clipping" : weight_clipping,
"advanced_options:bias_calibration" : bias_calibration,
"advanced_options:add_data_convert_ops" : 0,
"advanced_options:bias_clipping" : 0,
"advanced_options:channel_wise_quantization" : channel_wise_quantization
}

lables = '../../../test_data/labels.txt'
models_base_path = '../../../models/public/tflite/'

def download_models(mpath = models_base_path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(mpath)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(mpath)

    for model_name in models:
        model_path = mpath + model_name
        if(not os.path.isfile(model_path)):
            print("Downloading  ", model_name)
            url = models[model_name]
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def get_class_labels(output, org_image_rgb):
    output = np.squeeze(np.float32(output)) 
    source_img = org_image_rgb.convert("RGBA")
    draw = ImageDraw.Draw(source_img)

    outputoffset = 0 if(output.shape[0] == 1001) else 1 
    top_k = output.argsort()[-5:][::-1]
    labels = load_labels(lables)
    for j, k in enumerate(top_k):
        curr_class = f'\n  {j}  {output[k]:08.6f}  {labels[k+outputoffset]} \n'
        classes = classes + curr_class if ('classes' in locals()) else curr_class 
    draw.text((0,0), classes, fill='red')
    source_img = source_img.convert("RGB")
    classes = classes.replace("\n", ",")
    return(classes, source_img)

colors_list = [
( 255, 	 0,	  0 ), ( 0	 , 255,    0 ), ( 0	,   0,	 255 ), ( 255, 255,	    0  ), ( 0	 , 255,  255  ), ( 255,   0,	 255  ),
( 255, 	 64,  0 ), ( 64	 , 255,    0 ), ( 64,   0,	 255 ), ( 255, 255,	   64  ), ( 64	 , 255,  255  ), ( 255,   64,	 255  ),
( 196, 	128,  0 ), ( 128 , 196,    0 ), ( 128,  0,	 196 ), ( 196, 196,	  128  ), ( 128	 , 196,  196  ), ( 196,   128,	 196  ),
( 64, 	128,  0 ), ( 128 , 64,     0 ), ( 128,  0,	 64  ), ( 196,   0,    0  ), ( 196	 ,  64,   64  ), ( 64,    196,	  64  ),
( 64,   255, 64 ), ( 64	 , 64,   255 ),( 255, 64,	 64  ), (128,  255,   128  ), ( 128	, 128,    255  ),( 255,   128,	 128  ),
( 196,  64, 196 ), ( 196, 196,    64 ),( 64,  196,	196  ), (196,  255,   196  ), ( 196	, 196,    255  ),( 196,   196,	 128  )]

def mask_transform(inp):
    colors = np.asarray(colors_list)
    inp = np.squeeze(inp)
    colorimg = np.zeros((inp.shape[0], inp.shape[1], 3), dtype=np.float32)
    height, width = inp.shape
    inp = np.rint(inp)
    inp = inp.astype(np.uint8)
    for y in range(height):
        for x in range(width):
            if(inp[y][x] < 22):
                colorimg[y][x] = colors[inp[y][x]]
    inp = colorimg.astype(np.uint8)
    return inp

def RGB2YUV( rgb ):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
    yuv = np.dot(rgb,m)
    yuv[:,:, 1:] += 128.0
    rgb = np.clip(yuv, 0.0, 255.0)
    return yuv

def YUV2RGB( yuv ):
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 2.0320025777816772],
                 [ 1.14019975662231445, -0.5811380310058594 , 0.00001542569043522235] ])
    yuv[:,:, 1:] -= 128.0
    rgb = np.dot(yuv,m)
    rgb = np.clip(rgb, 0.0, 255.0)

    return rgb

def seg_mask_overlay(output_data, org_image_rgb):
  classes = ''
  output_data = np.squeeze(output_data)
  if (output_data.ndim > 2) :
    output_data = output_data.argmax(axis=2)
  output_data = np.squeeze(output_data)
  mask_image_rgb  = mask_transform(output_data) 
  org_image  = RGB2YUV(org_image_rgb)
  mask_image = RGB2YUV(mask_image_rgb)
  
  org_image[:,:, 1] = mask_image[:,:, 1]
  org_image[:,:, 2] = mask_image[:,:, 2]
  blend_image = YUV2RGB(org_image)
  blend_image = blend_image.astype(np.uint8)
  blend_image = Image.fromarray(blend_image).convert('RGB')
  
  return(classes, blend_image)

def det_box_overlay(outputs, org_image_rgb, od_type):
    classes = ''
    source_img = org_image_rgb.convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    if(od_type == "HasDetectionPostProcLayer"):  # model has detection post processing layer
        for i in range(int(outputs[3][0])):
            if(outputs[2][0][i] > 0.1) :
                ymin = outputs[0][0][i][0]
                xmin = outputs[0][0][i][1]
                ymax = outputs[0][0][i][2]
                xmax = outputs[0][0][i][3]
                draw.rectangle(((int(xmin*source_img.width), int(ymin*source_img.height)), (int(xmax*source_img.width), int(ymax*source_img.height))), outline = colors_list[int(outputs[1][0][i])%len(colors_list)], width=2)
    elif(od_type == "EfficientDetLite"): # model does not have detection post processing layer 
        for i in range(int(outputs[0].shape[1])):
            if(outputs[0][0][i][5] > 0.3) :
                ymin = outputs[0][0][i][1]
                xmin = outputs[0][0][i][2]
                ymax = outputs[0][0][i][3]
                xmax = outputs[0][0][i][4]
                print(outputs[0][0][i][6])
                draw.rectangle(((int(xmin), int(ymin)), (int(xmax), int(ymax))), outline = colors_list[int(outputs[0][0][i][6])%len(colors_list)], width=2)
    
    source_img = source_img.convert("RGB")
    return(classes, source_img)


mlperf_models_configs = {
    'mobilenet_v1_1.0_224' : {
        'model_path' : os.path.join(models_base_path, 'mobilenet_v1_1.0_224.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1001,
        'model_type': 'classification'
    },
    'resnet50_v1_5' : {
        'model_path' : os.path.join(models_base_path, 'resnet50_v1_5.tflite'),
        'mean': [123.68, 116.78,  103.94],
        'std' : [1, 1, 1],
        'num_images' : numImages,
        'num_classes': 1000,
        'model_type': 'classification'
    },
    'mobilenet_edgetpu_224_1.0' : {
        'model_path' : os.path.join(models_base_path, 'mobilenet_edgetpu_224_1.0_float.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1001,
        'model_type': 'classification'
   },
    'deeplabv3_mnv2_ade20k_float' : {
        'model_path' : os.path.join(models_base_path,'deeplabv3_mnv2_ade20k_float.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 32,
        'model_type': 'seg'
    },
    'ssd_mobilenet_v1_coco_2018_01_28' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v1_coco_2018_01_28_th_0p3.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v2_coco_2018_03_29' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_coco_2018_03_29.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v2_300_float' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_300_float.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v2_320x320_coco17_tpu-8' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_320x320_coco17_tpu-8.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'efficientdet-ti-lite0_k5s1_k3s2' : {
        'model_path' : os.path.join(models_base_path,'efficientdet-ti-lite0_k5s1_k3s2.tflite'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.01712475, 0.017507, 0.01742919],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'EfficientDetLite',
        'meta_layers_names_list' : '../testvecs/models/public/tflite/efficientdet-ti-lite0.prototxt',
        'meta_arch_type' : 5
    },
}
