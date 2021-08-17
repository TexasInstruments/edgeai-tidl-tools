import os
import platform
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import requests
import onnx

if platform.machine() == 'aarch64':
    numImages = 100
else : 
    numImages = 3

tensor_bits = 8
debug_level = 0
max_num_subgraphs = 16
accuracy_level = 1
calibration_frames = 3
calibration_iterations = 5
output_feature_16bit_names_list = ""#"conv1_2, fire9/concat_1"
params_16bit_names_list = "" #"fire3/squeeze1x1_2"

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

artifacts_folder = './onnxrt-artifacts'

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
"deny_list":"", #"MaxPool",
"accuracy_level":accuracy_level,
"advanced_options:calibration_frames": calibration_frames,
"advanced_options:calibration_iterations": calibration_iterations,
"advanced_options:output_feature_16bit_names_list" : output_feature_16bit_names_list,
"advanced_options:params_16bit_names_list" : params_16bit_names_list,
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

modelzoo_path = '../../../../../../jacinto-ai-modelzoo/models'
modelforest_path = '../../../../../../jacinto-ai-modelforest/models'


lables = '../../../test_data/labels.txt'
models_base_path = '../../../models/public/onnx/'

models = {
    #'mobilenetv2-7.onnx': {'url': 'https://github.com/vinitra-zz/models/raw/7301ce1e16891ed5f75dd15a6a53a643001288f0/vision/classification/mobilenet/model/mobilenetv2-7.onnx', 'dir': '../testvecs/models/public/onnx/'},
    'resnet18_opset9': {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/classification/imagenet1k/torchvision/resnet18_opset9.onnx', 'dir': '../testvecs/models/public/onnx/'},
    'deeplabv3lite_mobilenetv2': {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/segmentation/ade20k32/jai-pytorch/deeplabv3lite_mobilenetv2_512x512_ade20k32_20210308.onnx', 'dir': '../testvecs/models/public/onnx/'},
    'ssd-lite_mobilenetv2_fpn': {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_fpn_512x512_20201110_model.onnx', 'dir': '../testvecs/models/public/onnx/', 
                                      'model_prototxt' : 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_fpn_512x512_20201110_model.prototxt'},
}
def download(mpath, model_name, suffix, type):
    model_file_name = model_name + suffix
    model_path = mpath + model_file_name
    if(not os.path.isfile(model_path)):
        if(type in models[model_name].keys()):
            print("Downloading  ", model_file_name)
            url = models[model_name][type]
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)
            
def download_models(mpath = models_base_path):
     # Check whether the specified path exists or not
    isExist = os.path.exists(mpath)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(mpath)

    for model_name in models:
        download(mpath, model_name, '.onnx', 'model_url')
        download(mpath, model_name, '.prototxt', 'model_prototxt')
    #run shape inference
    for model_name in models:
        model_file_name = model_name + '.onnx'
        model_path = mpath + model_file_name
        print("Running shape inference for ", model_file_name)
        onnx.shape_inference.infer_shapes_path(model_path, model_path)


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

def det_box_overlay(outputs, org_image_rgb, disable_offload, od_type, framework):
    classes = ''
    source_img = org_image_rgb.convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    #mmdet
    if(framework == "MMDetection"):
        outputs = [np.squeeze(output_i) for output_i in outputs]
        num_boxes = int(outputs[0].shape[0])
        for i in range(num_boxes):
            if(outputs[0][i][4] > 0.3) :
                xmin = outputs[0][i][0]
                ymin = outputs[0][i][1]
                xmax = outputs[0][i][2]
                ymax = outputs[0][i][3]
                draw.rectangle(((int(xmin), int(ymin)), (int(xmax), int(ymax))), outline = colors_list[int(outputs[1][i])%len(colors_list)], width=2)
    #SSD
    elif(od_type == 'SSD'):
        outputs = [np.squeeze(output_i) for output_i in outputs]
        num_boxes = int(outputs[0].shape[0])
        for i in range(num_boxes):
            if(outputs[2][i] > 0.3) :
                xmin = outputs[0][i][0]
                ymin = outputs[0][i][1]
                xmax = outputs[0][i][2]
                ymax = outputs[0][i][3]
                draw.rectangle(((int(xmin*source_img.width), int(ymin*source_img.height)), (int(xmax*source_img.width), int(ymax*source_img.height))), outline = colors_list[int(outputs[1][i])%len(colors_list)], width=2)
    #yolov5
    elif(od_type == "YoloV5"):
        outputs = [np.squeeze(output_i) for output_i in outputs]
        num_boxes = int(outputs[0].shape[0])
        for i in range(num_boxes):
            if(outputs[0][i][4] > 0.3) :
                xmin = outputs[0][i][0]
                ymin = outputs[0][i][1]
                xmax = outputs[0][i][2]
                ymax = outputs[0][i][3]
                draw.rectangle(((int(xmin), int(ymin)), (int(xmax), int(ymax))), outline = colors_list[int(outputs[0][i][5])%len(colors_list)], width=2)
    source_img = source_img.convert("RGB")
    return(classes, source_img)


models_configs = {
    'resnet18-v1' : {
        'model_path' : os.path.join(models_base_path, 'resnet18_opset9.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'model_type': 'classification'
    },
    'mobilenetv2-1.0' : {
        'model_path' : os.path.join(models_base_path, 'mobilenetv2-7.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'model_type': 'classification'
    },
    'bisenetv2' : {
        'model_path' : os.path.join(models_base_path, 'bisenetv2.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'model_type': 'seg'
    },
    'shufflenet_v2_x1.0_opset9' : {
        'model_path' : os.path.join(modelzoo_path, 'vision/classification/imagenet1k/torchvision/shufflenet_v2_x1.0_opset9.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'model_type': 'classification'
    },
    'RegNetX-800MF_dds_8gpu_opset9' : {
        'model_path' : os.path.join(modelzoo_path, 'vision/classification/imagenet1k/pycls/RegNetX-800MF_dds_8gpu_opset9.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'model_type': 'classification'
    },
    'mlperf_ssd_resnet34-ssd1200' : {
        'model_path' : '../../../../../../models/public/onnx/mlperf_resnet34_ssd/ssd_shape.onnx',
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : '',
        'meta_layers_names_list' : '../testvecs/models/public/onnx/mlperf_resnet34_ssd/resnet34-ssd1200.prototxt',
        'meta_arch_type' : 3
    },
    'retinanet-lite_regnetx-800mf_fpn_bgr_512x512_20200908_model' : {
        'model_path' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/retinanet-lite_regnetx-800mf_fpn_bgr_512x512_20200908_model.onnx'),
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'RetinaNet',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/retinanet-lite_regnetx-800mf_fpn_bgr_512x512_20200908_model.prototxt'),
        'meta_arch_type' : 5
    },
    'ssd-lite_mobilenetv2_512x512_20201214_220055_model' : {
        'model_path' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_512x512_20201214_220055_model.onnx'),
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_512x512_20201214_220055_model.prototxt'),
        'meta_arch_type' : 3
    },
    'ssd-lite_mobilenetv2_fpn' : {
        'model_path' : os.path.join(models_base_path, 'ssd-lite_mobilenetv2_fpn.onnx'),
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(models_base_path, 'ssd-lite_mobilenetv2_fpn.prototxt'),
        'meta_arch_type' : 3
    },
    'ssd-lite_mobilenetv2_qat-p2_512x512_20201217_model' : {
        'model_path' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_qat-p2_512x512_20201217_model.onnx'),
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_qat-p2_512x512_20201217_model.prototxt'),
        'meta_arch_type' : 3
    },
    'ssd-lite_regnetx-1.6gf_bifpn168x4_bgr_768x768_20201026_model' : {
        'model_path' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/ssd-lite_regnetx-1.6gf_bifpn168x4_bgr_768x768_20201026_model.onnx'),
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/ssd-lite_regnetx-1.6gf_bifpn168x4_bgr_768x768_20201026_model.prototxt'),
        'meta_arch_type' : 3
    },
    'ssd-lite_regnetx-200mf_fpn_bgr_320x320_20201010_model' : {
        'model_path' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/ssd-lite_regnetx-200mf_fpn_bgr_320x320_20201010_model.onnx'),
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/ssd-lite_regnetx-200mf_fpn_bgr_320x320_20201010_model.prototxt'),
        'meta_arch_type' : 3
    },
    'ssd-lite_regnetx-800mf_fpn_bgr_512x512_20200919_model' : {
        'model_path' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/ssd-lite_regnetx-800mf_fpn_bgr_512x512_20200919_model.onnx'),
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/ssd-lite_regnetx-800mf_fpn_bgr_512x512_20200919_model.prototxt'),
        'meta_arch_type' : 3
    },
    'ssd_resnet_fpn_512x512_20200730-225222_model' : {
        'model_path' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/ssd_resnet_fpn_512x512_20200730-225222_model.onnx'),
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/ssd_resnet_fpn_512x512_20200730-225222_model.prototxt'),
        'meta_arch_type' : 3
    },
    'yolov3-lite_regnetx-1.6gf_bgr_512x512_20210202_model' : {
        'model_path' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/yolov3-lite_regnetx-1.6gf_bgr_512x512_20210202_model.onnx'),
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'RetinaNet',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/yolov3-lite_regnetx-1.6gf_bgr_512x512_20210202_model.prototxt'),
        'meta_arch_type' : 4
    },
    'yolov5m6_640_ti_lite_44p1_62p9' : {
        'model_path' : os.path.join(modelforest_path, 'vision/detection/coco/ultralytics-yolov5/yolov5m6_640_ti_lite_44p1_62p9.onnx'),
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'YoloV5',
        'framework' : '',
        'meta_layers_names_list' : os.path.join(modelforest_path, 'vision/detection/coco/ultralytics-yolov5/yolov5m6_640_ti_lite_metaarch.prototxt'),
        'meta_arch_type' : 6
    },
    'yolov5s6_640_ti_lite_37p4_56p0' : {
        'model_path' : os.path.join(modelforest_path, 'vision/detection/coco/ultralytics-yolov5/yolov5s6_640_ti_lite_37p4_56p0.onnx'),
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'YoloV5',
        'framework' : '',
        'meta_layers_names_list' : os.path.join(modelforest_path, 'vision/detection/coco/ultralytics-yolov5/yolov5s6_640_ti_lite_metaarch.prototxt'),
        'meta_arch_type' : 6
    },
    'yolov3_d53_416x416_20210116_005003_model' : {
        'model_path' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/yolov3_d53_416x416_20210116_005003_model.onnx'),
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'YoloV3',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/yolov3_d53_416x416_20210116_005003_model.prototxt'),
        'meta_arch_type' : 4
    },
    'yolov3_d53_relu_416x416_20210117_004118_model' : {
        'model_path' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/yolov3_d53_relu_416x416_20210117_004118_model.onnx'),
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'YoloV3',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(modelforest_path, 'vision/detection/coco/edgeai-mmdet/yolov3_d53_relu_416x416_20210117_004118_model.prototxt'),
        'meta_arch_type' : 4
    },
    'yolov3-10' : {
        'model_path' : '/home/a0230315/Downloads/yolov3-10.onnx',
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'YoloV3',
        'framework' : ''
    },
    'yolov5s_ti_lite_35p0_54p5' : {
        'model_path' : '../../../../../../models/public/onnx/yolov5s_ti_lite_35p0_54p5.onnx',
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'YoloV5',
        'framework' : '',
        'meta_layers_names_list' : '../testvecs/config/import/public/onnx/yolov5s_ti_lite_metaarch.prototxt',
        'meta_arch_type' : 6
    },
    'lraspp_mobilenet_v3_lite_large_512x512_20210527' : {
        'model_path' : '/home/a0230315/workarea/models/public/onnx/lraspp_mobilenet_v3_lite_large_512x512_20210527.onnx',
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'model_type': 'seg'
    },
    'deeplabv3lite_mobilenetv2' : {
        'model_path' : os.path.join(models_base_path, 'deeplabv3lite_mobilenetv2.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 19,
        'model_type': 'seg'
    },
    'fpnlite_aspp_mobilenetv2' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_mobilenetv2_768x384_20200120.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'model_type': 'seg'
    },
    'unetlite_aspp_mobilenetv2' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/unetlite_aspp_mobilenetv2_768x384_20200129.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'model_type': 'seg'
    },    
    'fpnlite_aspp_regnetx800mf' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_regnetx800mf_768x384_20200911.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'model_type': 'seg'
    },    
    'fpnlite_aspp_regnetx1.6gf' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_regnetx1.6gf_1024x512_20200914.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'model_type': 'seg'
    },    
    'fpnlite_aspp_regnetx3.2gf' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_regnetx3.2gf_1024x512_20200916.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'model_type': 'seg'
    }, 
    'deeplabv3_resnet50_1040x520' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/torchvision/deeplabv3_resnet50_1040x520_20200901-213517.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'model_type': 'seg'
    },
    'fcn_resnet50_1040x520' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/torchvision/ffcn_resnet50_1040x520_20200902-153444.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'model_type': 'seg'
    },
}

