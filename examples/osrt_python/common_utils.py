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
    numImages = 3

    # directory reach
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(os.path.dirname(current))
    # setting path
    sys.path.append(parent)
    from scripts.osrt_model_tools.tflite_tools import tflite_model_opt as tflOpt
    from scripts.osrt_model_tools.onnx_tools   import onnx_model_opt as onnxOpt

    from caffe2onnx.src.load_save_model import loadcaffemodel, saveonnxmodel
    from caffe2onnx.src.caffe2onnx import Caffe2Onnx
    from caffe2onnx.src.args_parser import parse_args
    from caffe2onnx.src.utils import freeze

artifacts_folder = '../../../model-artifacts/'
output_images_folder = '../../../output_images/'


tensor_bits = 8
debug_level = 0
max_num_subgraphs = 16
accuracy_level = 1
calibration_frames = 2
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
"advanced_options:add_data_convert_ops" : 3,
"advanced_options:channel_wise_quantization" : channel_wise_quantization
}

modelzoo_path = '../../../../../../jacinto-ai-modelzoo/models'
modelforest_path = '../../../../../../jacinto-ai-modelforest/models'


lables = '../../../test_data/labels.txt'
models_base_path = '../../../models/public/'

def gen_param_yaml(artifacts_folder_path, config, new_height, new_width):
    resize = []
    crop = []
    resize.append(new_width)
    resize.append(new_height)
    crop.append(new_width)
    crop.append(new_height)
    if(config['model_type'] == "classification"):
        model_type = "classification"
    elif(config['model_type'] == "od"):
        model_type = "detection"
    elif(config['model_type'] == "seg"):
        model_type = "segmentation"
    model_file = config['model_path'].split("/")[0]
    dict_file =[]
    layout = 'NCHW'
    if config['session_name'] == 'tflitert':
        layout = 'NHWC'
    
    model_file_name = os.path.basename(config['model_path'])
    
    dict_file.append( {'session' :  {'artifacts_folder': '',
                                     'model_folder': 'model',
                                     'model_path': model_file_name,
                                     'session_name': config['session_name']} ,
                      'task_type' : model_type,
                      'target_device': 'pc',
                      'postprocess':{'data_layout' : layout },
                      'preprocess' :{'data_layout' : layout ,
                                    'mean':config['mean'],
                                    'scale':config['std'],
                                    'resize':resize,
                                    'crop':crop
                                     } })
    
    if(config['model_type'] == "od"):
        if(config['od_type'] == "SSD"):
            dict_file[0]['postprocess']['formatter'] = {'name' : 'DetectionBoxSL2BoxLS', 'src_indices' : [5,4]}
        elif(config['od_type'] == "HasDetectionPostProcLayer"):
            dict_file[0]['postprocess']['formatter'] = {'name' : 'DetectionYXYX2XYXY','src_indices' : [1,0,3,2]}
        
        dict_file[0]['postprocess']['detection_thr'] = 0.3

    with open(os.path.join(artifacts_folder_path, "param.yaml"), 'w') as file:
        documents = yaml.dump(dict_file[0], file)

    if (config['session_name'] == 'tflitert') or (config['session_name'] == 'onnxrt'):
        shutil.copy(config['model_path'], os.path.join(artifacts_folder_path,model_file_name))

headers = {
'User-Agent': 'My User Agent 1.0',
'From': 'aid@ti.com'  # This is another valid field
}  

def get_url_from_link_file(url):
    if url.endswith('.link'):
        r = requests.get(url, allow_redirects=True, headers=headers)
        url = r.content.rstrip()
    return url

def download_model(models_configs, model_name): 

    if(model_name in models_configs.keys()):
        if('source' in models_configs[model_name].keys()):
            model_source = models_configs[model_name]['source']
            model_path = models_configs[model_name]['model_path']
            if(not os.path.isfile(model_path)):
                # Check whether the specified path exists or not
                if not os.path.exists(os.path.dirname(model_path)):
                    # Create a new directory because it does not exist 
                    os.makedirs(os.path.dirname(model_path))

                if('original_model_type' in models_configs[model_name].keys()) and models_configs[model_name]['original_model_type'] == 'caffe':
                    print("Downloading  ", model_source['prototext'])
                    r = requests.get(get_url_from_link_file(model_source['model_url']), allow_redirects=True, headers=headers)
                    open(model_source['prototext'], 'wb').write(r.content)

                    print("Downloading  ", model_source['caffe_model'])
                    r = requests.get(get_url_from_link_file(model_source['caffe_model_url']), allow_redirects=True, headers=headers)
                    open(model_source['caffe_model'], 'wb').write(r.content)

                    graph, params = loadcaffemodel(model_source['prototext'], model_source['caffe_model'])
                    c2o = Caffe2Onnx(graph, params, model_path)
                    onnxmodel = c2o.createOnnxModel()
                    freeze(onnxmodel)
                    saveonnxmodel(onnxmodel, model_path)

                else:
                    print("Downloading  ", model_path)
                    r = requests.get(get_url_from_link_file(model_source['model_url']), allow_redirects=True, headers=headers)
                    open(model_path, 'wb').write(r.content)
            
                filename = os.path.splitext(model_path)
                abs_path = os.path.realpath(model_path)

                mean = models_configs[model_name]['mean']
                std = models_configs[model_name]['std']

                if model_source['opt'] == True:
                    if filename[-1] == '.onnx':
                        onnxOpt.tidlOnnxModelOptimize(abs_path,abs_path, std, mean)
                    elif filename[-1] == '.tflite':
                        tflOpt.tidlTfliteModelOptimize(abs_path,abs_path, std, mean)

                if (filename[-1] == '.onnx') and (model_source['infer_shape'] == True) :
                    onnx.shape_inference.infer_shapes_path(model_path, model_path)
            
            if('meta_layers_names_list' in models_configs[model_name].keys()):
                meta_layers_names_list = models_configs[model_name]['meta_layers_names_list']
                if(not os.path.isfile(meta_layers_names_list)):
                    print("Downloading  ", meta_layers_names_list)
                    r = requests.get(get_url_from_link_file(model_source['meta_arch_url']), allow_redirects=True, headers=headers)
                    open(meta_layers_names_list, 'wb').write(r.content)
    else :
        print(f'{model_name} ot found in availbale list of model configs - {models_configs.keys()}')


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

def det_box_overlay(outputs, org_image_rgb, od_type, framework=None):
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
    
    elif(od_type == "HasDetectionPostProcLayer"):  # model has detection post processing layer
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


models_configs = {
    # ONNX RT OOB Models
    'cl-ort-resnet18-v1' : {
        'model_path' : os.path.join(models_base_path, 'resnet18_opset9.onnx'),
        'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/classification/imagenet1k/torchvision/resnet18_opset9.onnx', 'opt': True,  'infer_shape' : True},
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification'
    },
    'ss-ort-deeplabv3lite_mobilenetv2' : {
        'model_path' : os.path.join(models_base_path, 'deeplabv3lite_mobilenetv2.onnx'),
        'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/segmentation/ade20k32/jai-pytorch/deeplabv3lite_mobilenetv2_512x512_ade20k32_20210308.onnx', 'opt': True,  'infer_shape' : True},
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 19,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },
    'od-ort-ssd-lite_mobilenetv2_fpn' : {
        'model_path' : os.path.join(models_base_path, 'ssd-lite_mobilenetv2_fpn.onnx'),
        'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_fpn_512x512_20201110_model.onnx', 'opt': True,  'infer_shape' : True, \
                    'meta_arch_url' : 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_fpn_512x512_20201110_model.prototxt'},
        'mean': [0, 0, 0],
        'std' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(models_base_path, 'ssd-lite_mobilenetv2_fpn.prototxt'),
        'session_name' : 'onnxrt' ,
        'meta_arch_type' : 3
    },
    # TFLite RT OOB Models
    'cl-tfl-mobilenet_v1_1.0_224' : {
        'model_path' : os.path.join(models_base_path, 'mobilenet_v1_1.0_224.tflite'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite', 'opt': True},
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1001,
        'session_name' : 'tflitert',
        'model_type': 'classification'
    },
    'od-tfl-ssd_mobilenet_v2_300_float' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_300_float.tflite'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/mlperf/ssd_mobilenet_v2_300_float.tflite', 'opt': True},
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ss-tfl-deeplabv3_mnv2_ade20k_float' : {
        'model_path' : os.path.join(models_base_path,'deeplabv3_mnv2_ade20k_float.tflite'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/segmentation/ade20k32/mlperf/deeplabv3_mnv2_ade20k32_float.tflite', 'opt': True},
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 32,
        'session_name' : 'tflitert',
        'model_type': 'seg'
    },
    # TVM DLR OOB Models
    'cl-dlr-tflite_inceptionnetv3' : {
        'model_path' : os.path.join(models_base_path, 'inception_v3.tflite'),
        'source' : {'model_url': 'https://tfhub.dev/tensorflow/lite-model/inception_v3/1/default/1?lite-format=tflite', 'opt': True,  'infer_shape' : False},
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1001,
        'session_name' : 'tvmdlr',
        'model_type': 'classification'
    },
    'cl-dlr-onnx_mobilenetv2' : {
        'model_path' : os.path.join(models_base_path, 'mobilenetv2-1.0.onnx'),
        'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_opset9.onnx', 'opt': True,  'infer_shape' : False},
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'tvmdlr',
        'model_type': 'classification'
    },
    'cl-dlr-mxnet_mobilenetv3_large' : {
        'model_path' : os.path.join(models_base_path, 'mobilenetv3_large'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'tvmdlr',
        'model_type': 'classification'
    },
    # Caffe Model - Would be converted ot ONNX
    'cl-ort-caffe_mobilenet_v1' : {
        'model_path' : os.path.join(models_base_path, 'caffe_mobilenet_v1.onnx'),
        'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v1_prototext.link', 'opt': True,  'infer_shape' : False,
                    'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v1_caffemodel.link', 
                    'prototext' :   os.path.join(models_base_path, 'caffe_mobilenet_v1.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_mobilenet_v1.caffemodel') },
        'mean': [103.94, 116.78, 123.68],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification',
        'original_model_type': 'caffe'
    },
    
    'cl-ort-caffe_mobilenet_v2' : {
        'model_path' : os.path.join(models_base_path, 'caffe_mobilenet_v2.onnx'),
        'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v2_prototext.link', 'opt': True,  'infer_shape' : False,
                    'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v2_caffemodel.link', 
                    'prototext' :   os.path.join(models_base_path, 'caffe_mobilenet_v2.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_mobilenet_v2.caffemodel') },
        'mean': [103.94, 116.78, 123.68],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification',
        'original_model_type': 'caffe'
    },


    'cl-ort-caffe_squeezenet_v1_1' : {
        'model_path' : os.path.join(models_base_path, 'caffe_squeezenet_v1_1.onnx'),
        'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/squeezenet/squeezenet_v1_1.prototext', 'opt': True,  'infer_shape' : False,
                    'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/squeezenet/squeezenet_v1_1_caffemodel.link', 
                    'prototext' :   os.path.join(models_base_path, 'caffe_squeezenet_v1_1.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_squeezenet_v1_1.caffemodel') },
        'mean': [103.94, 116.78, 123.68],
        'std' : [1, 1, 1],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification',
        'original_model_type': 'caffe'
    },

    'cl-ort-caffe_resnet10' : {
        'model_path' : os.path.join(models_base_path, 'caffe_resnet10.onnx'),
        'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/resnet10/deploy.prototxt', 'opt': True,  'infer_shape' : False,
                    'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/resnet10/resnet10_cvgj_iter_320000.caffemodel', 
                    'prototext' :   os.path.join(models_base_path, 'caffe_resnet10.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_resnet10.caffemodel') },
        'mean': [0,0,0],
        'std' : [1, 1, 1],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification',
        'original_model_type': 'caffe'
    },

    'cl-ort-caffe_mobilenetv1_ssd' : {
        'model_path' : os.path.join(models_base_path, 'caffe_mobilenetv1_ssd.onnx'),
        'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/mobilenet_ssd/mobilenet_v1_ssd_prototext.link', 'opt': False,  'infer_shape' : False,
                    'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/mobilenet_ssd/mobilenet_v1_ssd_caffemodel.link', 
                    'meta_arch_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/mobilenet_ssd/mobilenet_v1_ssd_meta.prototxt',
                    'prototext' :   os.path.join(models_base_path, 'caffe_mobilenetv1_ssd.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_mobilenetv1_ssd.caffemodel') },
        'mean': [103.94, 116.78, 123.68],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(models_base_path, 'caffe_mobilenetv1_ssd_meta.prototxt'),
        'session_name' : 'onnxrt' ,
        'meta_arch_type' : 3,
        'original_model_type': 'caffe'
    },

    'cl-ort-caffe_pelee_ssd' : {
        'model_path' : os.path.join(models_base_path, 'caffe_pelee_ssd.onnx'),
        'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/pelee/pelee_ssd.prototxt', 'opt': False,  'infer_shape' : False,
                    'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/pelee/pelee_ssd.caffemodel', 
                    'meta_arch_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/pelee/pelee_ssd_meta.prototxt',
                    'prototext' :   os.path.join(models_base_path, 'caffe_pelee_ssd.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_pelee_ssd.caffemodel') },
        'mean': [103.94, 116.78, 123.68],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(models_base_path, 'caffe_pelee_ssd_meta.prototxt'),
        'session_name' : 'onnxrt' ,
        'meta_arch_type' : 3,
        'original_model_type': 'caffe'
    },

    'cl-ort-caffe_erfnet' : {
        'model_path' : os.path.join(models_base_path, 'caffe_erfnet.onnx'),
        'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/segmentation/cityscapes/caffe/erfnet/erfnet.prototxt', 'opt': False,  'infer_shape' : False,
                    'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/segmentation/cityscapes/caffe/erfnet/erfnet_caffemodel.link', 
                    'prototext' :   os.path.join(models_base_path, 'caffe_erfnet.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_erfnet.caffemodel') },
        'mean': [0,0,0],
        'std' : [1,1,1],
        'num_images' : numImages,
        'num_classes': 19,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg',
        'original_model_type': 'caffe'
    },
}

""" 
    'mobilenetv2-1.0' : {
        'model_path' : os.path.join(models_base_path, 'mobilenetv2-7.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification'
    },
    'bisenetv2' : {
        'model_path' : os.path.join(models_base_path, 'bisenetv2.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },
    'shufflenet_v2_x1.0_opset9' : {
        'model_path' : os.path.join(modelzoo_path, 'vision/classification/imagenet1k/torchvision/shufflenet_v2_x1.0_opset9.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification'
    },
    'RegNetX-800MF_dds_8gpu_opset9' : {
        'model_path' : os.path.join(modelzoo_path, 'vision/classification/imagenet1k/pycls/RegNetX-800MF_dds_8gpu_opset9.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
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
        'session_name' : 'onnxrt' ,
        'meta_arch_type' : 6
    },
    'lraspp_mobilenet_v3_lite_large_512x512_20210527' : {
        'model_path' : '/home/a0230315/workarea/models/public/onnx/lraspp_mobilenet_v3_lite_large_512x512_20210527.onnx',
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },

    'fpnlite_aspp_mobilenetv2' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_mobilenetv2_768x384_20200120.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },
    'unetlite_aspp_mobilenetv2' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/unetlite_aspp_mobilenetv2_768x384_20200129.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },    
    'fpnlite_aspp_regnetx800mf' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_regnetx800mf_768x384_20200911.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },    
    'fpnlite_aspp_regnetx1.6gf' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_regnetx1.6gf_1024x512_20200914.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },    
    'fpnlite_aspp_regnetx3.2gf' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_regnetx3.2gf_1024x512_20200916.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    }, 
    'deeplabv3_resnet50_1040x520' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/torchvision/deeplabv3_resnet50_1040x520_20200901-213517.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },
    'fcn_resnet50_1040x520' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/torchvision/ffcn_resnet50_1040x520_20200902-153444.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },

    

    'resnet50_v1_5' : {
        'model_path' : os.path.join(models_base_path, 'resnet50_v1_5.tflite'),
        'mean': [123.68, 116.78,  103.94],
        'std' : [1, 1, 1],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'tflitert',
        'model_type': 'classification'
    },
    'mobilenet_edgetpu_224_1.0' : {
        'model_path' : os.path.join(models_base_path, 'mobilenet_edgetpu_224_1.0_float.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1001,
        'session_name' : 'tflitert',
        'model_type': 'classification'
   },

    'ssd_mobilenet_v1_coco_2018_01_28' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v1_coco_2018_01_28_th_0p3.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v2_coco_2018_03_29' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_coco_2018_03_29.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },

    'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v2_320x320_coco17_tpu-8' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_320x320_coco17_tpu-8.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
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
        'session_name' : 'tflitert',
        'meta_arch_type' : 5
    },
"""


