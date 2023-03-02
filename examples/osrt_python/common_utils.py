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
mixed_precision_factor = -1
quantization_scale_type = 0
high_resolution_optimization = 0
pre_batchnorm_fold = 1
ti_internal_nc_flag = 1601

data_convert = 3
SOC = os.environ["SOC"]
if (quantization_scale_type == 3):
    data_convert = 0

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
"deny_list":"", #"MaxPool"
"deny_list:layer_type":"", 
"deny_list:layer_name":"",
"model_type":"",#OD
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
"advanced_options:add_data_convert_ops" : data_convert,
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
                                    'scale':config['scale'],
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
                scale = models_configs[model_name]['scale']

                if model_source['opt'] == True:
                    if filename[-1] == '.onnx':
                        onnxOpt.tidlOnnxModelOptimize(abs_path,abs_path, scale, mean)
                    elif filename[-1] == '.tflite':
                        tflOpt.tidlTfliteModelOptimize(abs_path,abs_path, scale, mean)

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

