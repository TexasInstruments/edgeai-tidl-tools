import os
import platform

models_base_path = '../../../../../models/public/'
if platform.machine() == 'aarch64':
    numImages = 100
else : 
    import requests
    import onnx
    numImages = 3

models_configs = {
    # ONNX RT OOB Models
    'cl-ort-resnet18-v1_yuv' : {
        'model_path' : os.path.join(models_base_path, 'resnet18_opset9_yuv.onnx'),
        'source' : {'model_url': 'dummy', 'opt': True,  'infer_shape' : True},
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification'
    },
    'cl-ort-resnet18-v1' : {
        'model_path' : os.path.join(models_base_path, 'resnet18_opset9.onnx'),
        'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/classification/imagenet1k/torchvision/resnet18_opset9.onnx', 'opt': True,  'infer_shape' : True},
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification'
    },
    # TFLite RT OOB Models
    'cl-tfl-mobilenet_v1_1.0_224_yuv' : {
        'model_path' : os.path.join(models_base_path, 'mobilenet_v1_1.0_224_yuv.tflite'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite', 'opt': True},
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1001,
        'session_name' : 'tflitert',
        'model_type': 'classification'
    },  
    'cl-tfl-mobilenet_v1_1.0_224' : {
        'model_path' : os.path.join(models_base_path, 'mobilenet_v1_1.0_224.tflite'),
        'source' : {'model_url': 'dummy', 'opt': True},        
        'num_images' : numImages,
        'num_classes': 1001,
        'session_name' : 'tflitert',
        'model_type': 'classification'
    },    
}
