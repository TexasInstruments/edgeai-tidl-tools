import os
import platform

models_base_path = '../../../models/public/'
if platform.machine() == 'aarch64':
    numImages = 100
else : 
    import requests
    import onnx
    numImages = 3

models_configs = {
    # ONNX RT OOB Models
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
    'cl-ort-resnet18-v1_4batch' : {
        'model_path' : os.path.join(models_base_path, 'resnet18_opset9_4batch.onnx'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/torchvision/resnet18_opset9_4batch.onnx', 'opt': True,  'infer_shape' : True},
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification'
    },
    'ss-ort-deeplabv3lite_mobilenetv2' : {
        'model_path' : os.path.join(models_base_path, 'deeplabv3lite_mobilenetv2.onnx'),
        'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/segmentation/ade20k32/jai-pytorch/deeplabv3lite_mobilenetv2_512x512_ade20k32_20210308.onnx', 'opt': True,  'infer_shape' : True},
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1001,
        'session_name' : 'tflitert',
        'model_type': 'classification'
    },
    'cl-tfl-mobilenetv2_4batch' : {
        'model_path' : os.path.join(models_base_path, 'mobilenetv2_4batch.tflite'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/tf1-models/mobilenetv2_4batch.tflite', 'opt': True},
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1001,
        'session_name' : 'tflitert',
        'model_type': 'classification'
    },
    'od-tfl-ssd_mobilenet_v2_300_float' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_300_float.tflite'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/mlperf/ssd_mobilenet_v2_300_float.tflite', 'opt': True},
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    # SSD Meta architecture based tflite OD model example
    'od-tfl-ssdlite_mobiledet_dsp_320x320_coco' : {
        'model_path' : os.path.join(models_base_path,'ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite', 'opt': True, \
                    'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/models/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.prototxt'},
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
        'meta_layers_names_list' : os.path.join(models_base_path, 'ssdlite_mobiledet_dsp_320x320_coco_20200519.prototxt'),
        'meta_arch_type' : 1,
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ss-tfl-deeplabv3_mnv2_ade20k_float' : {
        'model_path' : os.path.join(models_base_path,'deeplabv3_mnv2_ade20k_float.tflite'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/segmentation/ade20k32/mlperf/deeplabv3_mnv2_ade20k32_float.tflite', 'opt': True},
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
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
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1001,
        'session_name' : 'tvmdlr',
        'model_type': 'classification'
    },
    'cl-dlr-onnx_mobilenetv2' : {
        'model_path' : os.path.join(models_base_path, 'mobilenetv2-1.0.onnx'),
        'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_opset9.onnx', 'opt': True,  'infer_shape' : False},
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'tvmdlr',
        'model_type': 'classification'
    },
    'cl-dlr-timm_mobilenetv3_large_100' : {
        'model_path' : os.path.join(models_base_path, 'mobilenetv3_large_100.onnx'),
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'tvmdlr',
        'model_type': 'classification'
    },
    # benchmark models - For release testing
    'cl-0000_tflitert_imagenet1k_mlperf_mobilenet_v1_1.0_224_tflite' :{
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite', 'opt': True},
        'model_path' : os.path.join(models_base_path, 'mobilenet_v1_1.0_224.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'tflitert',
        'model_type': 'classification'
    },
    'cl-6360_onnxrt_imagenet1k_fbr-pycls_regnetx-200mf_onnx' :{
        'model_path' : os.path.join(models_base_path, 'regnetx-200mf.onnx'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/classification/imagenet1k/fbr-pycls/regnetx-200mf.onnx', 'opt': True,  'infer_shape' : True},
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification'
    },
    'cl-3090_tvmdlr_imagenet1k_torchvision_mobilenet_v2_tv_onnx' :{
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx', 'opt': True,  'infer_shape' : True},
        'model_path' : os.path.join(models_base_path, 'mobilenet_v2_tv.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'tvmdlr',
        'model_type': 'classification'
    },
    'od-2020_tflitert_coco_tf1-models_ssdlite_mobiledet_dsp_320x320_coco_20200519_tflite' : {
        'model_path' : os.path.join(models_base_path,'ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite', 'opt': True},
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer',
        'object_detection:confidence_threshold': 0.3,
        'object_detection:top_k': 200
    },
    'od-8020_onnxrt_coco_edgeai-mmdet_ssd_mobilenetv2_lite_512x512_20201214_model_onnx' : { 
        'model_path' : os.path.join(models_base_path, 'ssd_mobilenetv2_lite_512x512_20201214_model.onnx'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.onnx', 'opt': True,  'infer_shape' : True, \
                    'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models///vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.prototxt'},
        'mean': [0, 0, 0],
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(models_base_path, 'ssd_mobilenetv2_lite_512x512_20201214_model.prototxt'),
        'session_name' : 'onnxrt' ,
        'meta_arch_type' : 3
    },
    'od-8200_onnxrt_coco_edgeai-mmdet_yolox_nano_lite_416x416_20220214_model_onnx' :{  #wrong infer
        'model_path' : os.path.join(models_base_path, 'yolox_nano_lite_416x416_20220214_model.onnx'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_nano_lite_416x416_20220214_model.onnx', 'opt': True,  'infer_shape' : True, \
                    'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_nano_lite_416x416_20220214_model.prototxt'},
        'mean': [0, 0, 0],
        'scale' : [1, 1, 1],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(models_base_path, 'yolox_nano_lite_416x416_20220214_model.prototxt'),
        'session_name' : 'onnxrt' ,
        'meta_arch_type' : 6
    },
    'od-8220_onnxrt_coco_edgeai-mmdet_yolox_s_lite_640x640_20220221_model_onnx' :{  # infer wrong
        'model_path' : os.path.join(models_base_path, 'yolox_s_lite_640x640_20220221_model.onnx'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_s_lite_640x640_20220221_model.onnx', 'opt': True,  'infer_shape' : True, \
                    'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_s_lite_640x640_20220221_model.prototxt'},
        'mean': [0, 0, 0],
        'scale' : [1, 1, 1],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(models_base_path, 'yolox_s_lite_640x640_20220221_model.prototxt'),
        'session_name' : 'onnxrt' ,
        'meta_arch_type' : 6
    },    
    'od-8420_onnxrt_widerface_edgeai-mmdet_yolox_s_lite_640x640_20220307_model_onnx' :{  
        'model_path' : os.path.join(models_base_path, 'yolox_s_lite_640x640_20220307_model.onnx'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/widerface/edgeai-mmdet/yolox_s_lite_640x640_20220307_model.onnx', 'opt': True,  'infer_shape' : True, \
                    'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/widerface/edgeai-mmdet/yolox_s_lite_640x640_20220307_model.prototxt'},
        'mean': [0, 0, 0],
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'SSD',
        'framework' : 'MMDetection',
        'meta_layers_names_list' : os.path.join(models_base_path, 'yolox_s_lite_640x640_20220307_model.prototxt'),
        'session_name' : 'onnxrt' ,
        'meta_arch_type' : 6
    },
    'ss-2580_tflitert_ade20k32_mlperf_deeplabv3_mnv2_ade20k32_float_tflite' : {
        'model_path' : os.path.join(models_base_path,'deeplabv3_mnv2_ade20k_float.tflite'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/segmentation/ade20k32/mlperf/deeplabv3_mnv2_ade20k32_float.tflite', 'opt': True},
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 32,
        'session_name' : 'tflitert',
        'model_type': 'seg'
    },
    'ss-8610_onnxrt_ade20k32_edgeai-tv_deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4_onnx' : { # need post process changes
        'model_path' : os.path.join(models_base_path, 'deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4.onnx'),
        'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/segmentation/ade20k32/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4.onnx', 'opt': False,  'infer_shape' : True},
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 19,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },

    # Caffe Model - Would be converted ot ONNX
    'cl-ort-caffe_mobilenet_v1' : {
        'model_path' : os.path.join(models_base_path, 'caffe_mobilenet_v1.onnx'),
        'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v1_prototext.link', 'opt': True,  'infer_shape' : False,
                    'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v1_caffemodel.link', 
                    'prototext' :   os.path.join(models_base_path, 'caffe_mobilenet_v1.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_mobilenet_v1.caffemodel') },
        'mean': [103.94, 116.78, 123.68],
        'scale' : [0.017125, 0.017507, 0.017429],
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
        'scale' : [0.017125, 0.017507, 0.017429],
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
        'scale' : [1, 1, 1],
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
        'scale' : [1, 1, 1],
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
        'scale' : [0.017125, 0.017507, 0.017429],
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
        'scale' : [0.017125, 0.017507, 0.017429],
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
        'scale' : [1,1,1],
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
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification'
    },
    'bisenetv2' : {
        'model_path' : os.path.join(models_base_path, 'bisenetv2.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },
    'shufflenet_v2_x1.0_opset9' : {
        'model_path' : os.path.join(modelzoo_path, 'vision/classification/imagenet1k/torchvision/shufflenet_v2_x1.0_opset9.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification'
    },
    'RegNetX-800MF_dds_8gpu_opset9' : {
        'model_path' : os.path.join(modelzoo_path, 'vision/classification/imagenet1k/pycls/RegNetX-800MF_dds_8gpu_opset9.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification'
    },
    'mlperf_ssd_resnet34-ssd1200' : {
        'model_path' : '../../../../../../models/public/onnx/mlperf_resnet34_ssd/ssd_shape.onnx',
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.003921568627,0.003921568627,0.003921568627],
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
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },

    'fpnlite_aspp_mobilenetv2' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_mobilenetv2_768x384_20200120.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },
    'unetlite_aspp_mobilenetv2' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/unetlite_aspp_mobilenetv2_768x384_20200129.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },    
    'fpnlite_aspp_regnetx800mf' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_regnetx800mf_768x384_20200911.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },    
    'fpnlite_aspp_regnetx1.6gf' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_regnetx1.6gf_1024x512_20200914.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },    
    'fpnlite_aspp_regnetx3.2gf' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_regnetx3.2gf_1024x512_20200916.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    }, 
    'deeplabv3_resnet50_1040x520' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/torchvision/deeplabv3_resnet50_1040x520_20200901-213517.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },
    'fcn_resnet50_1040x520' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/torchvision/ffcn_resnet50_1040x520_20200902-153444.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'seg'
    },

    

    'resnet50_v1_5' : {
        'model_path' : os.path.join(models_base_path, 'resnet50_v1_5.tflite'),
        'mean': [123.68, 116.78,  103.94],
        'scale' : [1, 1, 1],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'tflitert',
        'model_type': 'classification'
    },
    'mobilenet_edgetpu_224_1.0' : {
        'model_path' : os.path.join(models_base_path, 'mobilenet_edgetpu_224_1.0_float.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 1001,
        'session_name' : 'tflitert',
        'model_type': 'classification'
   },

    'ssd_mobilenet_v1_coco_2018_01_28' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v1_coco_2018_01_28_th_0p3.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v2_coco_2018_03_29' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_coco_2018_03_29.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },

    'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v2_320x320_coco17_tpu-8' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_320x320_coco17_tpu-8.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'efficientdet-ti-lite0_k5s1_k3s2' : {
        'model_path' : os.path.join(models_base_path,'efficientdet-ti-lite0_k5s1_k3s2.tflite'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.01712475, 0.017507, 0.01742919],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od',
        'od_type' : 'EfficientDetLite',
        'meta_layers_names_list' : '../testvecs/models/public/tflite/efficientdet-ti-lite0.prototxt',
        'session_name' : 'tflitert',
        'meta_arch_type' : 5
    },
"""

