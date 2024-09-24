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
import platform
from benchmark_utils.config_utils import AttrDict, create_model_config


models_base_path = '../../../models/public/'
if platform.machine() == 'aarch64':
    numImages = 100
else : 
    import requests
    import onnx
    numImages = 3

models_configs = {
    ############ onnx models ##########
    'cl-ort-resnet18-v1' : create_model_config(
        source=AttrDict(
            model_url='https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/classification/imagenet1k/torchvision/resnet18_opset9.onnx',  
            infer_shape=True,
        ),
        preprocess=AttrDict(
            resize=256,
            crop=224,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name='onnxrt' ,
            model_path=os.path.join(models_base_path, 'resnet18_opset9.onnx'),
            input_mean=[123.675, 116.28, 103.53],
            input_scale=[0.017125, 0.017507, 0.017429],
            input_optimization=True,
        ),
            task_type = 'classification',
        extra_info=AttrDict(
            num_images = numImages ,
            num_classes = 1000
        )
    ),
    'cl-6360_onnxrt_imagenet1k_fbr-pycls_regnetx-200mf_onnx' : create_model_config(
        source=AttrDict(
            model_url='http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/classification/imagenet1k/fbr-pycls/regnetx-200mf.onnx',  
            infer_shape=True,
        ),
        preprocess=AttrDict(
            resize=256,
            crop=224,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=True,
        ),
        session=AttrDict(
            session_name='onnxrt' ,
            model_path=os.path.join(models_base_path, 'regnetx-200mf.onnx'),
            input_mean=[123.675, 116.28, 103.53],
            input_scale=[0.017125, 0.017507, 0.017429],
            input_optimization=True,
        ),
            task_type = 'classification',
        extra_info=AttrDict(
            num_images = numImages ,
            num_classes = 1000
        )
    ),
    'od-ort-ssd-lite_mobilenetv2_fpn' : create_model_config(
            source=AttrDict(
                model_url='https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_fpn_512x512_20201110_model.onnx', 
                meta_arch_url = 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_fpn_512x512_20201110_model.prototxt',
                infer_shape=True),
            preprocess=AttrDict(
                resize=512,
                crop=512,
                data_layout='NCHW',
                pad_color=0,
                resize_with_pad=False,
                reverse_channels=False),
            session=AttrDict(
                session_name='onnxrt',
                model_path=os.path.join(models_base_path, 'ssd-lite_mobilenetv2_fpn.onnx'),
                meta_layers_names_list = os.path.join(models_base_path, 'ssd-lite_mobilenetv2_fpn.prototxt'),
                meta_arch_type=3,
                input_mean=[0, 0, 0],
                input_scale=[0.003921568627,0.003921568627,0.003921568627],
                input_optimization=True),
            postprocess=AttrDict(
                    formatter='DetectionBoxSL2BoxLS',
                    resize_with_pad=False, keypoint=False, object6dpose=False, 
                    normalized_detections=False,
                    shuffle_indices=None, squeeze_axis=None, reshape_list=[(-1,5), (-1,1)], ignore_index=None
                ),
            task_type = 'detection',
            extra_info=AttrDict(
                od_type = 'SSD',
                framework = 'MMDetection',
                num_images = numImages ,
                num_classes = 91,
                label_offset_type = '80to90',
                label_offset = 1
                ),
    ),
    'od-8420_onnxrt_widerface_edgeai-mmdet_yolox_s_lite_640x640_20220307_model_onnx' : create_model_config( #didnt work
            source=AttrDict(
                model_url='http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/widerface/edgeai-mmdet/yolox_s_lite_640x640_20220307_model.onnx',
                meta_arch_url =  'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/widerface/edgeai-mmdet/yolox_s_lite_640x640_20220307_model.prototxt',
                infer_shape=True),
            preprocess=AttrDict(
                resize=640,
                crop=640,
                data_layout='NCHW',
                reverse_channels=True,
                resize_with_pad=[True, "corner"],
                pad_color=[114, 114, 114]
                ),
            session=AttrDict(
                session_name='onnxrt',
                model_path=os.path.join(models_base_path, 'yolox_s_lite_640x640_20220307_model.onnx'),
                meta_layers_names_list = os.path.join(models_base_path,  'yolox_s_lite_640x640_20220307_model.prototxt'),
                meta_arch_type=6,
                input_mean=[0, 0, 0],
                input_scale=[0.003921568627,0.003921568627,0.003921568627],
                input_optimization=True),
            postprocess=AttrDict(
                    formatter='DetectionBoxSL2BoxLS',
                    resize_with_pad=True, keypoint=False, object6dpose=False, 
                    normalized_detections=False,
                    shuffle_indices=None, squeeze_axis=None, reshape_list=[(-1,5), (-1,1)], ignore_index=None
                ),
            task_type = 'detection',
            extra_info=AttrDict(
                od_type = 'SSD',
                framework = 'MMDetection',
                num_images = numImages ,
                num_classes = 91,
                label_offset_type = '80to90',
                label_offset = 1
                ),
    ),
    'od-8020_onnxrt_coco_edgeai-mmdet_ssd_mobilenetv2_lite_512x512_20201214_model_onnx' : create_model_config(
            source=AttrDict(
                model_url='http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.onnx',
                meta_arch_url =  'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models///vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.prototxt',
                infer_shape=True),
            preprocess=AttrDict(
                resize=512,
                crop=512,
                data_layout='NCHW',
                pad_color=0,
                resize_with_pad=False,
                reverse_channels=False),
            session=AttrDict(
                session_name='onnxrt',
                model_path=os.path.join(models_base_path, 'ssd_mobilenetv2_lite_512x512_20201214_model.onnx'),
                meta_layers_names_list = os.path.join(models_base_path,  'ssd_mobilenetv2_lite_512x512_20201214_model.prototxt'),
                meta_arch_type=3,
                input_mean=[0, 0, 0],
                input_scale=[0.003921568627,0.003921568627,0.003921568627],
                input_optimization=True),
            postprocess=AttrDict(
                    formatter='DetectionBoxSL2BoxLS',
                    resize_with_pad=False, keypoint=False, object6dpose=False, 
                    normalized_detections=False,
                    shuffle_indices=None, squeeze_axis=None, reshape_list=[(-1,5), (-1,1)], ignore_index=None
                ),
            task_type = 'detection',
            extra_info=AttrDict(
                od_type = 'SSD',
                framework = 'MMDetection',
                num_images = numImages ,
                num_classes = 91,
                label_offset_type = '80to90',
                label_offset = 1
            ),
    ),
    'ss-ort-deeplabv3lite_mobilenetv2' : create_model_config(
            source=AttrDict(
                model_url='https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/segmentation/ade20k32/jai-pytorch/deeplabv3lite_mobilenetv2_512x512_ade20k32_20210308.onnx', 
                infer_shape=True),
            preprocess=AttrDict(
                resize=512,
                crop=512,
                data_layout='NCHW',
                pad_color=0,
                resize_with_pad=False,
                reverse_channels=False),
            session=AttrDict(
                session_name='onnxrt',
                model_path=os.path.join(models_base_path, 'deeplabv3lite_mobilenetv2.onnx'),
                meta_arch_type=3,
                input_mean= [123.675, 116.28, 103.53],
                input_scale= [0.017125, 0.017507, 0.017429],
                input_optimization=True),
            postprocess=AttrDict(
                with_argmax=True
                ),
            task_type = 'segmentation',
            extra_info=AttrDict(
                num_images = numImages ,
                num_classes = 19
                )
    ),
    ############### tflite models #############
    'cl-tfl-mobilenet_v1_1.0_224': create_model_config(
        source=AttrDict(
            model_url='http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite',  
        ),
        preprocess=AttrDict(
            resize=256,
            crop=224,
            data_layout='NHWC',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name='tflitert' ,
            model_path=os.path.join(models_base_path, 'mobilenet_v1_1.0_224.tflite'),
            input_mean=[127.5, 127.5, 127.5],
            input_scale= [1/127.5, 1/127.5, 1/127.5],
            input_optimization=True,
        ),
            task_type = 'classification',
        extra_info=AttrDict(
            num_images = numImages ,
            num_classes = 1001
        )
    ),
    'od-tfl-ssd_mobilenet_v2_300_float' : create_model_config(
            source=AttrDict(
                model_url= 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/mlperf/ssd_mobilenet_v2_300_float.tflite',
                ),
            preprocess=AttrDict(
                resize=300,
                crop=300,
                data_layout='NCHW',
                pad_color=0,
                resize_with_pad=False,
                reverse_channels=False),
            session=AttrDict(
                session_name='tflitert',
                model_path=os.path.join(models_base_path, 'ssd_mobilenet_v2_300_float.tflite'),
                input_mean=[127.5, 127.5, 127.5],
                input_scale=[1/127.5, 1/127.5, 1/127.5],
                input_optimization=True),
            postprocess=AttrDict(
                    formatter='DetectionYXYX2XYXY',
                    resize_with_pad=False, keypoint=False, object6dpose=False, 
                    normalized_detections=True,
                    shuffle_indices=None, squeeze_axis=0, reshape_list=None, ignore_index=None
                ),
            task_type = 'detection',
            extra_info=AttrDict(
                od_type = 'HasDetectionPostProcLayer',
                num_images = numImages ,
                num_classes = 91,
                label_offset_type = '90to90',
                label_offset = 1
            ),
    ),
    'od-tfl-ssdlite_mobiledet_dsp_320x320_coco' : create_model_config(
            source=AttrDict(
                model_url = 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite',
                meta_arch_url = 'http://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/models/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.prototxt'
                ),
            preprocess=AttrDict(
                resize=320,
                crop=320,
                data_layout='NCHW',
                pad_color=0,
                resize_with_pad=False,
                reverse_channels=False
                ),
            session=AttrDict(
                session_name='tflitert',
                model_path=os.path.join(models_base_path,'ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite'),
                meta_layers_names_list = os.path.join(models_base_path, 'ssdlite_mobiledet_dsp_320x320_coco_20200519.prototxt'),
                meta_arch_type=1,
                input_mean=[127.5, 127.5, 127.5],
                input_scale=[1/127.5, 1/127.5, 1/127.5],
                input_optimization=True),
            postprocess=AttrDict(
                    formatter='DetectionYXYX2XYXY',
                    resize_with_pad=False, keypoint=False, object6dpose=False, 
                    normalized_detections=True,
                    shuffle_indices=None, squeeze_axis=0, reshape_list=None, ignore_index=None
                ),
            task_type = 'detection',
            extra_info=AttrDict(
                od_type = 'HasDetectionPostProcLayer',
                num_images = numImages ,
                num_classes = 91,
                label_offset_type = '90to90',
                label_offset = 1
            ),
    ),
    'od-2020_tflitert_coco_tf1-models_ssdlite_mobiledet_dsp_320x320_coco_20200519_tflite' : create_model_config (
            source=AttrDict(
                model_url= 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite',
                meta_arch_url = 'http://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/models/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.prototxt'
                ),
            preprocess=AttrDict(
                resize=320,
                crop=320,
                data_layout='NCHW',
                pad_color=0,
                resize_with_pad=False,
                reverse_channels=False
                ),
            session=AttrDict(
                session_name='tflitert',
                model_path=os.path.join(models_base_path, 'ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite'),
                meta_layers_names_list = os.path.join(models_base_path, 'ssdlite_mobiledet_dsp_320x320_coco_20200519.prototxt'),
                meta_arch_type=1,
                input_mean=[127.5, 127.5, 127.5],
                input_scale=[1/127.5, 1/127.5, 1/127.5],
                input_optimization=True),
            postprocess=AttrDict(
                    formatter='DetectionYXYX2XYXY',
                    resize_with_pad=False, keypoint=False, object6dpose=False, 
                    normalized_detections=True,
                    shuffle_indices=None, squeeze_axis=0, reshape_list=None, ignore_index=None
                ),
            task_type = 'detection',
            extra_info=AttrDict(
                od_type = 'HasDetectionPostProcLayer',
                num_images = numImages ,
                num_classes = 91,
                label_offset_type = '90to90',
                label_offset = 1
            ),
    ),
    # benchmark models - For release testing
    'cl-0000_tflitert_imagenet1k_mlperf_mobilenet_v1_1.0_224_tflite': create_model_config(
        source=AttrDict(
            model_url =  'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/mlperf/mobilenet_v1_1.0_224.tflite',  
        ),
        preprocess=AttrDict(
            resize=224,
            crop=224,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name='tflitert' ,
            model_path=os.path.join(models_base_path, 'mobilenet_v1_1.0_224.tflite'),
            input_mean=[127.5, 127.5, 127.5],
            input_scale= [1/127.5, 1/127.5, 1/127.5],
            input_optimization=True,
        ),
            task_type = 'classification',
        extra_info=AttrDict(
            num_images = numImages ,
            num_classes = 1001
        )
    ),
    'ss-8610_onnxrt_ade20k32_edgeai-tv_deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4_onnx' : create_model_config(
            source=AttrDict(
                model_url='http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/segmentation/ade20k32/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4.onnx', 
                infer_shape=True),
            preprocess=AttrDict(
                resize=512,
                crop=512,
                data_layout='NCHW',
                pad_color=0,
                resize_with_pad=False,
                reverse_channels=False),
            session=AttrDict(
                session_name='onnxrt',
                model_path=os.path.join(models_base_path, 'deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4.onnx'),
                meta_arch_type=3,
                input_mean= [123.675, 116.28, 103.53],
                input_scale= [0.017125, 0.017507, 0.017429],
                input_optimization=False),
            postprocess=AttrDict(
                with_argmax=True
                ),
            task_type = 'segmentation',
            extra_info=AttrDict(
                num_images = numImages ,
                num_classes = 19
            )
    ),
    'ss-2580_tflitert_ade20k32_mlperf_deeplabv3_mnv2_ade20k32_float_tflite' : create_model_config(
            source=AttrDict(
                model_url='http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/segmentation/ade20k32/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4.onnx', 
                ),
            preprocess=AttrDict(
                resize=512,
                crop=512,
                data_layout='NCHW',
                pad_color=0,
                resize_with_pad=False,
                reverse_channels=False),
            session=AttrDict(
                session_name='tflitert',
                model_path=os.path.join(models_base_path, 'deeplabv3_mnv2_ade20k_float.tflite'),
                input_mean= [127.5, 127.5, 127.5],
                input_scale=  [1/127.5, 1/127.5, 1/127.5],
                input_optimization=True),
            postprocess=AttrDict(
                with_argmax=False
                ),
            task_type = 'segmentation',
            extra_info=AttrDict(
                num_images = numImages ,
                num_classes = 32
            )
    ),
    'ss-tfl-deeplabv3_mnv2_ade20k_float' : create_model_config(
            source=AttrDict(
                model_url='http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/segmentation/ade20k32/mlperf/deeplabv3_mnv2_ade20k32_float.tflite', 
                ),
            preprocess=AttrDict(
                resize=512,
                crop=512,
                data_layout='NHWC',
                pad_color=0,
                resize_with_pad=False,
                reverse_channels=False),
            session=AttrDict(
                session_name='tflitert',
                model_path=os.path.join(models_base_path, 'deeplabv3_mnv2_ade20k_float.tflite'),
                input_mean= [127.5, 127.5, 127.5],
                input_scale=  [1/127.5, 1/127.5, 1/127.5],
                input_optimization=True),
            postprocess=AttrDict(
                with_argmax=False
                ),
            task_type = 'segmentation',
            extra_info=AttrDict(
                num_images = numImages ,
                num_classes = 32
            )
    ),
    #Caffe Model - Would be converted ot ONNX
    'cl-ort-caffe_mobilenet_v1' : create_model_config(
        source=AttrDict(
            model_url='https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v1_prototext.link',
            caffe_model_url = 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v1_caffemodel.link', 
            prototext =  os.path.join(models_base_path, 'caffe_mobilenet_v1.prototxt'), 
            caffe_model = os.path.join(models_base_path,'caffe_mobilenet_v1.caffemodel'),
            infer_shape=False,
        ),
        preprocess=AttrDict(
            resize=256,
            crop=224,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name='onnxrt' ,
            model_path=os.path.join(models_base_path, 'caffe_mobilenet_v1.onnx'),
            input_mean= [103.94, 116.78, 123.68],
            input_scale=[0.017125, 0.017507, 0.017429],
            input_optimization=True,
        ),
            task_type = 'classification',
        extra_info=AttrDict(
            original_model_type = 'caffe',
            num_images = numImages ,
            num_classes = 1000
        )
    ),
    'cl-ort-caffe_mobilenet_v2' : create_model_config(
        source=AttrDict(
            model_url='https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v2_prototext.link',
            caffe_model_url = 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v2_caffemodel.link', 
            prototext =  os.path.join(models_base_path, 'caffe_mobilenet_v2.prototxt'), 
            caffe_model = os.path.join(models_base_path,'caffe_mobilenet_v2.caffemodel'),
            infer_shape=False,
        ),
        preprocess=AttrDict(
            resize=256,
            crop=224,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name='onnxrt' ,
            model_path=os.path.join(models_base_path, 'caffe_mobilenet_v2.onnx'),
            input_mean= [103.94, 116.78, 123.68],
            input_scale=[0.017125, 0.017507, 0.017429],
            input_optimization=True,
        ),
            task_type = 'classification',
        extra_info=AttrDict(
            original_model_type = 'caffe',
            num_images = numImages ,
            num_classes = 1000
        )
    ),
    'cl-ort-caffe_squeezenet_v1_1' : create_model_config(
        source=AttrDict(
            model_url='https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/squeezenet/squeezenet_v1_1.prototext',
            caffe_model_url = 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/squeezenet/squeezenet_v1_1_caffemodel.link', 
            prototext =  os.path.join(models_base_path, 'caffe_squeezenet_v1_1.prototxt'), 
            caffe_model = os.path.join(models_base_path,'caffe_squeezenet_v1_1.caffemodel'),
            infer_shape=False,
        ),
        preprocess=AttrDict(
            resize=256,
            crop=224,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name='onnxrt' ,
            model_path=os.path.join(models_base_path, 'caffe_squeezenet_v1_1.onnx'),
            input_mean= [103.94, 116.78, 123.68],
            input_scale= [1, 1, 1],
            input_optimization=True,
        ),
            task_type = 'classification',
        extra_info=AttrDict(
            original_model_type = 'caffe',
            num_images = numImages ,
            num_classes = 1000
        )
    ),
    'cl-ort-caffe_resnet10' : create_model_config(
        source=AttrDict(
            model_url='https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/resnet10/deploy.prototxt',
            caffe_model_url = 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/resnet10/resnet10_cvgj_iter_320000.caffemodel', 
            prototext =  os.path.join(models_base_path,  'caffe_resnet10.prototxt'), 
            caffe_model = os.path.join(models_base_path,'caffe_resnet10.caffemodel'),
            infer_shape=False,
        ),
        preprocess=AttrDict(
            resize=256,
            crop=224,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name='onnxrt' ,
            model_path=os.path.join(models_base_path, 'caffe_resnet10.onnx'),
            input_mean= [0,0,0],
            input_scale= [1, 1, 1],
            input_optimization=True,
        ),
            task_type = 'classification',
        extra_info=AttrDict(
            original_model_type = 'caffe',
            num_images = numImages ,
            num_classes = 1000
        )
    ),
    'cl-ort-caffe_mobilenetv1_ssd' : create_model_config(
        source=AttrDict(
            model_url= 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/mobilenet_ssd/mobilenet_v1_ssd_prototext.link',
            caffe_model_url = 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/mobilenet_ssd/mobilenet_v1_ssd_caffemodel.link',
            meta_arch_url = 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/mobilenet_ssd/mobilenet_v1_ssd_meta.prototxt',
            prototext =  os.path.join(models_base_path,  'caffe_mobilenetv1_ssd.prototxt'), 
            caffe_model = os.path.join(models_base_path, 'caffe_mobilenetv1_ssd.caffemodel'),
            infer_shape=False,
        ),
        preprocess=AttrDict(
            resize=300,
            crop=300,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name='onnxrt' ,
            model_path=os.path.join(models_base_path, 'caffe_mobilenetv1_ssd.onnx'),
            meta_layers_names_list = os.path.join(models_base_path, 'caffe_mobilenetv1_ssd_meta.prototxt'),
            meta_arch_type=3,
            input_mean= [103.94, 116.78, 123.68],
            input_scale= [0.017125, 0.017507, 0.017429],
            input_optimization=False,
        ),
        postprocess=AttrDict(
                    formatter='DetectionBoxSL2BoxLS',
                    resize_with_pad=False, keypoint=False, object6dpose=False, 
                    normalized_detections=False,
                    shuffle_indices=None, squeeze_axis=None, reshape_list=None, ignore_index=None
                ),
        task_type = 'detection',
        extra_info=AttrDict(
            original_model_type = 'caffe',
            framework = 'MMDetection',
            od_type = 'SSD',
            num_images = numImages ,
            num_classes = 91,
            label_offset_type = '80to90',
            label_offset = 1
        )  
    ),
    'cl-ort-caffe_pelee_ssd' : create_model_config(
        source=AttrDict(
            model_url= 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/pelee/pelee_ssd.prototxt',
            caffe_model_url =  'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/pelee/pelee_ssd.caffemodel',
            meta_arch_url = 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/pelee/pelee_ssd_meta.prototxt',
            prototext =  os.path.join(models_base_path,  'caffe_pelee_ssd.prototxt'), 
            caffe_model = os.path.join(models_base_path, 'caffe_pelee_ssd.caffemodel'),
            infer_shape=False,
        ),
        preprocess=AttrDict(
            resize=304,
            crop=304,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name='onnxrt' ,
            model_path=os.path.join(models_base_path, 'caffe_pelee_ssd.onnx'),
            meta_layers_names_list = os.path.join(models_base_path, 'caffe_pelee_ssd_meta.prototxt'),
            meta_arch_type=3,
            input_mean= [103.94, 116.78, 123.68],
            input_scale= [0.017125, 0.017507, 0.017429],
            input_optimization=False,
        ),
        postprocess=AttrDict(
                    formatter='DetectionBoxSL2BoxLS',
                    resize_with_pad=False, keypoint=False, object6dpose=False, 
                    normalized_detections=False,
                    shuffle_indices=None, squeeze_axis=0, reshape_list=None, ignore_index=None
                ),
        task_type = 'detection',
        extra_info=AttrDict(
            original_model_type = 'caffe',
            framework = 'MMDetection',
            od_type = 'SSD',
            num_images = numImages ,
            num_classes = 91,
            label_offset_type = '80to90',
            label_offset = 1
            )
    ),
    'cl-ort-caffe_erfnet': create_model_config(
            source=AttrDict(
                model_url= 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/segmentation/cityscapes/caffe/erfnet/erfnet.prototxt',
                caffe_model_url = 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/segmentation/cityscapes/caffe/erfnet/erfnet_caffemodel.link',
                meta_arch_url = 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/pelee/pelee_ssd_meta.prototxt',
                prototext =  os.path.join(models_base_path,  'caffe_erfnet.prototxt'), 
                caffe_model = os.path.join(models_base_path, 'caffe_erfnet.caffemodel'),
                infer_shape=False),
            preprocess=AttrDict(
                resize=(256,512),
                crop=(256,512),
                data_layout='NCHW',
                pad_color=0,
                resize_with_pad=False,
                reverse_channels=False),
            session=AttrDict(
                session_name='onnxrt',
                model_path=os.path.join(models_base_path, 'caffe_erfnet.onnx'),
                meta_arch_type=3,
                input_mean= [0,0,0],
                input_scale= [1,1,1],
                input_optimization=True),
            postprocess=AttrDict(
                with_argmax=True
            ),
            task_type = 'segmentation',
            extra_info=AttrDict(
                num_images = numImages ,
                num_classes = 19,
                original_model_type = 'caffe'
            )
    ),
    'od-8200_onnxrt_coco_edgeai-mmdet_yolox_nano_lite_416x416_20220214_model_onnx' : create_model_config(
            source=AttrDict(
                model_url='http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/edgeai-mmdet/yolox_nano_lite_416x416_20220214_model.onnx',
                meta_arch_url = 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/edgeai-mmdet/yolox_nano_lite_416x416_20220214_model.prototxt',
                infer_shape=True
                ),
            preprocess=AttrDict(
                resize=416,
                crop=416,
                data_layout='NCHW',
                pad_color=[114, 114, 114],
                resize_with_pad=[True, "corner"],
                reverse_channels=True
                ),
            session=AttrDict(
                session_name='onnxrt',
                model_path=os.path.join(models_base_path, 'yolox_nano_lite_416x416_20220214_model.onnx'),
                meta_layers_names_list = os.path.join(models_base_path, 'yolox_nano_lite_416x416_20220214_model.prototxt'),
                meta_arch_type=6,
                input_mean=[0, 0, 0],
                input_scale=[1, 1, 1],
                input_optimization=True
                ),
            postprocess=AttrDict(
                    formatter='DetectionBoxSL2BoxLS',
                    resize_with_pad=True, keypoint=False, object6dpose=False, 
                    normalized_detections=False,
                    shuffle_indices=None, squeeze_axis=None, reshape_list=[(-1,5), (-1,1)], ignore_index=None
                ),
            task_type = 'detection',
            extra_info=AttrDict(
                od_type = 'SSD',
                framework = 'MMDetection',
                num_images = numImages ,
                num_classes = 91,
                label_offset_type = '80to90',
                label_offset = 1
            )
    ),
    'od-8220_onnxrt_coco_edgeai-mmdet_yolox_s_lite_640x640_20220221_model_onnx' : create_model_config(
            source=AttrDict(
                model_url='http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_s_lite_640x640_20220221_model.onnx',
                meta_arch_url =  'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_s_lite_640x640_20220221_model.prototxt',
                infer_shape=True),
            preprocess=AttrDict(
                resize=640,
                crop=640,
                data_layout='NCHW',
                pad_color=[114, 114, 114],
                resize_with_pad=[True, "corner"],
                reverse_channels=True),
            session=AttrDict(
                session_name='onnxrt',
                model_path=os.path.join(models_base_path, 'yolox_s_lite_640x640_20220221_model.onnx'),
                meta_layers_names_list = os.path.join(models_base_path, 'yolox_s_lite_640x640_20220221_model.prototxt'),
                meta_arch_type=6,
                input_mean=[0, 0, 0],
                input_scale=[1, 1, 1],
                input_optimization=True),
            postprocess=AttrDict(
                    formatter='DetectionBoxSL2BoxLS',
                    resize_with_pad=True, keypoint=False, object6dpose=False, 
                    normalized_detections=False,
                    shuffle_indices=None, squeeze_axis=None, reshape_list=[(-1,5), (-1,1)], ignore_index=None
                ),
            task_type = 'detection',
            extra_info=AttrDict(
                od_type = 'SSD',
                framework = 'MMDetection',
                num_images = numImages ,
                num_classes = 91,
                label_offset_type = '80to90',
                label_offset = 1
                )
    ),
    'cl-ort-resnet18-v1_4batch' : create_model_config(
        source=AttrDict(
            model_url='http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/torchvision/resnet18_opset9_4batch.onnx',  
            infer_shape=True,
        ),
        preprocess=AttrDict(
            resize=256,
            crop=224,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name='onnxrt' ,
            model_path=os.path.join(models_base_path, 'resnet18_opset9_4batch.onnx'),
            input_mean=[123.675, 116.28, 103.53],
            input_scale=[0.017125, 0.017507, 0.017429],
            input_optimization=True,
        ),
            task_type = 'classification',
        extra_info=AttrDict(
            num_images = numImages ,
            num_classes = 1000
        )
    ),
    'cl-tfl-mobilenetv2_4batch': create_model_config(
        source=AttrDict(
            model_url='http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/tf1-models/mobilenetv2_4batch.tflite',  
        ),
        preprocess=AttrDict(
            resize=256,
            crop=224,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name='tflitert' ,
            model_path=os.path.join(models_base_path, 'mobilenetv2_4batch.tflite'),
            input_mean=[127.5, 127.5, 127.5],
            input_scale= [1/127.5, 1/127.5, 1/127.5],
            input_optimization=True,
        ),
            task_type = 'classification',
        extra_info=AttrDict(
            num_images = numImages ,
            num_classes = 1001
        )
    ),
    'cl-dlr-tflite_inceptionnetv3' : create_model_config(
        source=AttrDict(
            model_url= 'https://tfhub.dev/tensorflow/lite-model/inception_v3/1/default/1?lite-format=tflite',  
            infer_shape=False,
        ),
        preprocess=AttrDict(
            resize=299,
            crop=299,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name= 'tvmdlr',
            model_path=os.path.join(models_base_path, 'inception_v3.tflite'),
            input_mean= [0,0,0],
            input_scale= [1,1,1],
            input_optimization=True,
        ),
            task_type = 'classification',
        extra_info=AttrDict(
            num_images = numImages ,
            num_classes = 1001
        )
    ),
    'cl-dlr-onnx_mobilenetv2' : create_model_config(
        source=AttrDict(
            model_url= 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_opset9.onnx',  
            infer_shape=False,
        ),
        preprocess=AttrDict(
            resize=224,
            crop=224,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name= 'tvmdlr',
            model_path=os.path.join(models_base_path, 'mobilenetv2-1.0.onnx'),
            input_mean= [0,0,0],
            input_scale= [1,1,1],
            input_optimization=True,
        ),
            task_type = 'classification',
        extra_info=AttrDict(
            num_images = numImages ,
            num_classes = 1000
        )
    ),
    'cl-dlr-timm_mobilenetv3_large_100' : create_model_config(
        source=AttrDict(
        ),
        preprocess=AttrDict(
            resize=256,
            crop=224,
            data_layout='NCHW',
            resize_with_pad=False,
            reverse_channels=False,
        ),
        session=AttrDict(
            session_name= 'tvmdlr',
            model_path=os.path.join(models_base_path, 'mobilenetv3_large_100.onnx'),
            input_mean= [127.5, 127.5, 127.5],
            input_scale= [1/127.5, 1/127.5, 1/127.5],
            input_optimization=True,
        ),
            task_type = 'classification',
        extra_info=AttrDict(
            num_images = numImages ,
            num_classes = 1000
        )
    ),
}

######### old configs ###########

    # ONNX RT OOB Models
    # 'cl-ort-resnet18-v1' : {
    #     'model_path' : os.path.join(models_base_path, 'resnet18_opset9.onnx'),
    #     'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/classification/imagenet1k/torchvision/resnet18_opset9.onnx', 'opt': True,  'infer_shape' : True},
    #     'mean': [123.675, 116.28, 103.53],
    #     'scale' : [0.017125, 0.017507, 0.017429],
    #     'num_images' : numImages,
    #     'num_classes': 1000,
    #     'session_name' : 'onnxrt' ,
    #     'model_type': 'classification'
    # },
    # 'cl-ort-resnet18_1MP_low_latency' : {
    #     'model_path' : os.path.join(models_base_path, 'resnet18_1024x1024.onnx'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/high_resolution/imagenet1k/torchvision/resnet18_1024x1024.onnx', 'opt': True,  'infer_shape' : True},
    #     'mean': [123.675, 116.28, 103.53],
    #     'scale' : [0.017125, 0.017507, 0.017429],
    #     'num_images' : numImages,
    #     'num_classes': 1000,
    #     'session_name' : 'onnxrt' ,
    #     'model_type': 'classification',
    #     'optional_options' : 
    #     {
    #         'advanced_options:inference_mode' : 2,  # inference mode to run low latency inference
    #         'advanced_options:num_cores' : 4,
    #         'advanced_options:calibration_frames' : 1, 
    #         'advanced_options:calibration_iterations' : 1
    #     }
    # },
    # 'cl-ort-resnet18-v1_4batch' : {
    #     'model_path' : os.path.join(models_base_path, 'resnet18_opset9_4batch.onnx'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/torchvision/resnet18_opset9_4batch.onnx', 'opt': True,  'infer_shape' : True},
    #     'mean': [123.675, 116.28, 103.53],
    #     'scale' : [0.017125, 0.017507, 0.017429],
    #     'num_images' : numImages,
    #     'num_classes': 1000,
    #     'session_name' : 'onnxrt' ,
    #     'model_type': 'classification',
    #     'optional_options' : 
    #     {
    #         'advanced_options:inference_mode' : 1,  # inference mode to run high throughput (parallel batch processing) inference
    #         'advanced_options:num_cores' : 4, 
    #     }
    # },
    # 'ss-ort-deeplabv3lite_mobilenetv2' : {
    #     'model_path' : os.path.join(models_base_path, 'deeplabv3lite_mobilenetv2.onnx'),
    #     'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/segmentation/ade20k32/jai-pytorch/deeplabv3lite_mobilenetv2_512x512_ade20k32_20210308.onnx', 'opt': True,  'infer_shape' : True},
    #     'mean': [123.675, 116.28, 103.53],
    #     'scale' : [0.017125, 0.017507, 0.017429],
    #     'num_images' : numImages,
    #     'num_classes': 19,
    #     'session_name' : 'onnxrt' ,
    #     'model_type': 'seg'
    # },
    # 'od-ort-ssd-lite_mobilenetv2_fpn' : {
    #     'model_path' : os.path.join(models_base_path, 'ssd-lite_mobilenetv2_fpn.onnx'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_fpn_lite_512x512_20201110_model.onnx', 'opt': True,  'infer_shape' : True, \
    #                 'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_fpn_lite_512x512_20201110_model.prototxt'},
    #     'mean': [0, 0, 0],
    #     'scale' : [0.003921568627,0.003921568627,0.003921568627],
    #     'num_images' : numImages,
    #     'num_classes': 91,
    #     'model_type': 'od',
    #     'od_type' : 'SSD',
    #     'framework' : 'MMDetection',
    #     'meta_layers_names_list' : os.path.join(models_base_path, 'ssd-lite_mobilenetv2_fpn.prototxt'),
    #     'session_name' : 'onnxrt' ,
    #     'meta_arch_type' : 3
    # },
    # 'cl-ort-deit-tiny' : {
    #     'model_path' : os.path.join(models_base_path, 'deit_tiny_1.onnx'),
    #     'source' : {'model_url': 'dummy', 'opt': True,  'infer_shape' : True},
    #     'mean': [123.675, 116.28, 103.53],
    #     'scale' : [0.017125, 0.017507, 0.017429],
    #     'num_images' : numImages,
    #     'num_classes': 1000,
    #     'session_name' : 'onnxrt' ,
    #     'model_type': 'classification',
    #     'optional_options': {
    #         'advanced_options:quantization_scale_type':4
    #     }
    # },
    # # TFLite RT OOB Models
    # 'cl-tfl-mobilenet_v1_1.0_224' : {
    #     'model_path' : os.path.join(models_base_path, 'mobilenet_v1_1.0_224.tflite'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite', 'opt': True},
    #     'mean': [127.5, 127.5, 127.5],
    #     'scale' : [1/127.5, 1/127.5, 1/127.5],
    #     'num_images' : numImages,
    #     'num_classes': 1001,
    #     'session_name' : 'tflitert',
    #     'model_type': 'classification'
    # },
    # 'cl-tfl-mobilenetv2_4batch' : {
    #     'model_path' : os.path.join(models_base_path, 'mobilenetv2_4batch.tflite'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/tf1-models/mobilenetv2_4batch.tflite', 'opt': True},
    #     'mean': [127.5, 127.5, 127.5],
    #     'scale' : [1/127.5, 1/127.5, 1/127.5],
    #     'num_images' : numImages,
    #     'num_classes': 1001,
    #     'session_name' : 'tflitert',
    #     'model_type': 'classification',
    #     'optional_options' : 
    #     {
    #         'advanced_options:inference_mode' : 1,  # inference mode to run high throughput (parallel batch processing) inference
    #         'advanced_options:num_cores' : 4, 
    #     }
    # },
    # 'od-tfl-ssd_mobilenet_v2_300_float' : {
    #     'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_300_float.tflite'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/mlperf/ssd_mobilenet_v2_300_float.tflite', 'opt': True},
    #     'mean': [127.5, 127.5, 127.5],
    #     'scale' : [1/127.5, 1/127.5, 1/127.5],
    #     'num_images' : numImages,
    #     'num_classes': 91,
    #     'model_type': 'od',
    #     'session_name' : 'tflitert',
    #     'od_type' : 'HasDetectionPostProcLayer'
    # },
    # # SSD Meta architecture based tflite OD model example
    # 'od-tfl-ssdlite_mobiledet_dsp_320x320_coco' : {
    #     'model_path' : os.path.join(models_base_path,'ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite', 'opt': True, \
    #                 'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/models/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.prototxt'},
    #     'mean': [127.5, 127.5, 127.5],
    #     'scale' : [1/127.5, 1/127.5, 1/127.5],
    #     'num_images' : numImages,
    #     'num_classes': 91,
    #     'model_type': 'od',
    #     'session_name' : 'tflitert',
    #     'meta_layers_names_list' : os.path.join(models_base_path, 'ssdlite_mobiledet_dsp_320x320_coco_20200519.prototxt'),
    #     'meta_arch_type' : 1,
    #     'od_type' : 'HasDetectionPostProcLayer'
    # },
    # 'ss-tfl-deeplabv3_mnv2_ade20k_float' : {
    #     'model_path' : os.path.join(models_base_path,'deeplabv3_mnv2_ade20k_float.tflite'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/segmentation/ade20k32/mlperf/deeplabv3_mnv2_ade20k32_float.tflite', 'opt': True},
    #     'mean': [127.5, 127.5, 127.5],
    #     'scale' : [1/127.5, 1/127.5, 1/127.5],
    #     'num_images' : numImages,
    #     'num_classes': 32,
    #     'session_name' : 'tflitert',
    #     'model_type': 'seg'
    # },
    # 'ss-tfl-deeplabv3_mnv2_ade20k_float_low_latency' : {
    #     'model_path' : os.path.join(models_base_path,'deeplabv3_mnv2_ade20k_float.tflite'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/segmentation/ade20k32/mlperf/deeplabv3_mnv2_ade20k32_float.tflite', 'opt': True},
    #     'mean': [127.5, 127.5, 127.5],
    #     'scale' : [1/127.5, 1/127.5, 1/127.5],
    #     'num_images' : numImages,
    #     'num_classes': 32,
    #     'session_name' : 'tflitert',
    #     'model_type': 'seg',
    #     'optional_options' : 
    #     {
    #         'advanced_options:inference_mode' : 2,  # inference mode to run low latency inference
    #         'advanced_options:num_cores' : 4, 
    #     }
    # },
    # # TVM DLR OOB Models
    # 'cl-dlr-tflite_inceptionnetv3' : {
    #     'model_path' : os.path.join(models_base_path, 'inception_v3.tflite'),
    #     'source' : {'model_url': 'https://tfhub.dev/tensorflow/lite-model/inception_v3/1/default/1?lite-format=tflite', 'opt': True,  'infer_shape' : False},
    #     'mean': [127.5, 127.5, 127.5],
    #     'scale' : [1/127.5, 1/127.5, 1/127.5],
    #     'num_images' : numImages,
    #     'num_classes': 1001,
    #     'session_name' : 'tvmdlr',
    #     'model_type': 'classification'
    # },
    # 'cl-dlr-onnx_mobilenetv2' : {
    #     'model_path' : os.path.join(models_base_path, 'mobilenetv2-1.0.onnx'),
    #     'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_opset9.onnx', 'opt': True,  'infer_shape' : False},
    #     'mean': [127.5, 127.5, 127.5],
    #     'scale' : [1/127.5, 1/127.5, 1/127.5],
    #     'num_images' : numImages,
    #     'num_classes': 1000,
    #     'session_name' : 'tvmdlr',
    #     'model_type': 'classification'
    # },
    # 'cl-dlr-timm_mobilenetv3_large_100' : {
    #     'model_path' : os.path.join(models_base_path, 'mobilenetv3_large_100.onnx'),
    #     'mean': [127.5, 127.5, 127.5],
    #     'scale' : [1/127.5, 1/127.5, 1/127.5],
    #     'num_images' : numImages,
    #     'num_classes': 1000,
    #     'session_name' : 'tvmdlr',
    #     'model_type': 'classification'
    # },
    # # benchmark models - For release testing
    # 'cl-0000_tflitert_imagenet1k_mlperf_mobilenet_v1_1.0_224_tflite' :{
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite', 'opt': True},
    #     'model_path' : os.path.join(models_base_path, 'mobilenet_v1_1.0_224.tflite'),
    #     'mean': [127.5, 127.5, 127.5],
    #     'scale' : [1/127.5, 1/127.5, 1/127.5],
    #     'num_images' : numImages,
    #     'num_classes': 1000,
    #     'session_name' : 'tflitert',
    #     'model_type': 'classification'
    # },
    # 'cl-6360_onnxrt_imagenet1k_fbr-pycls_regnetx-200mf_onnx' :{
    #     'model_path' : os.path.join(models_base_path, 'regnetx-200mf.onnx'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/classification/imagenet1k/fbr-pycls/regnetx-200mf.onnx', 'opt': True,  'infer_shape' : True},
    #     'mean': [123.675, 116.28, 103.53],
    #     'scale' : [0.017125, 0.017507, 0.017429],
    #     'num_images' : numImages,
    #     'num_classes': 1000,
    #     'session_name' : 'onnxrt' ,
    #     'model_type': 'classification'
    # },
    # 'cl-3090_tvmdlr_imagenet1k_torchvision_mobilenet_v2_tv_onnx' :{
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx', 'opt': True,  'infer_shape' : True},
    #     'model_path' : os.path.join(models_base_path, 'mobilenet_v2_tv.onnx'),
    #     'mean': [123.675, 116.28, 103.53],
    #     'scale' : [0.017125, 0.017507, 0.017429],
    #     'num_images' : numImages,
    #     'num_classes': 1000,
    #     'session_name' : 'tvmdlr',
    #     'model_type': 'classification'
    # },
    # 'od-2020_tflitert_coco_tf1-models_ssdlite_mobiledet_dsp_320x320_coco_20200519_tflite' : {
    #     'model_path' : os.path.join(models_base_path,'ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite', 'opt': True},
    #     'mean': [127.5, 127.5, 127.5],
    #     'scale' : [1/127.5, 1/127.5, 1/127.5],
    #     'num_images' : numImages,
    #     'num_classes': 91,
    #     'model_type': 'od',
    #     'session_name' : 'tflitert',
    #     'od_type' : 'HasDetectionPostProcLayer',
    #     'object_detection:confidence_threshold' : 0.3,
    #     'object_detection:top_k' : 200
    # },
    # 'od-8020_onnxrt_coco_edgeai-mmdet_ssd_mobilenetv2_lite_512x512_20201214_model_onnx' : { 
    #     'model_path' : os.path.join(models_base_path, 'ssd_mobilenetv2_lite_512x512_20201214_model.onnx'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.onnx', 'opt': True,  'infer_shape' : True, \
    #                 'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models///vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.prototxt'},
    #     'mean': [0, 0, 0],
    #     'scale' : [0.003921568627,0.003921568627,0.003921568627],
    #     'num_images' : numImages,
    #     'num_classes': 91,
    #     'model_type': 'od',
    #     'od_type' : 'SSD',
    #     'framework' : 'MMDetection',
    #     'meta_layers_names_list' : os.path.join(models_base_path, 'ssd_mobilenetv2_lite_512x512_20201214_model.prototxt'),
    #     'session_name' : 'onnxrt' ,
    #     'meta_arch_type' : 3
    # },
    # 'od-8200_onnxrt_coco_edgeai-mmdet_yolox_nano_lite_416x416_20220214_model_onnx' :{  #wrong infer
    #     'model_path' : os.path.join(models_base_path, 'yolox_nano_lite_416x416_20220214_model.onnx'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_nano_lite_416x416_20220214_model.onnx', 'opt': True,  'infer_shape' : True, \
    #                 'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_nano_lite_416x416_20220214_model.prototxt'},
    #     'mean': [0, 0, 0],
    #     'scale' : [1, 1, 1],
    #     'num_images' : numImages,
    #     'num_classes': 91,
    #     'model_type': 'od',
    #     'od_type' : 'SSD',
    #     'framework' : 'MMDetection',
    #     'session_name' : 'onnxrt' ,
    #     'meta_layers_names_list' : os.path.join(models_base_path, 'yolox_nano_lite_416x416_20220214_model.prototxt'),
    #     'meta_arch_type' : 6
    # },
    # 'od-8220_onnxrt_coco_edgeai-mmdet_yolox_s_lite_640x640_20220221_model_onnx' :{  # infer wrong
    #     'model_path' : os.path.join(models_base_path, 'yolox_s_lite_640x640_20220221_model.onnx'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_s_lite_640x640_20220221_model.onnx', 'opt': True,  'infer_shape' : True, \
    #                 'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_s_lite_640x640_20220221_model.prototxt'},
    #     'mean': [0, 0, 0],
    #     'scale' : [1, 1, 1],
    #     'num_images' : numImages,
    #     'num_classes': 91,
    #     'model_type': 'od',
    #     'od_type' : 'SSD',
    #     'framework' : 'MMDetection',
    #     'session_name' : 'onnxrt' ,
    #     'meta_layers_names_list' : os.path.join(models_base_path, 'yolox_s_lite_640x640_20220221_model.prototxt'),
    #     'meta_arch_type' : 6
    # },    
    # 'od-8420_onnxrt_widerface_edgeai-mmdet_yolox_s_lite_640x640_20220307_model_onnx' :{  
    #     'model_path' : os.path.join(models_base_path, 'yolox_s_lite_640x640_20220307_model.onnx'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/widerface/edgeai-mmdet/yolox_s_lite_640x640_20220307_model.onnx', 'opt': True,  'infer_shape' : True, \
    #                 'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/widerface/edgeai-mmdet/yolox_s_lite_640x640_20220307_model.prototxt'},
    #     'mean': [0, 0, 0],
    #     'scale' : [0.003921568627,0.003921568627,0.003921568627],
    #     'num_images' : numImages,
    #     'num_classes': 91,
    #     'model_type': 'od',
    #     'od_type' : 'SSD',
    #     'framework' : 'MMDetection',
    #     'session_name' : 'onnxrt' ,
    #     'meta_layers_names_list' : os.path.join(models_base_path, 'yolox_s_lite_640x640_20220307_model.prototxt'),
    #     'meta_arch_type' : 6
    # },
    # 'ss-2580_tflitert_ade20k32_mlperf_deeplabv3_mnv2_ade20k32_float_tflite' : {
    #     'model_path' : os.path.join(models_base_path,'deeplabv3_mnv2_ade20k_float.tflite'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/segmentation/ade20k32/mlperf/deeplabv3_mnv2_ade20k32_float.tflite', 'opt': True},
    #     'mean': [127.5, 127.5, 127.5],
    #     'scale' : [1/127.5, 1/127.5, 1/127.5],
    #     'num_images' : numImages,
    #     'num_classes': 32,
    #     'session_name' : 'tflitert',
    #     'model_type': 'seg'
    # },
    # 'ss-8610_onnxrt_ade20k32_edgeai-tv_deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4_onnx' : { # need post process changes
    #     'model_path' : os.path.join(models_base_path, 'deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4.onnx'),
    #     'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/segmentation/ade20k32/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4.onnx', 'opt': False,  'infer_shape' : True},
    #     'mean': [123.675, 116.28, 103.53],
    #     'scale' : [0.017125, 0.017507, 0.017429],
    #     'num_images' : numImages,
    #     'num_classes': 19,
    #     'session_name' : 'onnxrt' ,
    #     'model_type': 'seg'
    # },

    # # Caffe Model - Would be converted ot ONNX
    # 'cl-ort-caffe_mobilenet_v1' : {
    #     'model_path' : os.path.join(models_base_path, 'caffe_mobilenet_v1.onnx'),
    #     'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v1_prototext.link', 'opt': True,  'infer_shape' : False,
    #                 'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v1_caffemodel.link', 
    #                 'prototext' :   os.path.join(models_base_path, 'caffe_mobilenet_v1.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_mobilenet_v1.caffemodel') },
    #     'mean': [103.94, 116.78, 123.68],
    #     'scale' : [0.017125, 0.017507, 0.017429],
    #     'num_images' : numImages,
    #     'num_classes': 1000,
    #     'session_name' : 'onnxrt' ,
    #     'model_type': 'classification',
    #     'original_model_type': 'caffe'
    # },
    
    # 'cl-ort-caffe_mobilenet_v2' : {
    #     'model_path' : os.path.join(models_base_path, 'caffe_mobilenet_v2.onnx'),
    #     'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v2_prototext.link', 'opt': True,  'infer_shape' : False,
    #                 'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v2_caffemodel.link', 
    #                 'prototext' :   os.path.join(models_base_path, 'caffe_mobilenet_v2.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_mobilenet_v2.caffemodel') },
    #     'mean': [103.94, 116.78, 123.68],
    #     'scale' : [0.017125, 0.017507, 0.017429],
    #     'num_images' : numImages,
    #     'num_classes': 1000,
    #     'session_name' : 'onnxrt' ,
    #     'model_type': 'classification',
    #     'original_model_type': 'caffe'
    # },


    # 'cl-ort-caffe_squeezenet_v1_1' : {
    #     'model_path' : os.path.join(models_base_path, 'caffe_squeezenet_v1_1.onnx'),
    #     'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/squeezenet/squeezenet_v1_1.prototext', 'opt': True,  'infer_shape' : False,
    #                 'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/squeezenet/squeezenet_v1_1_caffemodel.link', 
    #                 'prototext' :   os.path.join(models_base_path, 'caffe_squeezenet_v1_1.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_squeezenet_v1_1.caffemodel') },
    #     'mean': [103.94, 116.78, 123.68],
    #     'scale' : [1, 1, 1],
    #     'num_images' : numImages,
    #     'num_classes': 1000,
    #     'session_name' : 'onnxrt' ,
    #     'model_type': 'classification',
    #     'original_model_type': 'caffe'
    # },

    # 'cl-ort-caffe_resnet10' : {
    #     'model_path' : os.path.join(models_base_path, 'caffe_resnet10.onnx'),
    #     'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/resnet10/deploy.prototxt', 'opt': True,  'infer_shape' : False,
    #                 'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/resnet10/resnet10_cvgj_iter_320000.caffemodel', 
    #                 'prototext' :   os.path.join(models_base_path, 'caffe_resnet10.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_resnet10.caffemodel') },
    #     'mean': [0,0,0],
    #     'scale' : [1, 1, 1],
    #     'num_images' : numImages,
    #     'num_classes': 1000,
    #     'session_name' : 'onnxrt' ,
    #     'model_type': 'classification',
    #     'original_model_type': 'caffe'
    # },

    # 'cl-ort-caffe_mobilenetv1_ssd' : {
    #     'model_path' : os.path.join(models_base_path, 'caffe_mobilenetv1_ssd.onnx'),
    #     'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/mobilenet_ssd/mobilenet_v1_ssd_prototext.link', 'opt': False,  'infer_shape' : False,
    #                 'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/mobilenet_ssd/mobilenet_v1_ssd_caffemodel.link', 
    #                 'meta_arch_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/mobilenet_ssd/mobilenet_v1_ssd_meta.prototxt',
    #                 'prototext' :   os.path.join(models_base_path, 'caffe_mobilenetv1_ssd.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_mobilenetv1_ssd.caffemodel') },
    #     'mean': [103.94, 116.78, 123.68],
    #     'scale' : [0.017125, 0.017507, 0.017429],
    #     'num_images' : numImages,
    #     'num_classes': 91,
    #     'model_type': 'od',
    #     'od_type' : 'SSD',
    #     'framework' : 'MMDetection',
    #     'meta_layers_names_list' : os.path.join(models_base_path, 'caffe_mobilenetv1_ssd_meta.prototxt'),
    #     'session_name' : 'onnxrt' ,
    #     'meta_arch_type' : 3,
    #     'original_model_type': 'caffe'
    # },

    # 'cl-ort-caffe_pelee_ssd' : {
    #     'model_path' : os.path.join(models_base_path, 'caffe_pelee_ssd.onnx'),
    #     'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/pelee/pelee_ssd.prototxt', 'opt': False,  'infer_shape' : False,
    #                 'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/pelee/pelee_ssd.caffemodel', 
    #                 'meta_arch_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/pelee/pelee_ssd_meta.prototxt',
    #                 'prototext' :   os.path.join(models_base_path, 'caffe_pelee_ssd.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_pelee_ssd.caffemodel') },
    #     'mean': [103.94, 116.78, 123.68],
    #     'scale' : [0.017125, 0.017507, 0.017429],
    #     'num_images' : numImages,
    #     'num_classes': 91,
    #     'model_type': 'od',
    #     'od_type' : 'SSD',
    #     'framework' : 'MMDetection',
    #     'meta_layers_names_list' : os.path.join(models_base_path, 'caffe_pelee_ssd_meta.prototxt'),
    #     'session_name' : 'onnxrt' ,
    #     'meta_arch_type' : 3,
    #     'original_model_type': 'caffe'
    # },

    # 'cl-ort-caffe_erfnet' : {
    #     'model_path' : os.path.join(models_base_path, 'caffe_erfnet.onnx'),
    #     'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/segmentation/cityscapes/caffe/erfnet/erfnet.prototxt', 'opt': False,  'infer_shape' : False,
    #                 'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/segmentation/cityscapes/caffe/erfnet/erfnet_caffemodel.link', 
    #                 'prototext' :   os.path.join(models_base_path, 'caffe_erfnet.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_erfnet.caffemodel') },
    #     'mean': [0,0,0],
    #     'scale' : [1,1,1],
    #     'num_images' : numImages,
    #     'num_classes': 19,
    #     'session_name' : 'onnxrt' ,
    #     'model_type': 'seg',
    #     'original_model_type': 'caffe'
    # },

#################################



# models_configs = {
#     # ONNX RT OOB Models
#     'cl-ort-resnet18-v1' : {
#         'model_path' : os.path.join(models_base_path, 'resnet18_opset9.onnx'),
#         'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/classification/imagenet1k/torchvision/resnet18_opset9.onnx', 'opt': True,  'infer_shape' : True},
#         'mean': [123.675, 116.28, 103.53],
#         'scale' : [0.017125, 0.017507, 0.017429],
#         'num_images' : numImages,
#         'num_classes': 1000,
#         'session_name' : 'onnxrt' ,
#         'model_type': 'classification'
#     },
#     'cl-ort-resnet18-v1_4batch' : {
#         'model_path' : os.path.join(models_base_path, 'resnet18_opset9_4batch.onnx'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/torchvision/resnet18_opset9_4batch.onnx', 'opt': True,  'infer_shape' : True},
#         'mean': [123.675, 116.28, 103.53],
#         'scale' : [0.017125, 0.017507, 0.017429],
#         'num_images' : numImages,
#         'num_classes': 1000,
#         'session_name' : 'onnxrt' ,
#         'model_type': 'classification'
#     },
#     'ss-ort-deeplabv3lite_mobilenetv2' : {
#         'model_path' : os.path.join(models_base_path, 'deeplabv3lite_mobilenetv2.onnx'),
#         'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/segmentation/ade20k32/jai-pytorch/deeplabv3lite_mobilenetv2_512x512_ade20k32_20210308.onnx', 'opt': True,  'infer_shape' : True},
#         'mean': [123.675, 116.28, 103.53],
#         'scale' : [0.017125, 0.017507, 0.017429],
#         'num_images' : numImages,
#         'num_classes': 19,
#         'session_name' : 'onnxrt' ,
#         'model_type': 'segmentation'
#     },
#     'od-ort-ssd-lite_mobilenetv2_fpn' : {
#         'model_path' : os.path.join(models_base_path, 'ssd-lite_mobilenetv2_fpn.onnx'),
#         'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_fpn_512x512_20201110_model.onnx', 'opt': True,  'infer_shape' : True, \
#                     'meta_arch_url' : 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_fpn_512x512_20201110_model.prototxt'},
#         'mean': [0, 0, 0],
#         'scale' : [0.003921568627,0.003921568627,0.003921568627],
#         'num_images' : numImages,
#         'num_classes': 91,
#         'model_type': 'detection',
#         'od_type' : 'SSD',
#         'framework' : 'MMDetection',
#         'meta_layers_names_list' : os.path.join(models_base_path, 'ssd-lite_mobilenetv2_fpn.prototxt'),
#         'session_name' : 'onnxrt' ,
#         'meta_arch_type' : 3
#     },
#     # TFLite RT OOB Models
#     'cl-tfl-mobilenet_v1_1.0_224' : {
#         'model_path' : os.path.join(models_base_path, 'mobilenet_v1_1.0_224.tflite'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite', 'opt': True},
#         'mean': [127.5, 127.5, 127.5],
#         'scale' : [1/127.5, 1/127.5, 1/127.5],
#         'num_images' : numImages,
#         'num_classes': 1001,
#         'session_name' : 'tflitert',
#         'model_type': 'classification'
#     },
#     'cl-tfl-mobilenetv2_4batch' : {
#         'model_path' : os.path.join(models_base_path, 'mobilenetv2_4batch.tflite'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/tf1-models/mobilenetv2_4batch.tflite', 'opt': True},
#         'mean': [127.5, 127.5, 127.5],
#         'scale' : [1/127.5, 1/127.5, 1/127.5],
#         'num_images' : numImages,
#         'num_classes': 1001,
#         'session_name' : 'tflitert',
#         'model_type': 'classification'
#     },
#     'od-tfl-ssd_mobilenet_v2_300_float' : {
#         'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_300_float.tflite'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/mlperf/ssd_mobilenet_v2_300_float.tflite', 'opt': True},
#         'mean': [127.5, 127.5, 127.5],
#         'scale' : [1/127.5, 1/127.5, 1/127.5],
#         'num_images' : numImages,
#         'num_classes': 91,
#         'model_type': 'detection',
#         'session_name' : 'tflitert',
#         'od_type' : 'HasDetectionPostProcLayer'
#     },
#     # SSD Meta architecture based tflite OD model example
#     'od-tfl-ssdlite_mobiledet_dsp_320x320_coco' : {
#         'model_path' : os.path.join(models_base_path,'ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite', 'opt': True, \
#                     'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/models/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.prototxt'},
#         'mean': [127.5, 127.5, 127.5],
#         'scale' : [1/127.5, 1/127.5, 1/127.5],
#         'num_images' : numImages,
#         'num_classes': 91,
#         'model_type': 'detection',
#         'session_name' : 'tflitert',
#         'meta_layers_names_list' : os.path.join(models_base_path, 'ssdlite_mobiledet_dsp_320x320_coco_20200519.prototxt'),
#         'meta_arch_type' : 1,
#         'od_type' : 'HasDetectionPostProcLayer'
#     },
#     'ss-tfl-deeplabv3_mnv2_ade20k_float' : {
#         'model_path' : os.path.join(models_base_path,'deeplabv3_mnv2_ade20k_float.tflite'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/segmentation/ade20k32/mlperf/deeplabv3_mnv2_ade20k32_float.tflite', 'opt': True},
#         'mean': [127.5, 127.5, 127.5],
#         'scale' : [1/127.5, 1/127.5, 1/127.5],
#         'num_images' : numImages,
#         'num_classes': 32,
#         'session_name' : 'tflitert',
#         'model_type': 'segmentation'
#     },
#     # TVM DLR OOB Models
#     'cl-dlr-tflite_inceptionnetv3' : {
#         'model_path' : os.path.join(models_base_path, 'inception_v3.tflite'),
#         'source' : {'model_url': 'https://tfhub.dev/tensorflow/lite-model/inception_v3/1/default/1?lite-format=tflite', 'opt': True,  'infer_shape' : False},
#         'mean': [127.5, 127.5, 127.5],
#         'scale' : [1/127.5, 1/127.5, 1/127.5],
#         'num_images' : numImages,
#         'num_classes': 1001,
#         'session_name' : 'tvmdlr',
#         'model_type': 'classification'
#     },
#     'cl-dlr-onnx_mobilenetv2' : {
#         'model_path' : os.path.join(models_base_path, 'mobilenetv2-1.0.onnx'),
#         'source' : {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_opset9.onnx', 'opt': True,  'infer_shape' : False},
#         'mean': [127.5, 127.5, 127.5],
#         'scale' : [1/127.5, 1/127.5, 1/127.5],
#         'num_images' : numImages,
#         'num_classes': 1000,
#         'session_name' : 'tvmdlr',
#         'model_type': 'classification'
#     },
#     'cl-dlr-timm_mobilenetv3_large_100' : {
#         'model_path' : os.path.join(models_base_path, 'mobilenetv3_large_100.onnx'),
#         'mean': [127.5, 127.5, 127.5],
#         'scale' : [1/127.5, 1/127.5, 1/127.5],
#         'num_images' : numImages,
#         'num_classes': 1000,
#         'session_name' : 'tvmdlr',
#         'model_type': 'classification'
#     },
#     # benchmark models - For release testing
#     'cl-0000_tflitert_imagenet1k_mlperf_mobilenet_v1_1.0_224_tflite' :{
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite', 'opt': True},
#         'model_path' : os.path.join(models_base_path, 'mobilenet_v1_1.0_224.tflite'),
#         'mean': [127.5, 127.5, 127.5],
#         'scale' : [1/127.5, 1/127.5, 1/127.5],
#         'num_images' : numImages,
#         'num_classes': 1000,
#         'session_name' : 'tflitert',
#         'model_type': 'classification'
#     },
#     'cl-6360_onnxrt_imagenet1k_fbr-pycls_regnetx-200mf_onnx' :{
#         'model_path' : os.path.join(models_base_path, 'regnetx-200mf.onnx'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/classification/imagenet1k/fbr-pycls/regnetx-200mf.onnx', 'opt': True,  'infer_shape' : True},
#         'mean': [123.675, 116.28, 103.53],
#         'scale' : [0.017125, 0.017507, 0.017429],
#         'num_images' : numImages,
#         'num_classes': 1000,
#         'session_name' : 'onnxrt' ,
#         'model_type': 'classification'
#     },
#     'cl-3090_tvmdlr_imagenet1k_torchvision_mobilenet_v2_tv_onnx' :{
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx', 'opt': True,  'infer_shape' : True},
#         'model_path' : os.path.join(models_base_path, 'mobilenet_v2_tv.onnx'),
#         'mean': [123.675, 116.28, 103.53],
#         'scale' : [0.017125, 0.017507, 0.017429],
#         'num_images' : numImages,
#         'num_classes': 1000,
#         'session_name' : 'tvmdlr',
#         'model_type': 'classification'
#     },
#     'od-2020_tflitert_coco_tf1-models_ssdlite_mobiledet_dsp_320x320_coco_20200519_tflite' : {
#         'model_path' : os.path.join(models_base_path,'ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite', 'opt': True},
#         'mean': [127.5, 127.5, 127.5],
#         'scale' : [1/127.5, 1/127.5, 1/127.5],
#         'num_images' : numImages,
#         'num_classes': 91,
#         'model_type': 'detection',
#         'session_name' : 'tflitert',
#         'od_type' : 'HasDetectionPostProcLayer',
#         'object_detection:confidence_threshold': 0.3,
#         'object_detection:top_k': 200
#     },
#     'od-8020_onnxrt_coco_edgeai-mmdet_ssd_mobilenetv2_lite_512x512_20201214_model_onnx' : { 
#         'model_path' : os.path.join(models_base_path, 'ssd_mobilenetv2_lite_512x512_20201214_model.onnx'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.onnx', 'opt': True,  'infer_shape' : True, \
#                     'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models///vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.prototxt'},
#         'mean': [0, 0, 0],
#         'scale' : [0.003921568627,0.003921568627,0.003921568627],
#         'num_images' : numImages,
#         'num_classes': 91,
#         'model_type': 'detection',
#         'od_type' : 'SSD',
#         'framework' : 'MMDetection',
#         'meta_layers_names_list' : os.path.join(models_base_path, 'ssd_mobilenetv2_lite_512x512_20201214_model.prototxt'),
#         'session_name' : 'onnxrt' ,
#         'meta_arch_type' : 3
#     },
#     'od-8200_onnxrt_coco_edgeai-mmdet_yolox_nano_lite_416x416_20220214_model_onnx' :{  #wrong infer
#         'model_path' : os.path.join(models_base_path, 'yolox_nano_lite_416x416_20220214_model.onnx'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_nano_lite_416x416_20220214_model.onnx', 'opt': True,  'infer_shape' : True, \
#                     'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_nano_lite_416x416_20220214_model.prototxt'},
#         'mean': [0, 0, 0],
#         'scale' : [1, 1, 1],
#         'num_images' : numImages,
#         'num_classes': 91,
#         'model_type': 'detection',
#         'od_type' : 'SSD',
#         'framework' : 'MMDetection',
#         'meta_layers_names_list' : os.path.join(models_base_path, 'yolox_nano_lite_416x416_20220214_model.prototxt'),
#         'session_name' : 'onnxrt' ,
#         'meta_arch_type' : 6
#     },
#     'od-8220_onnxrt_coco_edgeai-mmdet_yolox_s_lite_640x640_20220221_model_onnx' :{  # infer wrong
#         'model_path' : os.path.join(models_base_path, 'yolox_s_lite_640x640_20220221_model.onnx'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_s_lite_640x640_20220221_model.onnx', 'opt': True,  'infer_shape' : True, \
#                     'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/coco/edgeai-mmdet/yolox_s_lite_640x640_20220221_model.prototxt'},
#         'mean': [0, 0, 0],
#         'scale' : [1, 1, 1],
#         'num_images' : numImages,
#         'num_classes': 91,
#         'model_type': 'detection',
#         'od_type' : 'SSD',
#         'framework' : 'MMDetection',
#         'meta_layers_names_list' : os.path.join(models_base_path, 'yolox_s_lite_640x640_20220221_model.prototxt'),
#         'session_name' : 'onnxrt' ,
#         'meta_arch_type' : 6
#     },    
#     'od-8420_onnxrt_widerface_edgeai-mmdet_yolox_s_lite_640x640_20220307_model_onnx' :{  
#         'model_path' : os.path.join(models_base_path, 'yolox_s_lite_640x640_20220307_model.onnx'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/widerface/edgeai-mmdet/yolox_s_lite_640x640_20220307_model.onnx', 'opt': True,  'infer_shape' : True, \
#                     'meta_arch_url' : 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/detection/widerface/edgeai-mmdet/yolox_s_lite_640x640_20220307_model.prototxt'},
#         'mean': [0, 0, 0],
#         'scale' : [0.003921568627,0.003921568627,0.003921568627],
#         'num_images' : numImages,
#         'num_classes': 91,
#         'model_type': 'detection',
#         'od_type' : 'SSD',
#         'framework' : 'MMDetection',
#         'meta_layers_names_list' : os.path.join(models_base_path, 'yolox_s_lite_640x640_20220307_model.prototxt'),
#         'session_name' : 'onnxrt' ,
#         'meta_arch_type' : 6
#     },
#     'ss-2580_tflitert_ade20k32_mlperf_deeplabv3_mnv2_ade20k32_float_tflite' : {
#         'model_path' : os.path.join(models_base_path,'deeplabv3_mnv2_ade20k_float.tflite'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/segmentation/ade20k32/mlperf/deeplabv3_mnv2_ade20k32_float.tflite', 'opt': True},
#         'mean': [127.5, 127.5, 127.5],
#         'scale' : [1/127.5, 1/127.5, 1/127.5],
#         'num_images' : numImages,
#         'num_classes': 32,
#         'session_name' : 'tflitert',
#         'model_type': 'segmentation'
#     },
#     'ss-8610_onnxrt_ade20k32_edgeai-tv_deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4_onnx' : { # need post process changes
#         'model_path' : os.path.join(models_base_path, 'deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4.onnx'),
#         'source' : {'model_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models//vision/segmentation/ade20k32/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4.onnx', 'opt': False,  'infer_shape' : True},
#         'mean': [123.675, 116.28, 103.53],
#         'scale' : [0.017125, 0.017507, 0.017429],
#         'num_images' : numImages,
#         'num_classes': 19,
#         'session_name' : 'onnxrt' ,
#         'model_type': 'segmentation'
#     },

#     # Caffe Model - Would be converted ot ONNX
#     'cl-ort-caffe_mobilenet_v1' : {
#         'model_path' : os.path.join(models_base_path, 'caffe_mobilenet_v1.onnx'),
#         'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v1_prototext.link', 'opt': True,  'infer_shape' : False,
#                     'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v1_caffemodel.link', 
#                     'prototext' :   os.path.join(models_base_path, 'caffe_mobilenet_v1.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_mobilenet_v1.caffemodel') },
#         'mean': [103.94, 116.78, 123.68],
#         'scale' : [0.017125, 0.017507, 0.017429],
#         'num_images' : numImages,
#         'num_classes': 1000,
#         'session_name' : 'onnxrt' ,
#         'model_type': 'classification',
#         'original_model_type': 'caffe'
#     },
    
#     'cl-ort-caffe_mobilenet_v2' : {
#         'model_path' : os.path.join(models_base_path, 'caffe_mobilenet_v2.onnx'),
#         'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v2_prototext.link', 'opt': True,  'infer_shape' : False,
#                     'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/mobilenet/mobilenet_v2_caffemodel.link', 
#                     'prototext' :   os.path.join(models_base_path, 'caffe_mobilenet_v2.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_mobilenet_v2.caffemodel') },
#         'mean': [103.94, 116.78, 123.68],
#         'scale' : [0.017125, 0.017507, 0.017429],
#         'num_images' : numImages,
#         'num_classes': 1000,
#         'session_name' : 'onnxrt' ,
#         'model_type': 'classification',
#         'original_model_type': 'caffe'
#     },


#     'cl-ort-caffe_squeezenet_v1_1' : {
#         'model_path' : os.path.join(models_base_path, 'caffe_squeezenet_v1_1.onnx'),
#         'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/squeezenet/squeezenet_v1_1.prototext', 'opt': True,  'infer_shape' : False,
#                     'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/squeezenet/squeezenet_v1_1_caffemodel.link', 
#                     'prototext' :   os.path.join(models_base_path, 'caffe_squeezenet_v1_1.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_squeezenet_v1_1.caffemodel') },
#         'mean': [103.94, 116.78, 123.68],
#         'scale' : [1, 1, 1],
#         'num_images' : numImages,
#         'num_classes': 1000,
#         'session_name' : 'onnxrt' ,
#         'model_type': 'classification',
#         'original_model_type': 'caffe'
#     },

#     'cl-ort-caffe_resnet10' : {
#         'model_path' : os.path.join(models_base_path, 'caffe_resnet10.onnx'),
#         'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/resnet10/deploy.prototxt', 'opt': True,  'infer_shape' : False,
#                     'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/classification/imagenet1k/caffe/resnet10/resnet10_cvgj_iter_320000.caffemodel', 
#                     'prototext' :   os.path.join(models_base_path, 'caffe_resnet10.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_resnet10.caffemodel') },
#         'mean': [0,0,0],
#         'scale' : [1, 1, 1],
#         'num_images' : numImages,
#         'num_classes': 1000,
#         'session_name' : 'onnxrt' ,
#         'model_type': 'classification',
#         'original_model_type': 'caffe'
#     },

#     'cl-ort-caffe_mobilenetv1_ssd' : {
#         'model_path' : os.path.join(models_base_path, 'caffe_mobilenetv1_ssd.onnx'),
#         'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/mobilenet_ssd/mobilenet_v1_ssd_prototext.link', 'opt': False,  'infer_shape' : False,
#                     'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/mobilenet_ssd/mobilenet_v1_ssd_caffemodel.link', 
#                     'meta_arch_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/mobilenet_ssd/mobilenet_v1_ssd_meta.prototxt',
#                     'prototext' :   os.path.join(models_base_path, 'caffe_mobilenetv1_ssd.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_mobilenetv1_ssd.caffemodel') },
#         'mean': [103.94, 116.78, 123.68],
#         'scale' : [0.017125, 0.017507, 0.017429],
#         'num_images' : numImages,
#         'num_classes': 91,
#         'model_type': 'detection',
#         'od_type' : 'SSD',
#         'framework' : 'MMDetection',
#         'meta_layers_names_list' : os.path.join(models_base_path, 'caffe_mobilenetv1_ssd_meta.prototxt'),
#         'session_name' : 'onnxrt' ,
#         'meta_arch_type' : 3,
#         'original_model_type': 'caffe'
#     },

#     'cl-ort-caffe_pelee_ssd' : {
#         'model_path' : os.path.join(models_base_path, 'caffe_pelee_ssd.onnx'),
#         'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/pelee/pelee_ssd.prototxt', 'opt': False,  'infer_shape' : False,
#                     'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/pelee/pelee_ssd.caffemodel', 
#                     'meta_arch_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/detection/voc2012/caffe/pelee/pelee_ssd_meta.prototxt',
#                     'prototext' :   os.path.join(models_base_path, 'caffe_pelee_ssd.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_pelee_ssd.caffemodel') },
#         'mean': [103.94, 116.78, 123.68],
#         'scale' : [0.017125, 0.017507, 0.017429],
#         'num_images' : numImages,
#         'num_classes': 91,
#         'model_type': 'detection',
#         'od_type' : 'SSD',
#         'framework' : 'MMDetection',
#         'meta_layers_names_list' : os.path.join(models_base_path, 'caffe_pelee_ssd_meta.prototxt'),
#         'session_name' : 'onnxrt' ,
#         'meta_arch_type' : 3,
#         'original_model_type': 'caffe'
#     },

#     'cl-ort-caffe_erfnet' : {
#         'model_path' : os.path.join(models_base_path, 'caffe_erfnet.onnx'),
#         'source' : {'model_url': 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/segmentation/cityscapes/caffe/erfnet/erfnet.prototxt', 'opt': False,  'infer_shape' : False,
#                     'caffe_model_url' : 'https://github.com/kumardesappan/ai-model-zoo/raw/main/models/vision/segmentation/cityscapes/caffe/erfnet/erfnet_caffemodel.link', 
#                     'prototext' :   os.path.join(models_base_path, 'caffe_erfnet.prototxt'), 'caffe_model' : os.path.join(models_base_path,'caffe_erfnet.caffemodel') },
#         'mean': [0,0,0],
#         'scale' : [1,1,1],
#         'num_images' : numImages,
#         'num_classes': 19,
#         'session_name' : 'onnxrt' ,
#         'model_type': 'segmentation',
#         'original_model_type': 'caffe'
#     },
# }

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
        'model_type': 'segmentation'
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'detection',
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
        'model_type': 'segmentation'
    },

    'fpnlite_aspp_mobilenetv2' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_mobilenetv2_768x384_20200120.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'segmentation'
    },
    'unetlite_aspp_mobilenetv2' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/unetlite_aspp_mobilenetv2_768x384_20200129.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'segmentation'
    },    
    'fpnlite_aspp_regnetx800mf' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_regnetx800mf_768x384_20200911.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'segmentation'
    },    
    'fpnlite_aspp_regnetx1.6gf' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_regnetx1.6gf_1024x512_20200914.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'segmentation'
    },    
    'fpnlite_aspp_regnetx3.2gf' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/edgeai-jai/fpnlite_aspp_regnetx3.2gf_1024x512_20200916.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'segmentation'
    }, 
    'deeplabv3_resnet50_1040x520' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/torchvision/deeplabv3_resnet50_1040x520_20200901-213517.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'segmentation'
    },
    'fcn_resnet50_1040x520' : {
        'model_path' : os.path.join(modelforest_path, 'vision/segmentation/cityscapes/torchvision/ffcn_resnet50_1040x520_20200902-153444.onnx'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 21,
        'session_name' : 'onnxrt' ,
        'model_type': 'segmentation'
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
        'model_type': 'detection',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v2_coco_2018_03_29' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_coco_2018_03_29.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'detection',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },

    'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'detection',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v2_320x320_coco17_tpu-8' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_320x320_coco17_tpu-8.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'detection',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'scale' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'detection',
        'session_name' : 'tflitert',
        'od_type' : 'HasDetectionPostProcLayer'
    },
    'efficientdet-ti-lite0_k5s1_k3s2' : {
        'model_path' : os.path.join(models_base_path,'efficientdet-ti-lite0_k5s1_k3s2.tflite'),
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.01712475, 0.017507, 0.01742919],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'detection',
        'od_type' : 'EfficientDetLite',
        'meta_layers_names_list' : '../testvecs/models/public/tflite/efficientdet-ti-lite0.prototxt',
        'session_name' : 'tflitert',
        'meta_arch_type' : 5
    },
"""

