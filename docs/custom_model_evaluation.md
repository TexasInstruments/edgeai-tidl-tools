# Compile and Benchmark Custom Model
<!-- TOC -->

- [Compile and Benchmark Custom Model](#compile-and-benchmark-custom-model)
  - [New Model Evaluation](#new-model-evaluation)
  - [Custom Model Evaluation](#custom-model-evaluation)

<!-- /TOC -->
## New Model Evaluation
- The custom model that needs to be evaluated is falling into one of the below supported out-of-box example tasks categories, then the python scripts in this repository can be used as it is by following the steps described in this section
  - Image classification
  - Object detection
  - Pixel level semantic Segmentation
- Complete model evaluation process (both compilation and Inference) can be carried out by using the python API. 
- Refer the documentation available [here](examples/osrt_python/README.md) to familiarize with the steps to compile the out-of-box models and all the available compilation options for TIDL offload.
- [Models Dictionary](examples/osrt_python/model_configs.py) in the python examples directory lists all the validated models with this repository
- Define a entry for your custom model in this dictionary and add the new key in the model list of the python script based on your model format and runtime, for example - to evaluate a Tflite model update below entry [here](examples/osrt_python/tfl/tflrt_delegate.py)

 ```
#models = ['cl-tfl-mobilenet_v1_1.0_224', 'ss-tfl-deeplabv3_mnv2_ade20k_float', 'od-tfl-ssd_mobilenet_v2_300_float']
models = ['cl-tfl-custom-model']

```

- An example dictionary entry of an existing model for reference

 ```
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

```


## Custom Model Evaluation

Coming soon