# Compile and Benchmark Custom Model
<!-- TOC -->

- [Compile and Benchmark Custom Model](#compile-and-benchmark-custom-model)
  - [New Model Evaluation](#new-model-evaluation)
  - [Custom Model Evaluation](#custom-model-evaluation)
    - [OSRT APIs for TIDL Acceleration](#osrt-apis-for-tidl-acceleration)
      - [TFLite Runtime](#tflite-runtime)
      - [ONNX Runtime](#onnx-runtime)
      - [TVM Compiler](#tvm-compiler)
  - [Reporting issues with Model deployment](#reporting-issues-with-model-deployment)

<!-- /TOC -->

## New Model Evaluation
- Refer to this section if the custom model you are evaluating falls into one of the below supported out-of-box example tasks categories. For such cases the example python scripts in this repository can be used as it is by following the steps described in this section
  - Image classification
  - Object detection
  - Pixel level semantic Segmentation
- Complete model evaluation process (both compilation and Inference) can be carried out by using the out of box python examples 
- Refer to the documentation available [here](../examples/osrt_python/README.md) to familiarize with the steps to compile the out-of-box models and all the available compilation options for TIDL offload
- [Models Dictionary](../examples/osrt_python/model_configs.py) in the python examples directory lists all the validated models with this repository
- Define an entry for your custom model in this dictionary and add the new key in the model list of the python script based on your model format and runtime, for example - to evaluate a Tflite model update below entry [here](../examples/osrt_python/tfl/tflrt_delegate.py)
  - Preprocessing parameters, i.e. ```mean``` and ```scale``` must match what was used during training to achieve accurate results. Values in the below config are typical for [models started with an imagenet-trained backbone](https://github.com/pytorch/examples/blob/cdef4d43fb1a2c6c4349daa5080e4e8731c34569/imagenet/main.py#L236) (scaled for uint8 inputs instead of [0,1]). These values do not originate from TIDL or related TI tools


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
        'model_type': 'classification',
        'optional_options': {
          #include any additional configurations for this model only, e.g.:
          'debug_level': 1,
        }.
    },

```
- Append ```-m model_name``` to your python command to easily select your model with the osrt_python scripts
- As a first step, run the model with default OSRT runtime options (without TIDL acceleration) by passing ```-d``` argument as described [here](../examples/osrt_python/README.md). A right functional result from this step confirms your model configuration dictionary is right and model is working fine with the out of box example code.
- Now model compilation and inference steps can be executed.

## Custom Model Evaluation

- The first and important step in custom model deployment is writing python inference code for your custom model. The user may need to write this python code either using TFlite runtime or ONNX runtime based on the model type.
- User can refer the official documentation from OSRT or a simple Colab notebook for end-to-end working reference in the below table as a starting point to create inference script for their custom model.

<div align="center">

| Official Python API documentation | Simple Colab notebook for end-to-end working reference |
|----|---|
| [**TFLite Python API**](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python) | [**TFLite Colab Notebook**](../examples/jupyter_notebooks/colab/infer_tflite.ipynb)|
| [**ONNX Runtime Python API**](https://onnxruntime.ai/docs/get-started/with-python.html)|[**ONNX Runtime Colab Notebook**](../examples/jupyter_notebooks/colab/infer_ort.ipynb)|

</div>

- Validate the inference code for functionality with couple of input samples and required Pre and Post processing. 
  - **Note - Please continue with following steps, only after getting right functional results from this step**
- Update the inference script to compile the model with TIDL acceleration by passing required compilation options. Refer [here](../examples/osrt_python/README.md#basic-options) for detailed documentation on all the required and optional parameters.
- Run the python code with  compilation options using representative input data samples for model compilation and calibration. 
  - Default options expects minimum 20 input data samples (```calibration_frames```) for calibration. User can set as minimum as 1 also for quick model compilation (This may impact the accuracy of fixed point inference).
- At the end of model compilation step, model-artifacts for inference will be generated in user specified path.
- Create OSRT inference session with TIDL acceleration option for running inference with generated model artifacts in the above step.
  - User can either update existing python code written for compilation or copy the compilation code to new file and update with accelerated inference option.
- Refer the below tables for creating OSRT sessions with Compilation and Accelerated inference options.

### OSRT APIs for TIDL Acceleration

#### TFLite Runtime 

<div align="center">

| Session Name | API and Options to Create Session |
|-------------|----------------------------------|
| **Default RT Session** | ``` tflite.Interpreter(model_path=config['model_path']) ``` |
| **RT Session Model Compilation**        | ``` options['artifacts_folder'] = './model-artifacts-dir/' ``` <br /> ``` options['tidl_tools_path'] = './path-to-tidl_tools/' ``` <br /> ```# include additional ```**[compilation options](../examples/osrt_python/README.md#optional-options)**```like tensor_bits, deny_list, etc.```<br /> <br /> ```tflite.Interpreter(model_path=config['model_path'], experimental_delegates=[tflite.load_delegate('tidl_model_import_tflite.so', options)]) ```|
| **RT Session with TIDL acceleration** |``` options['artifacts_folder'] = './model-artifacts-dir/' ``` <br /> ``` options['tidl_tools_path'] = './path-to-tidl_tools/' #only needed for emulation on x86 host PC``` <br /> <br />  ```tflite.Interpreter(model_path=config['model_path'], experimental_delegates=[tflite.load_delegate('libtidl_tfl_delegate.so', options)]) ``` |

</div>


#### ONNX Runtime 


<div align="center">



| Session Name | API and Options to Create Session |
|-------------|----------------------------------|
| **Default RT Session** | ```so = rt.SessionOptions()``` <br /> ```ep_list = ['CPUExecutionProvider']``` <br /> ``` sess = rt.InferenceSession(config['model_path'] , providers=ep_list,sess_options=so)```|
| **RT Session Model Compilation**        |   ``` options['artifacts_folder'] = './model-artifacts-dir/' ``` <br /> ``` options['tidl_tools_path'] = './path-to-tidl_tools/' ```  <br /> ```# include additional ```**[compilation options](../examples/osrt_python/README.md#optional-options)**```like tensor_bits, deny_list, etc.```<br /> <br /> ```so = rt.SessionOptions()``` <br /> ```ep_list = ['TIDLCompilationProvider','CPUExecutionProvider']``` <br /> ``` sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[options, {}], sess_options=so)```|
| **RT Session with TIDL acceleration** |``` options['artifacts_folder'] = './model-artifacts-dir/' ``` <br /> ``` options['tidl_tools_path'] = './path-to-tidl_tools/' #only needed for emulation on x86 host PC ``` <br /> <br /> ```so = rt.SessionOptions()``` <br /> ```ep_list = ['TIDLExecutionProvider','CPUExecutionProvider']``` <br /> ``` sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[options, {}], sess_options=so)``` |

</div>


#### TVM Compiler 


<div align="center">


| Session Name | API and Options to Create Session |
|-------------|----------------------------------|
| **Default TVM Compiler** | ```with tvm.transform.PassContext(opt_level=3):``` <br > &emsp; ```graph, lib, params = relay.build(mod, target=build_target, params=params)```|
| **TVM Compiler for TIDL accelartion**        |   ```compiler = tidl.TIDLCompiler(platform="J7" ,``` <br /> ```tidl_tools_path  = './path-to-tidl_tools/',```    <br />      ``` artifacts_folder = './model-artifacts-dir/'```, <br />  ``` advanced_options = {'calibration_iterations' : 1},)``` <br />  ```with tidl.build_config(tidl_compiler=compiler):``` <br /> &emsp; ```graph, lib, params = relay.build_module.build(mod, target=build_target, params=params) ``` |

</div>

        



- User can also refer the out of box python examples provided [here](../examples/osrt_python/) to understand the APIs and flow 

## Reporting issues with Model deployment
-	Please refer the steps detailed out [Troubleshooting Guide](./tidl_osr_debug.md) for debugging any functional and performance issue
-	If the issues could not be resolved with the above Troubleshooting Guide, please share below details to reproduce the issue
    - Python code used for Model compilation and inference
    - A representation model (ONNX or Tflite file) – Need not to be exact model, trainable parameters can be random as well
    - Representative input data samples for model compilation/calibration
    - Complete console log of both model compilation and inference with ``` debug_level=1``` and ``` debug_level=3```
    - Please indicate where the issue is encountered (compilation or inference; host emulation or target) and what type of problem (model accuracy, model performance, associated scripts, etc.)
