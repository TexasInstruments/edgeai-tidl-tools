import time
import platform
import os
import sys
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-z','--run_model_zoo', action='store_true',  help='Run model zoo models')
args = parser.parse_args()
# directory reach
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
# setting path
sys.path.append(parent)
from common_utils import *
from model_configs import *

output_images_folder = '../../../output_images/'

if platform.machine() == 'aarch64':
    numImages = 100
else : 
    numImages = 3

model_input_height = 299
model_input_width = 299

# preprocessing / postprocessing for tflite model
def preprocess_for_tflite_inceptionnetv3(image_path):
    import cv2
    import numpy as np

    # read the image using openCV
    img = cv2.imread(image_path)
    # convert to RGB
    img = img[:,:,::-1]
    
    # This TFLite model is trained using 299x299 images.
    # The general rule of thumb for classification models
    # is to scale the input image while preserving
    # the original aspect ratio, so we scale the short edge
    # to 299 pixels, and then
    # center-crop the scaled image to 224x224
    orig_height, orig_width, _ = img.shape
    short_edge = min(img.shape[:2])
    new_height = (orig_height * model_input_height) // short_edge
    new_width = (orig_width * model_input_width) // short_edge
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    startx = new_width//2 - (model_input_width//2)
    starty = new_height//2 - (model_input_height//2)
    img = img[starty:starty+model_input_height,startx:startx+model_input_width]
    
    # apply scaling and mean subtraction.
    # if your model is built with an input
    # normalization layer, then you might
    # need to skip this
    img = img.astype('uint8')
    #for mean, scale, ch in zip([128, 128, 128], [0.0078125, 0.0078125, 0.0078125], range(img.shape[2])):
    #        img[:,:,ch] = ((img[:,:,ch] - mean) * scale)
    # convert HWC to NHWC
    img = np.expand_dims(img, axis=0)
    return img

def postprocess_for_tflite_inceptionnetv3(res):
    return res[0].flatten()[1:]

# preprocessing / postprocessing for onnx model 
def preprocess_for_onnx_mobilenetv2(image_path):
    import cv2
    import numpy as np
    # read the image using openCV
    img = cv2.imread(image_path)
    # convert to RGB
    img = img[:,:,::-1]
    
    # Most of the onnx models are trained using
    # 224x224 images. The general rule of thumb
    # is to scale the input image while preserving
    # the original aspect ratio so that the
    # short edge is 256 pixels, and then
    # center-crop the scaled image to 224x224
    orig_height, orig_width, _ = img.shape
    short_edge = min(img.shape[:2])
    new_height = (orig_height * 256) // short_edge
    new_width = (orig_width * 256) // short_edge
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    startx = new_width//2 - (224//2)
    starty = new_height//2 - (224//2)
    img = img[starty:starty+224,startx:startx+224]
    
    # apply scaling and mean subtraction.
    # if your model is built with an input
    # normalization layer, then you might
    # need to skip this
    img = img.astype('uint8')
    #for mean, scale, ch in zip([123.675, 116.28, 103.53], [0.017125, 0.017507, 0.017429], range(img.shape[2])):
    #        img[:,:,ch] = ((img.astype('float32')[:,:,ch] - mean) * scale)
     
    # convert HWC to NCHW
    img = np.expand_dims(np.transpose(img, (2,0,1)),axis=0)
    
    return img

def postprocess_for_onnx_mobilenetv2(res):
    return res[0].flatten()

# preprocessing / postprocessing for mxnet model
def preprocess_for_mxnet_mobilenetv3(image_path):
    import cv2
    import numpy as np

    # read the image using openCV
    img = cv2.imread(image_path)

    # convert to RGB
    img = img[:,:,::-1]

    # Most of the onnx models are trained using
    # 224x224 images. The general rule of thumb
    # is to scale the input image while preserving
    # the original aspect ratio so that the
    # short edge is 256 pixels, and then
    # center-crop the scaled image to 224x224
    orig_height, orig_width, _ = img.shape
    short_edge = min(img.shape[:2])
    new_height = (orig_height * 256) // short_edge
    new_width = (orig_width * 256) // short_edge
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    startx = new_width//2 - (224//2)
    starty = new_height//2 - (224//2)
    img = img[starty:starty+224,startx:startx+224]

    # apply scaling and mean subtraction.
    # if your model is built with an input
    # normalization layer, then you might
    # need to skip this
    img = img.astype('float32')
    for mean, scale, ch in zip([128.0, 128.0, 128.0], [0.0078125, 0.0078125, 0.0078125], range(img.shape[2])):
            img[:,:,ch] = ((img.astype('float32')[:,:,ch] - mean) * scale)

    # convert HWC to NCHW
    img = np.expand_dims(np.transpose(img, (2,0,1)),axis=0)

    return img

def model_create_and_run(model_dir,
                            model_input_name,
                            preprocess_func,
                            postprocess_func, mIdx):
    from dlr import DLRModel
    import numpy
    print(f'\n\nRunning Inference on Model -  {model_dir}\n')

    model = DLRModel(model_dir, 'cpu')
    test_files = ['../../../test_data/airshow.jpg']

    proc_time = 0.0
    for i in range(numImages):
        img_path = test_files[i%len(test_files)]
        img = preprocess_func(img_path)
        start_time = time.time()
        res = model.run({model_input_name : img})
        stop_time = time.time()
        proc_time += (stop_time - start_time)*1000
    
    print(f'\n Processing time in ms : {proc_time/numImages:10.1f}\n')

    res = postprocess_func(res)
    res_np = np.array(res)

    model = model_dir.split("/")[-2]
    output_image_file_name = "py_out_"+model+'_'+os.path.basename(img_path)

    image_pil = Image.open(img_path).convert('RGB')
    classes, image = get_class_labels(res,image_pil)
    print(classes)

    print("\nSaving image to ", output_images_folder)
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)
    image.save(output_images_folder + output_image_file_name, "JPEG") 
    print("\nSaving output tensor to ", output_binary_folder)
    if not os.path.exists(output_binary_folder):
        os.makedirs(output_binary_folder)
    output_bin_file_name = output_image_file_name.replace(".jpg","") + ".bin"
    res_np.tofile(output_binary_folder + output_bin_file_name)
    
    log = f'\n \nCompleted_Model : {mIdx+1:5d}, Name : {model:50s}, Total time : {proc_time/numImages:10.2f}, Offload Time : {proc_time/numImages:10.2f} , DDR RW MBs : 0, Output Image File : {output_image_file_name}, Output Bin File : {output_bin_file_name}\n \n ' #{classes} \n \n'
    print(log) 

model_output_directory = '../../../model-artifacts/cl-dlr-tflite_inceptionnetv3'
if platform.machine() == 'aarch64':
    model_output_directory = model_output_directory+'_device'
model_output_directory = model_output_directory+'/artifacts'

model_create_and_run(model_output_directory, 'input',
                        preprocess_for_tflite_inceptionnetv3,
                        postprocess_for_tflite_inceptionnetv3, 0)

model_output_directory = '../../../model-artifacts/cl-dlr-onnx_mobilenetv2'
if ( args.run_model_zoo ):
    model_output_directory = '../../../model-artifacts/cl-3090_tvmdlr_imagenet1k_torchvision_mobilenet_v2_tv_onnx'
if platform.machine() == 'aarch64':
    model_output_directory = model_output_directory+'_device'
model_output_directory = model_output_directory+'/artifacts'

model_create_and_run(model_output_directory, 'input.1Net_IN',
                        preprocess_for_onnx_mobilenetv2,
                        postprocess_for_onnx_mobilenetv2, 1)

if platform.machine() == 'aarch64':
    model_output_directory = '../../../model-artifacts/cl-dlr-timm_mobilenetv3_large_100_device/artifacts'
    model_create_and_run(model_output_directory, 'input.1',
                        preprocess_for_mxnet_mobilenetv3,
                        postprocess_for_onnx_mobilenetv2, 1)

    model_output_directory = '../../../model-artifacts/cl-dlr-timm_mobilenetv3_large_100_device_c7x/artifacts'
    model_create_and_run(model_output_directory, 'input.1',
                        preprocess_for_mxnet_mobilenetv3,
                        postprocess_for_onnx_mobilenetv2, 1)
