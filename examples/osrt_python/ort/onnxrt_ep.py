import onnxruntime as rt
import time
import os
import sys
import numpy as np
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import argparse
import re
import multiprocessing
import platform
import shutil
#import onnx
# directory reach
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
# setting path
sys.path.append(parent)
from common_utils import *
from model_configs import *

model_optimizer_found = False
if platform.machine() != 'aarch64':
    try:
        from osrt_model_tools.onnx_tools.tidl_onnx_model_optimizer import optimize
        model_optimizer_found = True
    except ModuleNotFoundError as e:
        print("Skipping import of model optimizer")

required_options = {
"tidl_tools_path":tidl_tools_path,
"artifacts_folder":artifacts_folder
}

parser = argparse.ArgumentParser()
parser.add_argument('-c','--compile', action='store_true', help='Run in Model compilation mode')
parser.add_argument('-d','--disable_offload', action='store_true',  help='Disable offload to TIDL')
parser.add_argument('-z','--run_model_zoo', action='store_true',  help='Run model zoo models')
parser.add_argument('-o','--graph_optimize', action='store_true',  help='Run ONNX model optimization thourgh onnx-graph-surgeon-tidl')
parser.add_argument('-m','--models', action='append', default=[], help='Model name to be added to the list to run')
args = parser.parse_args()
os.environ["TIDL_RT_PERFSTATS"] = "1"

so = rt.SessionOptions()
so.log_severity_level=3


print("Available execution providers : ", rt.get_available_providers())

calib_images = ['../../../test_data/airshow.jpg',
                '../../../test_data/ADE_val_00001801.jpg']
class_test_images = ['../../../test_data/airshow.jpg']
od_test_images    = ['../../../test_data/ADE_val_00001801.jpg']
seg_test_images   = ['../../../test_data/ADE_val_00001801.jpg']


sem = multiprocessing.Semaphore(0)
if platform.machine() == 'aarch64':
    ncpus = 1
else:
    ncpus = os.cpu_count()

#ncpus = 1
idx = 0
nthreads = 0
run_count = 0

if "SOC" in os.environ:
    SOC = os.environ["SOC"]
else:
    print("Please export SOC var to proceed")
    exit(-1)

if (platform.machine() == 'aarch64'  and args.compile == True):
    print("Compilation of models is only supported on x86 machine \n\
        Please do the compilation on PC and copy artifacts for running on TIDL devices " )
    exit(-1)

if(SOC == "am62"):
    args.disable_offload = True
    args.compile = False

def get_benchmark_output(interpreter):
    benchmark_dict = interpreter.get_TI_benchmark_data()
    proc_time = copy_time = 0
    cp_in_time = cp_out_time = 0
    subgraphIds = []
    for stat in benchmark_dict.keys():
        if 'proc_start' in stat:
            value = stat.split("ts:subgraph_")
            value = value[1].split("_proc_start")
            subgraphIds.append(value[0])
    for i in range(len(subgraphIds)):
        proc_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_start']
        cp_in_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_start']
        cp_out_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_start']
        copy_time += cp_in_time + cp_out_time
    copy_time = copy_time if len(subgraphIds) == 1 else 0
    totaltime = benchmark_dict['ts:run_end'] -  benchmark_dict['ts:run_start']
    return copy_time, proc_time, totaltime


def infer_image(sess, image_files, config):
  input_details = sess.get_inputs()
  input_name = input_details[0].name
  floating_model = (input_details[0].type == 'tensor(float)')
  height = input_details[0].shape[2]
  width  = input_details[0].shape[3]
  channel = input_details[0].shape[1]
  batch  = input_details[0].shape[0]
  imgs= []
  shape = [batch, channel, height, width]
  input_data = np.zeros(shape)
  for i in range(batch):
      imgs.append(Image.open(image_files[i]).convert('RGB').resize((width, height), PIL.Image.LANCZOS))
      temp_input_data = np.expand_dims(imgs[i], axis=0)
      temp_input_data = np.transpose(temp_input_data, (0, 3, 1, 2))
      input_data[i] = temp_input_data[0]
  if floating_model:
    input_data = np.float32(input_data)
    for mean, scale, ch in zip(config['mean'], config['scale'], range(input_data.shape[1])):
        input_data[:,ch,:,:] = ((input_data[:,ch,:,:]- mean) * scale)
  else:
    input_data = np.uint8(input_data)
    config['mean'] = [0, 0, 0]
    config['scale']  = [1, 1, 1]

  start_time = time.time()
  output = list(sess.run(None, {input_name: input_data}))
  #output = list(sess.run(None, {input_name: input_data, input_details[1].name: np.array([[416,416]], dtype = np.float32)}))

  stop_time = time.time()
  infer_time = stop_time - start_time

  copy_time, sub_graphs_proc_time, totaltime = get_benchmark_output(sess)
  proc_time = totaltime - copy_time

  return imgs, output, proc_time, sub_graphs_proc_time, height, width

def run_model(model, mIdx):
    print("\nRunning_Model : ", model, " \n")
    if platform.machine() != 'aarch64':
        download_model(models_configs, model)
    config = models_configs[model]

    # graph optimization
    if args.graph_optimize:
        if model_optimizer_found:
            if (args.compile or args.disable_offload) and (platform.machine() != 'aarch64'):
                copy_path = config['model_path'][:-5] + "_org.onnx"
                # check if copy path exists and prompt for permission to overwrite
                if os.path.isfile(copy_path):
                    overwrite_permission = input(f"\033[96mThe file {copy_path} exists, do you want to overwrite? [Y/n] \033[00m")
                    if overwrite_permission != 'Y':
                        print("Aborting run...")
                        sys.exit(-1)
                    else:
                        print(f"\033[93m[WARNING] File {copy_path} will be overwritten\033[00m")

                shutil.copy2(config['model_path'], copy_path)
                print(f"\033[93mOptimization Enabled: Moving {config['model_path']} to {copy_path} before overwriting by optimization\033[00m")
                optimize(model= config['model_path'], out_model= config['model_path'])  # verbose=True for more details
            else:
                print("model optimization is only supported in compilation or disabled offload mode on x86 machines")
        else:
            print("model optimizer not found, -o flag has no effect")


    #set input images for demo
    config = models_configs[model]
    if config['model_type'] == 'classification':
        test_images = class_test_images
    elif config['model_type'] == 'od':
        test_images = od_test_images
    elif config['model_type'] == 'seg':
        test_images = seg_test_images

    delegate_options = {}
    delegate_options.update(required_options)
    delegate_options.update(optional_options)
    if 'optional_options' in config:
        delegate_options.update(config['optional_options'])


    # stripping off the ss-ort- from model namne
    delegate_options['artifacts_folder'] = delegate_options['artifacts_folder'] + '/' + model + '/' #+ 'tempDir/'

    # disabling onnxruntime optimizations for vision transformers
    if (model == 'cl-ort-deit-tiny'):
        so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    if config['model_type'] == 'od':
        delegate_options['object_detection:meta_layers_names_list'] = config['meta_layers_names_list'] if ('meta_layers_names_list' in config) else ''
        delegate_options['object_detection:meta_arch_type'] = config['meta_arch_type'] if ('meta_arch_type' in config) else -1
    
    # delete the contents of this folder
    if args.compile or args.disable_offload:
        os.makedirs(delegate_options['artifacts_folder'], exist_ok=True)
        for root, dirs, files in os.walk(delegate_options['artifacts_folder'], topdown=False):
            [os.remove(os.path.join(root, f)) for f in files]
            [os.rmdir(os.path.join(root, d)) for d in dirs]

    if(args.compile == True):
        input_image = calib_images
        import onnx
        log = f'\nRunning shape inference on model {config["model_path"]} \n'
        print(log)
        onnx.shape_inference.infer_shapes_path(config['model_path'], config['model_path'])
    else:
        input_image = test_images

    numFrames = config['num_images']
    if(args.compile):
        if numFrames > delegate_options['advanced_options:calibration_frames']:
            numFrames = delegate_options['advanced_options:calibration_frames']

    ############   set interpreter  ################################
    if args.disable_offload :
        EP_list = ['CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] , providers=EP_list,sess_options=so)
    elif args.compile:
        EP_list = ['TIDLCompilationProvider','CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[delegate_options, {}], sess_options=so)
    else:
        EP_list = ['TIDLExecutionProvider','CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[delegate_options, {}], sess_options=so)
    ################################################################

    # run session
    for i in range(numFrames):
        #img, output, proc_time, sub_graph_time = infer_image(sess, input_image[i%len(input_image)], config)
        start_index = i%len(input_image)
        input_details = sess.get_inputs()
        batch = input_details[0].shape[0]
        input_images = []
        # for batch processing diff image needed for a single  input
        for j in range(batch):
            input_images.append(input_image[(start_index+j)%len(input_image)])
        imgs, output, proc_time, sub_graph_time, height, width  = infer_image(sess, input_images, config)
        total_proc_time = total_proc_time + proc_time if ('total_proc_time' in locals()) else proc_time
        sub_graphs_time = sub_graphs_time + sub_graph_time if ('sub_graphs_time' in locals()) else sub_graph_time

    total_proc_time = total_proc_time /1000000
    sub_graphs_time = sub_graphs_time/1000000

    # output post processing
    output_file_name = "py_out_"+model+'_'+os.path.basename(input_image[i%len(input_image)])
    if(args.compile == False):  # post processing enabled only for inference
        images = []
        if config['model_type'] == 'classification':
            for j in range(batch):
                classes, image = get_class_labels(output[0][j],imgs[j])
                print("\n", classes)
                images.append(image)
        elif config['model_type'] == 'od':
             for j in range(batch):
                classes, image = det_box_overlay(output, imgs[j], config['od_type'], config['framework'])
                images.append(image)

        elif config['model_type'] == 'seg':
            for j in range(batch):
                imgs[j] = imgs[j].resize((output[0][j].shape[-1], output[0][j].shape[-2]),PIL.Image.LANCZOS)
                classes, image = seg_mask_overlay(output[0][j],imgs[j])
                images.append(image)
        else:
            print("Not a valid model type")
        for j in range(batch):
            output_file_name = "py_out_"+model+'_'+os.path.basename(input_images[j])
            print("\nSaving image to ", output_images_folder)
            if not os.path.exists(output_images_folder):
                os.makedirs(output_images_folder)
            images[j].save(output_images_folder + output_file_name, "JPEG")

    if args.compile or args.disable_offload :
        gen_param_yaml(delegate_options['artifacts_folder'], config, int(height), int(width))
    log = f'\n \nCompleted_Model : {mIdx+1:5d}, Name : {model:50s}, Total time : {total_proc_time/(i+1):10.2f}, Offload Time : {sub_graphs_time/(i+1):10.2f} , DDR RW MBs : 0, Output File : {output_file_name} \n \n ' #{classes} \n \n'
    print(log)
    if ncpus > 1:
        sem.release()


#models = models_configs.keys()
if len(args.models) > 0:
    models = args.models
else :
    models = ['cl-ort-resnet18-v1', 'od-ort-ssd-lite_mobilenetv2_fpn']
    if(SOC == "am69a"):
        models.append('cl-ort-resnet18-v1_4batch') # Model to demonstrate multi core parallel batch processing
        models.append('cl-ort-resnet18_1MP_low_latency') # Model to demonstrate multi core low latency inference
    if(SOC not in ("am62a","am67a")):
        models.append('ss-ort-deeplabv3lite_mobilenetv2')

if ( args.run_model_zoo ):
    models = [
             'od-8020_onnxrt_coco_edgeai-mmdet_ssd_mobilenetv2_lite_512x512_20201214_model_onnx',
             'od-8200_onnxrt_coco_edgeai-mmdet_yolox_nano_lite_416x416_20220214_model_onnx',
            #  'od-8420_onnxrt_widerface_edgeai-mmdet_yolox_s_lite_640x640_20220307_model_onnx',# not working -
             'ss-8610_onnxrt_ade20k32_edgeai-tv_deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4_onnx',
             'od-8220_onnxrt_coco_edgeai-mmdet_yolox_s_lite_640x640_20220221_model_onnx',
             'cl-6360_onnxrt_imagenet1k_fbr-pycls_regnetx-200mf_onnx'
            ]
log = f'\nRunning {len(models)} Models - {models}\n'
print(log)

def join_one(nthreads):
    global run_count
    sem.acquire()
    run_count = run_count + 1
    return nthreads - 1

def spawn_one(models, idx, nthreads):
    p = multiprocessing.Process(target=run_model, args=(models,idx,))
    p.start()
    return idx + 1, nthreads + 1

if ncpus > 1:
    for t in range(min(len(models), ncpus)):
        idx, nthreads = spawn_one(models[idx], idx, nthreads)

    while idx < len(models):
        nthreads = join_one(nthreads)
        idx, nthreads = spawn_one(models[idx], idx, nthreads)

    for n in range(nthreads):
        nthreads = join_one(nthreads)
else :
    for mIdx, model in enumerate(models):
        run_model(model, mIdx)



"""
models = [#'mlperf_ssd_resnet34-ssd1200',
          'retinanet-lite_regnetx-800mf_fpn_bgr_512x512_20200908_model',
          'ssd-lite_mobilenetv2_512x512_20201214_220055_model',
          'ssd-lite_mobilenetv2_fpn_512x512_20201110_model',
          'ssd-lite_mobilenetv2_qat-p2_512x512_20201217_model',
          'ssd-lite_regnetx-1.6gf_bifpn168x4_bgr_768x768_20201026_model',
          'ssd-lite_regnetx-200mf_fpn_bgr_320x320_20201010_model',
          'ssd-lite_regnetx-800mf_fpn_bgr_512x512_20200919_model',
          'yolov3-lite_regnetx-1.6gf_bgr_512x512_20210202_model',
          #'yolov5s_ti_lite_35p0_54p5',
          'yolov5s6_640_ti_lite_37p4_56p0',
          'yolov5m6_640_ti_lite_44p1_62p9',
          #'ssd_resnet_fpn_512x512_20200730-225222_model',
          #'yolov3_d53_relu_416x416_20210117_004118_model',
          #'yolov3_d53_416x416_20210116_005003_model'
          ]
"""
