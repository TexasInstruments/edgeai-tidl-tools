import onnxruntime as rt
import time
import os
import numpy as np
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from utils import *
import argparse
import re
import multiprocessing
import platform
#import onnx

parser = argparse.ArgumentParser()
parser.add_argument('-c','--compile', action='store_true', help='Run in Model compilation mode')
parser.add_argument('-d','--disable_offload', action='store_true',  help='Disable offload to TIDL')
args = parser.parse_args()
os.environ["TIDL_RT_PERFSTATS"] = "1"

so = rt.SessionOptions()

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


def infer_image(sess, image_file, config):
  input_details = sess.get_inputs()
  input_name = input_details[0].name
  floating_model = (input_details[0].type == 'tensor(float)')
  height = input_details[0].shape[2]
  width  = input_details[0].shape[3]
  img    = Image.open(image_file).convert('RGB').resize((width, height), PIL.Image.LANCZOS)
  #img    = Image.open(image_file).convert('RGB').resize((416,416))
  input_data = np.expand_dims(img, axis=0)
  input_data = np.transpose(input_data, (0, 3, 1, 2))

  if floating_model:
    input_data = np.float32(input_data)
    for mean, scale, ch in zip(config['mean'], config['std'], range(input_data.shape[1])):
        input_data[:,ch,:,:] = ((input_data[:,ch,:,:]- mean) * scale)
  
  start_time = time.time()
  output = list(sess.run(None, {input_name: input_data}))
  #output = list(sess.run(None, {input_name: input_data, input_details[1].name: np.array([[416,416]], dtype = np.float32)}))

  stop_time = time.time()
  infer_time = stop_time - start_time

  copy_time, sub_graphs_proc_time, totaltime = get_benchmark_output(sess)
  proc_time = totaltime - copy_time

  return img, output, proc_time, sub_graphs_proc_time, height, width

def run_model(model, mIdx):
    print("\nRunning_Model : ", model, " \n")
    config = models_configs[model]

    #onnx shape inference
    #if not os.path.isfile(os.path.join(models_base_path, model + '_shape.onnx')):
    #    print("Writing model with shapes after running onnx shape inference -- ", os.path.join(models_base_path, model + '_shape.onnx'))
    #    onnx.shape_inference.infer_shapes_path(config['model_path'], config['model_path'])#os.path.join(models_base_path, model + '_shape.onnx'))
    
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

    # stripping off the ss-ort- from model namne
    model_name = model[7:]
    delegate_options['artifacts_folder'] = delegate_options['artifacts_folder'] + '/' + model_name + '/' #+ 'tempDir/' 

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
        img, output, proc_time, sub_graph_time, height, width = infer_image(sess, input_image[i%len(input_image)], config)
        total_proc_time = total_proc_time + proc_time if ('total_proc_time' in locals()) else proc_time
        sub_graphs_time = sub_graphs_time + sub_graph_time if ('sub_graphs_time' in locals()) else sub_graph_time
    
    total_proc_time = total_proc_time /1000000
    sub_graphs_time = sub_graphs_time/1000000

    # output post processing
    output_file_name = "post_proc_out_"+model+'_'+os.path.basename(input_image[i%len(input_image)])
    if(args.compile == False):  # post processing enabled only for inference
        if config['model_type'] == 'classification':
            classes, image = get_class_labels(output[0],img)
            print("\n", classes)
        elif config['model_type'] == 'od':
            classes, image = det_box_overlay(output, img, args.disable_offload, config['od_type'], config['framework'])
        elif config['model_type'] == 'seg':
            classes, image = seg_mask_overlay(output[0], img)
        else:
            print("Not a valid model type")

        print("\nSaving image to ", output_images_folder)
        if not os.path.exists(output_images_folder):
            os.makedirs(output_images_folder)
        image.save(output_images_folder + output_file_name, "JPEG") 
    gen_param_yaml(delegate_options, config, int(height), int(width))
    log = f'\n \nCompleted_Model : {mIdx+1:5d}, Name : {model:50s}, Total time : {total_proc_time/(i+1):10.1f}, Offload Time : {sub_graphs_time/(i+1):10.1f} , DDR RW MBs : 0, Output File : {output_file_name} \n \n ' #{classes} \n \n'
    print(log) 
    if ncpus > 1:
        sem.release()


#models = models_configs.keys()

models = ['cl-ort-resnet18-v1', 'ss-ort-deeplabv3lite_mobilenetv2', 'od-ort-ssd-lite_mobilenetv2_fpn']
if platform.machine() != 'aarch64':
    download_models()

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