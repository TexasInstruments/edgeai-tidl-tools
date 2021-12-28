import tflite_runtime.interpreter as tflite
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
# directory reach
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
# setting path
sys.path.append(parent)
from common_utils import *

artifacts_folder = '../../../model-artifacts/tfl/'
output_images_folder = '../../../output_images/tfl-py/'

required_options = {
"tidl_tools_path":tidl_tools_path,
"artifacts_folder":artifacts_folder
}
parser = argparse.ArgumentParser()
parser.add_argument('-c','--compile', action='store_true', help='Run in Model compilation mode')
parser.add_argument('-d','--disable_offload', action='store_true',  help='Disable offload to TIDL')
args = parser.parse_args()
os.environ["TIDL_RT_PERFSTATS"] = "1"

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
            subgraphIds.append(int(re.sub("[^0-9]", "", stat)))
    for i in range(len(subgraphIds)):
        proc_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_start']
        cp_in_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_start']
        cp_out_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_start']
        copy_time += cp_in_time + cp_out_time
    
    write_total  = benchmark_dict['ddr:read_end'] - benchmark_dict['ddr:read_start']
    read_total = benchmark_dict['ddr:write_end'] - benchmark_dict['ddr:write_start']
    totaltime = benchmark_dict['ts:run_end'] -  benchmark_dict['ts:run_start']
  
    copy_time = copy_time if len(subgraphIds) == 1 else 0
    return copy_time, totaltime, proc_time, write_total/1000000, read_total/1000000



def infer_image(interpreter, image_file, config):
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  floating_model = input_details[0]['dtype'] == np.float32
  height = input_details[0]['shape'][1]
  width  = input_details[0]['shape'][2]
  new_height = height  #valid height for modified resolution for given network
  new_width = width  #valid width for modified resolution for given network
  img    = Image.open(image_file).convert('RGB').resize((new_width, new_height), PIL.Image.LANCZOS)
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = np.float32(input_data)
    for mean, scale, ch in zip(config['mean'], config['std'], range(input_data.shape[3])):
        input_data[:,:,:, ch] = ((input_data[:,:,:, ch]- mean) * scale)
  else:
    config['mean'] = [0, 0, 0]
    config['std']  = [1, 1, 1]

  interpreter.resize_tensor_input(input_details[0]['index'], [1, new_height, new_width, 3])
  interpreter.allocate_tensors()
  interpreter.set_tensor(input_details[0]['index'], input_data)
  
  start_time = time.time()
  #interpreter invoke call
  interpreter.invoke()
  stop_time = time.time()

  copy_time, proc_time, sub_graphs_proc_time, ddr_write, ddr_read  = get_benchmark_output(interpreter)
  proc_time = proc_time - copy_time

  outputs = [interpreter.get_tensor(output_detail['index']) for output_detail in output_details]
  return img, outputs, proc_time, sub_graphs_proc_time, ddr_write, ddr_read, new_height, new_width

def run_model(model, mIdx):
    print("\nRunning_Model : ", model)
    #set input images for demo
    if platform.machine() != 'aarch64':
        download_model(models_configs, model)
    config = models_configs[model]
 
    if config['model_type'] == 'classification':
        test_images = class_test_images
    elif config['model_type'] == 'od':
        test_images = od_test_images
    elif config['model_type'] == 'seg':
        test_images = seg_test_images

    #set delegate options
    delegate_options = {}
    delegate_options.update(required_options)
    delegate_options.update(optional_options)
    # stripping off the ss-tfl- from model namne
    model_name = model[7:]
    delegate_options['artifacts_folder'] = delegate_options['artifacts_folder'] + '/' + model_name + '/'

    if config['model_type'] == 'od':
        delegate_options['object_detection:meta_layers_names_list'] = config['meta_layers_names_list'] if ('meta_layers_names_list' in config) else ''
        delegate_options['object_detection:meta_arch_type'] = config['meta_arch_type'] if ('meta_arch_type' in config) else -1

    # delete the contents of this folder
    if (args.compile or args.disable_offload):
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
        interpreter = tflite.Interpreter(model_path=config['model_path'], num_threads=2)
    elif args.compile:
        interpreter = tflite.Interpreter(model_path=config['model_path'], \
                        experimental_delegates=[tflite.load_delegate(os.path.join(tidl_tools_path, 'tidl_model_import_tflite.so'), delegate_options)])
    else:
        interpreter = tflite.Interpreter(model_path=config['model_path'], \
                        experimental_delegates=[tflite.load_delegate('libtidl_tfl_delegate.so', delegate_options)])
    ################################################################
    
    # run interpreter
    for i in range(numFrames):
        img, output, proc_time, sub_graph_time, ddr_write, ddr_read, new_height, new_width  = infer_image(interpreter, input_image[i%len(input_image)], config)
        total_proc_time = total_proc_time + proc_time if ('total_proc_time' in locals()) else proc_time
        sub_graphs_time = sub_graphs_time + sub_graph_time if ('sub_graphs_time' in locals()) else sub_graph_time
        total_ddr_write = total_ddr_write + ddr_write if ('total_ddr_write' in locals()) else ddr_write
        total_ddr_read  = total_ddr_read + ddr_read if ('total_ddr_read' in locals()) else ddr_read
    
    total_proc_time = total_proc_time/1000000
    sub_graphs_time = sub_graphs_time/1000000
    output_file_name = "post_proc_out_"+model+'_'+os.path.basename(input_image[i%len(input_image)])

    # output post processing
    if(args.compile == False):  # post processing enabled only for inference
        if config['model_type'] == 'classification':
            classes, image = get_class_labels(output[0],img)
            print(classes)
        elif config['model_type'] == 'od':
            classes, image = det_box_overlay(output, img, config['od_type'])
        elif config['model_type'] == 'seg':
            classes, image = seg_mask_overlay(output[0], img)
        else:
            print("Not a valid model type")

        print("\nSaving image to ", output_images_folder)
        if not os.path.exists(output_images_folder):
            os.makedirs(output_images_folder)
        image.save(output_images_folder + output_file_name, "JPEG") 
    gen_param_yaml(delegate_options['artifacts_folder'], config, int(new_height), int(new_width))
    log = f'\n \nCompleted_Model : {mIdx+1:5d}, Name : {model:50s}, Total time : {total_proc_time/(i+1):10.2f}, Offload Time : {sub_graphs_time/(i+1):10.2f} , DDR RW MBs : {(total_ddr_write+total_ddr_read)/(i+1):10.2f}, Output File : {output_file_name}\n \n ' #{classes} \n \n'
    print(log) 
    if ncpus > 1:
        sem.release()

models = ['cl-tfl-mobilenet_v1_1.0_224', 'ss-tfl-deeplabv3_mnv2_ade20k_float', 'od-tfl-ssd_mobilenet_v2_300_float']
log = f'Running {len(models)} Models - {models}\n'
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

