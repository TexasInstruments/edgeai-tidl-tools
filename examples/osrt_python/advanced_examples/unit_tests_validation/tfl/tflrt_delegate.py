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

required_options = {
"tidl_tools_path":tidl_tools_path,
"artifacts_folder":artifacts_folder,
}
parser = argparse.ArgumentParser()
parser.add_argument('-c','--compile', action='store_true', help='Run in Model compilation mode')
parser.add_argument('-d','--disable_offload', action='store_true',  help='Disable offload to TIDL')

args = parser.parse_args()
os.environ["TIDL_RT_PERFSTATS"] = "1"

sem = multiprocessing.Semaphore(0)
if platform.machine() == 'aarch64':
    ncpus = 1
else:
    ncpus = os.cpu_count()
#ncpus = 1
idx = 0
nthreads = 0
run_count = 0
DEVICE = os.environ["DEVICE"]

if(DEVICE == "am62"):
    args.disable_offload = True
    args.compile = False

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



def infer_image(interpreter):
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  outputs = {}

  for i in range(len(input_details)):
    np.random.seed(0)
    input_data = np.random.rand(*input_details[i]['shape']).astype(input_details[i]['dtype'])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[i]['index'], input_data)
 
  start_time = time.time()
  #interpreter invoke call
  interpreter.invoke()
  stop_time = time.time()

  if(DEVICE != "am62"):
    copy_time, proc_time, sub_graphs_proc_time, ddr_write, ddr_read  = get_benchmark_output(interpreter)
    proc_time = proc_time - copy_time
  else:
    copy_time = proc_time = sub_graphs_proc_time = ddr_write = ddr_read  = 0
    proc_time = (stop_time - start_time) * 1000000000
  for output_detail in output_details:
    outputs[output_detail['index']] = interpreter.get_tensor(output_detail['index'])

  ## can return output name and corresponding tensor as dictionary, and save using output name to identify which output
  return outputs, proc_time, sub_graphs_proc_time, ddr_write, ddr_read

def run_model(model, mIdx):
    print("\nRunning_Model : ", model)
    config = models_configs[model]

    #set delegate options
    delegate_options = {}
    delegate_options.update(required_options)
    delegate_options.update(optional_options)
    # stripping off the ss-tfl- from model namne
    delegate_options['artifacts_folder'] = delegate_options['artifacts_folder'] + '/' + model + '/'

    if config['model_type'] == 'od':
        delegate_options['object_detection:meta_layers_names_list'] = config['meta_layers_names_list'] if ('meta_layers_names_list' in config) else ''
        delegate_options['object_detection:meta_arch_type'] = config['meta_arch_type'] if ('meta_arch_type' in config) else -1

    # delete the contents of this folder
    if (args.compile or args.disable_offload):
        os.makedirs(delegate_options['artifacts_folder'], exist_ok=True)
        for root, dirs, files in os.walk(delegate_options['artifacts_folder'], topdown=False):
            [os.remove(os.path.join(root, f)) for f in files]
            [os.rmdir(os.path.join(root, d)) for d in dirs]

    ############   set interpreter  ################################
    if args.disable_offload : 
        interpreter = tflite.Interpreter(model_path=config['model_path'], num_threads=2, experimental_preserve_all_tensors=True)
    elif args.compile:
        interpreter = tflite.Interpreter(model_path=config['model_path'], \
                        experimental_delegates=[tflite.load_delegate(os.path.join(tidl_tools_path, 'tidl_model_import_tflite.so'), delegate_options)])
    else:
        interpreter = tflite.Interpreter(model_path=config['model_path'], \
                        experimental_delegates=[tflite.load_delegate('libtidl_tfl_delegate.so', delegate_options)])
    ################################################################
    
    # run interpreter
    outputs, proc_time, sub_graph_time, ddr_write, ddr_read  = infer_image(interpreter)
    total_proc_time = total_proc_time + proc_time if ('total_proc_time' in locals()) else proc_time
    sub_graphs_time = sub_graphs_time + sub_graph_time if ('sub_graphs_time' in locals()) else sub_graph_time
    total_ddr_write = total_ddr_write + ddr_write if ('total_ddr_write' in locals()) else ddr_write
    total_ddr_read  = total_ddr_read + ddr_read if ('total_ddr_read' in locals()) else ddr_read
    total_proc_time = total_proc_time/1000000
    sub_graphs_time = sub_graphs_time/1000000
    
    #### Example code to save intermediate layer outputs for comparison with TIDL outputs
    ##
    # for t in interpreter.get_tensor_details():
    #     print(interpreter.get_tensor(t['index']))
    #
    # ip = interpreter.get_tensor(20)
    # ip = np.transpose(ip,[0,3,1,2]) ## TIDL output in NCHW format and Tflite traces in NHWC format --  if number of input dimensions = 4, else transpose not needed
    # ip.tofile("PATH/edgeai-tidl-tools/examples/osrt_python/advanced_examples/unit_tests_validation/traces_ref/OUT20.bin")
    
    # output post processing
    if(args.compile == False):  # post processing enabled only for inference
        for output_index, output_tensor in outputs.items():
            out = np.array(output_tensor, dtype = np.float32)
            if(args.disable_offload):
                out.tofile('../outputs/output_ref/tflite/' + os.path.basename(config['model_path']) + '_' + str(output_index) + '.bin') # save as model name + output name
            else:
                out.tofile('../outputs/output_test/tflite/' + os.path.basename(config['model_path']) + '_' + str(output_index) + '.bin')
    print('Completed model - ', os.path.basename(config['model_path']))
    if ncpus > 1:
        sem.release()

models = ['add_const']

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

