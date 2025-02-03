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
#import onnx
# directory reach
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
# setting path
sys.path.append(parent)
from common_utils import *

model_optimizer_found = False
if platform.machine() != "aarch64":
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
parser.add_argument('-u','--unit_test', action='store_true', help='Run unit test case')
parser.add_argument('-n','--ncpus', type=int, default=None, help='Number of threads to spawn')
parser.add_argument(
    "-o",
    "--graph_optimize",
    action="store_true",
    help="Run ONNX model optimization thourgh onnx-graph-surgeon-tidl",
)

args = parser.parse_args()
os.environ["TIDL_RT_PERFSTATS"] = "1"

so = rt.SessionOptions()
# so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
so.log_severity_level=3

print("Available execution providers : ", rt.get_available_providers())

calib_images = ['../../../../../test_data/airshow.jpg',
                '../../../../../test_data/ADE_val_00001801.jpg']
class_test_images = ['../../../../../test_data/airshow.jpg']
od_test_images    = ['../../../../../test_data/ADE_val_00001801.jpg']
seg_test_images   = ['../../../../../test_data/ADE_val_00001801.jpg']


sem = multiprocessing.Semaphore(0)
if platform.machine() == 'aarch64':
    ncpus = 1
else:
    if args.ncpus and args.ncpus > 0 and args.ncpus < os.cpu_count():
        ncpus = args.ncpus
    else:
        ncpus = os.cpu_count()

idx = 0
nthreads = 0
run_count = 0

SOC = os.environ["SOC"]

if(SOC == "am62"):
    args.disable_offload = True
    args.compile = False

if args.compile == True and tidl_tools_path == None:
    print("TIDL_TOOLS_PATH is not set" )
    exit(-1)

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


def infer_image(sess):
  input_details = sess.get_inputs()
  output_details = sess.get_outputs()
  input_dict = {}
  output_dict = {}

  for i in range(len(input_details)):
    np.random.seed(0)
    if(input_details[i].type == 'tensor(float)'):
        input_data = np.random.randn(*input_details[i].shape).astype(np.float32)
    elif(input_details[i].type == 'tensor(int64)'):
        input_data = np.random.randn(*input_details[i].shape).astype(np.int64)
    elif(input_details[i].type == 'tensor(uint8)'):
        input_data = np.random.randn(*input_details[i].shape).astype(np.uint8)
    else:
        input_data = np.random.randn(*input_details[i].shape).astype(np.float32)
        
    input_dict[input_details[i].name] = input_data
  
  start_time = time.time()
  
  output = list(sess.run(None, input_dict))
  for i in range(len(output_details)):
    output_dict[output_details[i].name] = output[i]

  stop_time = time.time()
  infer_time = stop_time - start_time

  copy_time, sub_graphs_proc_time, totaltime = get_benchmark_output(sess)
  proc_time = totaltime - copy_time

  return output_dict, proc_time, sub_graphs_proc_time

def run_model(model, mIdx):
    print("\nRunning_Model : ", model, " \n")
    config = models_configs[model]

    # Run graph optimization
    if args.graph_optimize:
        if model_optimizer_found:
            if (args.compile or args.disable_offload) and (
                platform.machine() != "aarch64"
            ):
                copy_path = config["model_path"][:-5] + "_org.onnx"
                # Check if copy path exists and prompt for permission to overwrite
                if os.path.isfile(copy_path):
                    overwrite_permission = input(
                        f"\033[96mThe file {copy_path} exists, do you want to overwrite? [Y/n] \033[00m"
                    )
                    if overwrite_permission != "Y":
                        print("Aborting run...")
                        sys.exit(-1)
                    else:
                        print(
                            f"\033[93m[WARNING] File {copy_path} will be overwritten\033[00m"
                        )

                shutil.copy2(config["model_path"], copy_path)
                print(
                    f"\033[93mOptimization Enabled: Moving {config['model_path']} to {copy_path} before overwriting by optimization\033[00m"
                )
                optimize(
                    model=config["model_path"], out_model=config["model_path"]
                )
            else:
                print(
                    "Model optimization is only supported in compilation or disabled offload mode on x86 machines"
                )
        else:
            print("Model optimizer not found, -o flag has no effect")

    delegate_options = {}
    delegate_options.update(required_options)
    delegate_options.update(optional_options)
    # This is to handle options which are specific to any model
    if 'model_specific_options' in config.keys():
        delegate_options.update(config.get('model_specific_options'))

    # stripping off the ss-ort- from model namne
    delegate_options['artifacts_folder'] = delegate_options['artifacts_folder'] + '/' + model + '/artifacts' #+ 'tempDir/' 

    if config['task_type'] == 'detection':
        delegate_options['object_detection:meta_layers_names_list'] = config['extra_info']['meta_layers_names_list'] if ('meta_layers_names_list' in config) else ''
        delegate_options['object_detection:meta_arch_type'] = config['extra_info']['meta_arch_type'] if ('meta_arch_type' in config) else -1

    # delete the contents of this folder
    if args.compile or args.disable_offload:
        os.makedirs(delegate_options['artifacts_folder'], exist_ok=True)
        for root, dirs, files in os.walk(delegate_options['artifacts_folder'], topdown=False):
            [os.remove(os.path.join(root, f)) for f in files]
            [os.rmdir(os.path.join(root, d)) for d in dirs]

    if(args.compile == True):
        import onnx
        log = f'\nRunning shape inference on model {config["model_path"]} \n'
        print(log)
        onnx.shape_inference.infer_shapes_path(config['model_path'], config['model_path'])
    
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
    if platform.machine() == 'aarch64':
        num_frames = 100
    else:
        num_frames = 1
    for i in range(num_frames):
        output_dict, proc_time, sub_graph_time = infer_image(sess)
        total_proc_time = total_proc_time + proc_time if ('total_proc_time' in locals()) else proc_time
        sub_graphs_time = sub_graphs_time + sub_graph_time if ('sub_graphs_time' in locals()) else sub_graph_time
    
    total_proc_time = total_proc_time /1000000
    sub_graphs_time = sub_graphs_time/1000000

    # output post processing
    if(args.compile == False):  # post processing enabled only for inference
        for output_name, output_tensor in output_dict.items():
            out = np.array(output_tensor, dtype = np.float32)
            if(args.disable_offload):
                out.tofile('../outputs/output_ref/onnx/' + os.path.basename(config['model_path']) + '_' + output_name + '.bin')
            else:
                out.tofile('../outputs/output_test/onnx/' + os.path.basename(config['model_path']) + '_' + output_name + '.bin')
    print('Completed model - ', os.path.basename(config['model_path']))
    log = f'\n \nName : {model:50s}, Total time : {total_proc_time/(i+1):10.2f}, Offload Time : {sub_graphs_time/(i+1):10.2f} , DDR RW MBs : 0\n \n ' #{classes} \n \n'
    print(log)
    if ncpus > 1:
        sem.release()


models = ['add_eltwise']

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
