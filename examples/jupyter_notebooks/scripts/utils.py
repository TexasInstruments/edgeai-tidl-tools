import ipywidgets as widgets
import itertools
import matplotlib.patches as mpatches
import re
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
"""
jai_benchmark from edgeai-benchmark for getting preprocessing configurations
for all supported prebuilt models 
"""
from jai_benchmark.config_settings import ConfigSettings
from jai_benchmark.utils import get_name_key_pair_list
from configs import select_configs
from jai_benchmark.sessions.tflitert_session import TFLiteRTSession
from jai_benchmark.sessions.tvmdlr_session import TVMDLRSession
from jai_benchmark.sessions.onnxrt_session import ONNXRTSession
from jai_benchmark.utils.artifacts_id_to_model_name import model_id_artifacts_pair

import os
import sys
from pathlib import Path
import platform

if platform.machine() != 'aarch64':
    import requests
    import onnx


models = {
    '../../models/public/onnx/resnet18_opset9.onnx': {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/classification/imagenet1k/torchvision/resnet18_opset9.onnx', 'type': 'onnx'},
    '../../models/public/tflite/mobilenet_v1_1.0_224.tflite': {'model_url': 'https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_224/1/default/1?lite-format=tflite', 'type': 'tflite'},

}

def download_model(mpath):
    if(not os.path.isfile(mpath)):
        # Check whether the specified path exists or not
        isExist = os.path.exists(os.path.dirname(mpath))
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(os.path.dirname(mpath))
        if mpath in models:
            model_info = models[mpath]
            print("Downloading  ", mpath)
            url = model_info['model_url']
            r = requests.get(url, allow_redirects=True)
            open(mpath, 'wb').write(r.content)
            #run shape inference
            if model_info['type'] is 'onnx':
                print("Running shape inference for ", mpath)
                onnx.shape_inference.infer_shapes_path(mpath, mpath)
        else : 
            print(f'Model infor for {mpath} Not found')


image_id_name_pairs = {
    'dog' : 'sample-images/dog.jpg',
    'cat' : 'sample-images/cat.jpg',
    'rat' : 'sample-images/rat.jpg',
}

'''
model_id_artifacts_pair = {
    # TFLite CL
    'vcls-10-010-0-tflitert': 'TFL-CL-000-mobileNetV1',
    'vcls-10-401-0-tflitert': 'TFL-CL-001-mobileNetV2',
    'vcls-10-403-0-tflitert': 'TFL-CL-002-SqueezeNet',
    'vcls-10-405-8-tflitert': 'TFL-CL-003-InceptionNetV1',
    'vcls-10-406-0-tflitert': 'TFL-CL-004-InceptionNetV3',
    'vcls-10-409-0-tflitert': 'TFL-CL-005-resNet50V1',
    'vcls-10-410-0-tflitert': 'TFL-CL-006-resNet50V2',
    'vcls-10-407-0-tflitert': 'TFL-CL-007-mnasNet',
    'vcls-10-011-0-tflitert': 'TFL-CL-008-mobileNet-edgeTPU',
    'vcls-10-440-0-tflitert': 'TFL-CL-009-efficientNet-edgeTPU-s',
    'vcls-10-441-0-tflitert': 'TFL-CL-010-efficientNet-edgeTPU-m',
    'vcls-10-430-0-tflitert': 'TFL-CL-013-efficientNet-lite0',
    'vcls-10-434-0-tflitert': 'TFL-CL-014-efficientNet-lite4',
    'vcls-10-404-0-tflitert': 'TFL-CL-015-denseNet',
    'vcls-10-012-0-tflitert': 'TFL-CL-016-resNet50V1p5',
    'vcls-10-431-0-tflitert': 'TFL-CL-017-efficient-net-lite1',
    'vcls-10-432-0-tflitert': 'TFL-CL-018-efficient-net-lite2',
    'vcls-10-442-0-tflitert': 'TFL-CL-019-efficient-edgetpu-L',
    'vcls-10-402-0-tflitert': 'TFL-CL-020-mobileNetV2-1p4',
    'vcls-10-400-0-tflitert': 'TFL-CL-021-mobileNetv1', #added later
    'vcls-10-400-8-tflitert': 'TFL-CL-022-mobileNetv1-qat', #added later
    'vcls-10-401-8-tflitert': 'TFL-CL-023-mobileNetV2-qat', #added later
    'vcls-10-408-0-tflitert': 'TFL-CL-024-mnasnet-tflite', #added later
    'vcls-10-450-0-tflitert': 'TFL-CL-025-xceptionNet-tflite', # this is replaced with tflite model now

    # TFLite OD
    'vdet-12-010-0-tflitert': 'TFL-OD-200-ssd-mobV1-coco-300x300',
    'vdet-12-011-0-tflitert': 'TFL-OD-201-ssd-mobV2-coco-300x300', 
    'vdet-12-400-0-tflitert': 'TFL-OD-202-ssdLite-mobDet-DSP-coco-320x320',
    'vdet-12-401-0-tflitert': 'TFL-OD-203-ssdLite-mobDet-EdgeTPU-coco-320x320',
    'vdet-12-404-0-tflitert': 'TFL-OD-204-ssd-mobV1-FPN-coco-640x640',
    'vdet-12-403-0-tflitert': 'TFL-OD-205-ssd-mobV2-mnas-fpn-coco-320x320',
    'vdet-12-402-0-tflitert': 'TFL-OD-206-ssd-mobV2-coco-mlperf-300x300', # added later

    # TFLite SS
    'vseg-17-010-0-tflitert': 'TFL-SS-250-deeplab-mobV2-ade20k-512x512',
    'vseg-17-400-0-tflitert': 'TFL-SS-254-deeplabv3-mobv2-ade20k-512x512',
    'vseg-16-400-0-tflitert': 'TFL-SS-255-to-be-named', #not part of excel so no info yet
    'vseg-18-400-0-tflitert': 'TFL-SS-256-to-be-named', #not part of excel so no info yet
    'vseg-18-401-0-tflitert': 'TFL-SS-257-to-be-named', #not part of excel so no info yet

    # TVM- CL
    'vcls-10-020-0-tvmdlr': 'TVM-CL-300-resNet18V2',
    'vcls-10-450-0-tvmdlr': 'TVM-CL-302-xceptionNet-mxnet',
    'vcls-10-408-0-tvmdlr': 'TVM-CL-304-mnasnet-tflite',
    'vcls-10-100-0-tvmdlr': 'TVM-CL-306-mobileNetV1',
    'vcls-10-101-0-tvmdlr': 'TVM-CL-307-mobileNetV2',
    'vcls-10-101-8-tvmdlr': 'TVM-CL-338-mobileNetV2-qat',
    'vcls-10-301-0-tvmdlr': 'TVM-CL-308-shufflentV2',
    'vcls-10-302-0-tvmdlr': 'TVM-CL-309-mobileNetV2-tv',
    'vcls-10-302-8-tvmdlr': 'TVM-CL-339-mobileNetV2-tv-qat',
    'vcls-10-304-0-tvmdlr': 'TVM-CL-310-resnet18',
    'vcls-10-305-0-tvmdlr': 'TVM-CL-311-resnet50',
    'vcls-10-031-0-tvmdlr': 'TVM-CL-312-regnetX-400mf',
    'vcls-10-032-0-tvmdlr': 'TVM-CL-313-regnetX-800mf',
    'vcls-10-033-0-tvmdlr': 'TVM-CL-314-regnetX-1.6gf',
    'vcls-10-030-0-tvmdlr': 'TVM-CL-336-regnetx-200mf',
    'vcls-10-306-0-tvmdlr': 'TVM-CL-337-vgg16',
    # TVM - OD
    'vdet-12-012-0-tvmdlr': 'TVM-OD-500-ssd1200-resnet34-1200x1200',
    'vdet-12-020-0-tvmdlr': 'TVM-OD-501-yolov3-416x416',
    'vdet-12-060-0-tvmdlr': 'TVM-OD-502-yolov3-mobv1-gluon-mxnet', #added later
    'vdet-12-061-0-tvmdlr': 'TVM-OD-503-ssd-mobv1-gluon-mxnet', #added later
    # TVM - SS
    'vseg-16-100-0-tvmdlr': 'TVM-SS-550-deeplabv3lite-mobv2-cs-qat-768x384',
    'vseg-16-102-0-tvmdlr': 'TVM-SS-551-unetlite-aspp-mobv2-tv-cs-qat-768x384',
    'vseg-16-101-0-tvmdlr': 'TVM-SS-552-fpnlite-aspp-mobv2-cs-qat-768x384',
    'vseg-16-103-0-tvmdlr': 'TVM-SS-553-fpnlite-aspp-regnetx800mf-cs-768x384',
    'vseg-16-104-0-tvmdlr': 'TVM-SS-554-fpnlite-aspp-regnetx1.6gf-cs-1024x512',
    'vseg-16-105-0-tvmdlr': 'TVM-SS-555-fpnlite-aspp-regnetx3.2gf-cs-1536x768',
    'vseg-16-300-0-tvmdlr': 'TVM-SS-556-deeplabv3-res50-1040x520',
    'vseg-16-301-0-tvmdlr': 'TVM-SS-557-fcn-res50-1040x520',
}
'''

image_Ids = [key for key in image_id_name_pairs]


selected_image_id = widgets.Dropdown(
    options=image_Ids,
    value=image_Ids[0],
    description='Select Image:',
    disabled=False,
)

model_id_name_pairs = {
    'MZ_00_mobilenet_v1' : 'MZ_00_mobilenet_v1_2019-09-06_17-15-44_opset9.onnx',
    'MZ_01_mobilenet_v2_qat' : 'MZ_01_mobilenet_v2_qat_2020-12-13_16-53-07_71.73.onnx',
    'MZ_02_shufflenetv2_x1p0' : 'MZ_02_shufflenetv2_x1p0_opset9_onnx',
    'MZ_03_mobilenetv2_tv' : 'MZ_03_mobilenetv2_tv_x1_opset9.onnx',
    'MZ_04_resnet18' : 'MZ_04_resnet18_opset9.onnx',
    'MZ_05_resnet50' : 'MZ_05_resnet50_opset9.onnx',
    'MZ_06_RegNetX-400MF' : 'MZ_06_RegNetX-400MF_dds_8gpu_opset9.onnx',
    'MZ_07_RegNetX-800MF' : 'MZ_07_RegNetX-800MF_dds_8gpu_opset9.onnx',
    'MZ_08_RegNetX-1.6GF' : 'MZ_08_RegNetX-1.6GF_dds_8gpu_opset9.onnx',
    'MZ_09_mobilenet_v2_1p4_qat' : 'MZ_09_mobilenet_v2_1p4_qat_75.22.onnx',
}

model_ids = [key for key in model_id_name_pairs]

selected_model_id = widgets.Dropdown(
    options=model_ids,
    value=model_ids[1],
    description='Select Model:',
    disabled=False,
)


Image_source = widgets.RadioButtons(
    options=['Local', 'Web'],
    value='Local',
    # rows=10,
    description='Image Soure:',
    disabled=False
)

image_ulr = widgets.Text(
    value='https://github.com/dmlc/mxnet.js/blob/master/data/cat.png',
    #placeholder='NA',
    description='Enter url :',
    disabled=False
)



'''
Get statistics  from the model and Visualize

During the execution ofthe model several benchmarking data like timestamps at different checkpoints, DDR bandwidth etc are collected and stored. `get_TI_benchmark_data()` can be used to collect the statistics. This function returns a dictionary of `annotations` and the corresponding markers. You can use `print(stats)` in the cell below to see the format of the returned data.

```
ts:run_start                : 1234,
ts:run_end                  : 5678,
ts:subgraph_0_copy_in_start : 9012,
...
ddr:read_start              : 12345678,
ddr:write_start             : 90123456,
...
```

The timestamps are prefixed as `ts:` and the DDR counters are prefixed as `ddr:`.

The next cell shows an example script to use the timestamps and display a descriptive graph. The script isolates the subgraph statistics into `copy_in`, `proc` and `copy_out` durations, total runtime, and the durations of the intermittent `TVM` operations, and displays a bar graph.
'''
def plot_TI_performance_data(stats, axis=None):
    do_show = False
    
    if axis is None:
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots()
        do_show = True

    # ---- processing the data -----
    records = []

    # extract the timestamps and normalize to run_start
    perfstats = {k.replace('ts:', '') : v for k, v in stats.items() if k.startswith('ts:')}
    d = {k : (v - perfstats['run_start']) / 1000000 for k, v in perfstats.items()}

    # get number of subgraphs and sort in order of execution
    subn = set([elem.replace('_copy_in_start', '') for elem in d.keys() if elem.startswith('subgraph_') and elem.endswith('_copy_in_start')])
    subn = sorted(list(subn), key=lambda x : d[f'{x}_copy_in_start'])

    # populate subgraph records
    for pre, node in [(i, idx) for idx, i in enumerate(subn)]:
        records.append(
            ([(d[pre + el + '_start'], d[pre + el + '_end'] - d[pre + el + '_start']) for el in ['_copy_in', '_proc', '_copy_out']],
             f'c7x_mma_op_{node}',
             ('tab:orange', 'tab:green', 'tab:purple')))

    # populate records for ARM operations
    records.append(
        ([(d[f], d[l] - d[f]) for f, l in zip(['run_start'] + [f'{i}_copy_out_end' for i in subn], [f'{i}_copy_in_start' for i in subn] + ['run_end'])],
         'cpu_op',
         ('tab:blue')))

    # populate total runtime record
    #records.append(([(0, d['run_end'] - d['run_start'])], "total", ('tab:red')))

        
    # Set legend
    legends = [
        #mpatches.Patch(color='red', label='total'),
        mpatches.Patch(color='blue', label='cpu_operation'),
        mpatches.Patch(color='orange', label='in_tensor copy'),
        mpatches.Patch(color='green', label='c7x_mma_operation'),
        mpatches.Patch(color='purple', label='out_tensor copy')
    ]
    axis.legend(handles=legends, ncol = len(legends))

    # Set Y markers
    ticks = [i + (0.1 * i) + 0.5 for i in range(len(records))]
    names = [x[1] for x in records]
    axis.set_yticks(ticks)
    axis.set_yticklabels(names)

    # plot colored bars 
    for i, r in enumerate(records):
        axis.broken_barh(r[0], (i + i * 0.1, 1), facecolors=r[2])

    axis.set_xlabel('ms')
    axis.set_ylabel('Processing Task location')

    # text annotation for total runtime
    #axis.text(d['run_end'] / 2, ticks[-1], '%.02f milliseconds' % (d['run_end'] - d['run_start']),
    #     {'color': 'white', 'fontsize': 24, 'ha': 'center', 'va': 'center'})

    if do_show:
        plt.show()

def plot_TI_DDRBW_data(stats, axis=None):
    do_show = False
    
    if axis is None:
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots()
        do_show = True

    read_total = stats['ddr:read_end'] - stats['ddr:read_start']
    write_total = stats['ddr:write_end'] - stats['ddr:write_start']
    
    axis.scatter(0.5, 0.5, s=read_total//100)
    axis.annotate(f'{read_total/1000000:05.2f}', (0.5, 0.5), c='white', fontsize=18, va='center', ha='center')
    axis.scatter(1.5, 0.5, s=write_total//100)
    axis.annotate(f'{write_total/1000000:05.2f}', (1.5, 0.5), c='white', fontsize=18, va='center', ha='center')
    
    axis.set_yticks([])
    
    axis.set_xlim([0, 2])
    axis.set_xticks([0.5, 1.5])
    axis.set_xticklabels(['DDR bytes read', 'DDR bytes written'])
        
    if do_show:
        plt.show()
 
'''
Utility function to get class names from imagenet class IDs

We define a utility function which we will be using multiple times in this notebook. This function `imagenet_class_to_name` takes a class ID as input, and builds a dictionary using `/datasets/imagenet/data/val_text.txt` which is used for looking up the class ID. ***ImageNet*** may define a list of names for a given class, which is broken down into a list of individual strings for storing into the dictionary
'''
def imagenet_class_to_name(cls):
    # build imagenet class ID to name list dictionary
    imagenet_dict = {}
    with open('sample-images/imagenet-labels.txt') as f:
        for l in f.readlines():
            c = int(l.split(':')[0])
            p = [name.strip() for name in l.split(':')[1].strip()[1:][:-2].split(',')]
            imagenet_dict[c] = p

    # return name list using dictionary lookup
    return imagenet_dict[cls]

"""
def imagenet_class_to_name(cls):  
    # build imagenet class ID to name list dictionary
    imagenet_dict = {}
    with open('sample-images/labels.txt') as f:
        for c , l in enumerate(f.readlines()):
            p = [name.strip() for name in l.split(',')]
            imagenet_dict[c] = p
    # return name list using dictionary lookup
    return imagenet_dict[cls] 
"""
    
def get_benchmark_output(benchmark_dict):
    proc_time = copy_time = 0
    cp_in_time = cp_out_time = 0
    subgraphIds = []
    for stat in benchmark_dict.keys():
        if 'proc_start' in stat:
            subgraphIds.append(stat.replace('ts:subgraph_', '').replace('_proc_start', ''))
    for i in range(len(subgraphIds)):
        proc_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_start']
        cp_in_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_start']
        cp_out_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_start']
    copy_time = cp_in_time + cp_out_time
    copy_time = copy_time if len(subgraphIds) == 1 else 0
    total_time   = benchmark_dict['ts:run_end'] - benchmark_dict['ts:run_start']
    write_total  = benchmark_dict['ddr:read_end'] - benchmark_dict['ddr:read_start']
    read_total   = benchmark_dict['ddr:write_end'] - benchmark_dict['ddr:write_start']

    total_time = total_time - copy_time

    return total_time/1000000, proc_time/1000000, read_total/1000000, write_total/1000000

colors_list = [
( 255, 	 0,	  0 ), ( 0	 , 255,    0 ), ( 0	,   0,	 255 ), ( 255, 255,	    0  ), ( 0	 , 255,  255  ), ( 255,   0,	 255  ),
( 255, 	 64,  0 ), ( 64	 , 255,    0 ), ( 64,   0,	 255 ), ( 255, 255,	   64  ), ( 64	 , 255,  255  ), ( 255,   64,	 255  ),
( 196, 	128,  0 ), ( 128 , 196,    0 ), ( 128,  0,	 196 ), ( 196, 196,	  128  ), ( 128	 , 196,  196  ), ( 196,   128,	 196  ),
( 64, 	128,  0 ), ( 128 , 64,     0 ), ( 128,  0,	 64  ), ( 196,   0,    0  ), ( 196	 ,  64,   64  ), ( 64,    196,	  64  ),
( 64,   255, 64 ), ( 64	 , 64,   255 ),( 255, 64,	 64  ), (128,  255,   128  ), ( 128	, 128,    255  ),( 255,   128,	 128  ),
( 196,  64, 196 ), ( 196, 196,    64 ),( 64,  196,	196  ), (196,  255,   196  ), ( 196	, 196,    255  ),( 196,   196,	 128  )]


def det_box_overlay(outputs, org_image_rgb, thr, postprocess, org_size, size):
    source_img = org_image_rgb.convert("RGBA")
    draw = ImageDraw.Draw(source_img)

    info_dict = {}
    info_dict['data'] = np.asarray(draw)
    info_dict['data_shape'] = (org_size[1], org_size[0], 3)
    info_dict['resize_shape'] = (size[1], size[0], 3)
    info_dict['resize_border'] = (0, 0, 0, 0)
    outputs, info_dict = postprocess(outputs, info_dict)

    for i in range(int(outputs.shape[0])):
        if(outputs[i][5] > thr) :
            xmin = outputs[i][0]
            ymin = outputs[i][1]
            xmax = outputs[i][2]
            ymax = outputs[i][3]
            draw.rectangle(((int(xmin), int(ymin)), (int(xmax), int(ymax))), outline = colors_list[int(outputs[i][4])%len(colors_list)], width=int(((source_img.width/300)) +1))
    source_img = source_img.convert("RGB")
    return(source_img)

def det_box_overlay_onnxrt(outputs, org_image_rgb, thr):
    source_img = org_image_rgb.convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    num_boxes = int(outputs[0].shape[1])
    for i in range(num_boxes):
        if(outputs[2][0][i] > thr) :
            xmin = outputs[0][0][i][0]
            ymin = outputs[0][0][i][1]
            xmax = outputs[0][0][i][2]
            ymax = outputs[0][0][i][3]
            draw.rectangle(((int(xmin*source_img.width), int(ymin*source_img.height)), (int(xmax*source_img.width), int(ymax*source_img.height))), outline = colors_list[int(outputs[1][0][i])%len(colors_list)], width=4)
    
    source_img = source_img.convert("RGB")
    return(source_img)

def mask_transform(inp):
    colors = np.asarray(colors_list)
    inp = np.squeeze(inp)
    colorimg = np.zeros((inp.shape[0], inp.shape[1], 3), dtype=np.float32)
    height, width = inp.shape
    inp = np.rint(inp)
    inp = inp.astype(np.uint8)
    for y in range(height):
        for x in range(width):
            if(inp[y][x] < 22):
                colorimg[y][x] = colors[inp[y][x]]
    inp = colorimg.astype(np.uint8)
    return inp

def RGB2YUV( rgb ):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
    yuv = np.dot(rgb,m)
    yuv[:,:, 1:] += 128.0
    rgb = np.clip(yuv, 0.0, 255.0)
    return yuv

def YUV2RGB( yuv ):
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 2.0320025777816772],
                 [ 1.14019975662231445, -0.5811380310058594 , 0.00001542569043522235] ])
    yuv[:,:, 1:] -= 128.0
    rgb = np.dot(yuv,m)
    rgb = np.clip(rgb, 0.0, 255.0)

    return rgb
def seg_mask_overlay(output_data, org_image_rgb, layout):
  output_data = np.squeeze(output_data)
  if (output_data.ndim > 2) :
    if layout == 'NCHW':
        output_data = output_data.argmax(axis=0)
    else : 
        output_data = output_data.argmax(axis=1)

  output_data = np.squeeze(output_data)
  mask_image_rgb  = mask_transform(output_data) 
  org_image  = RGB2YUV(org_image_rgb.resize(output_data.shape))
  mask_image = RGB2YUV(mask_image_rgb)
  
  org_image[:,:, 1] = mask_image[:,:, 1]
  org_image[:,:, 2] = mask_image[:,:, 2]
  blend_image = YUV2RGB(org_image)
  blend_image = blend_image.astype(np.uint8)
  blend_image = Image.fromarray(blend_image).convert('RGB')
  
  return(blend_image)

'''
def get_ky_name_pair(eval_list, runtime_type):
    eval_list_pair_key       = [k+"_"+runtime_type for k in eval_list] 
    merged_list = [(model_id_artifacts_pair[k],k.split('_')[0]) for k in eval_list_pair_key if k in model_id_artifacts_pair.keys()]
    return merged_list
'''


def get_selected_artifacts_id():       
    model_id = None
    result_artifacts_key = None
    try:
        fname = '/opt/model-selection-tool/last_model_name.txt'
        fp = open(fname, 'r')
        model_id = fp.readline()
        fp.close()
        #remove file so that user can change model from drop-down.
        os.remove(fname)
    except FileNotFoundError:
        #print("file {} does not exist".format(fname))
        return result_artifacts_key

    print("model id from model selection tool", model_id)
    for artifacts_key in model_id_artifacts_pair:
        if model_id_artifacts_pair[artifacts_key] in model_id:
            # removing RT type vcls-10-402-0_tflitert -> vcls-10-402-0
            result_artifacts_key = artifacts_key.split("_")[0]
            #print("new Found key for model id from model selection tool ", result_artifacts_key)
            break
    return result_artifacts_key   

def get_eval_configs(task_type, runtime_type, num_quant_bits, last_artifacts_id=None, high_resolution = False):
    
    if runtime_type == 'tflitert':
        session_type = TFLiteRTSession
        session_type_dict = {'onnx': 'onnxrt', 'tflite': 'tflitert', 'mxnet': 'tvmdlr'}

    if runtime_type == 'tvmdlr':
        session_type = TVMDLRSession
        session_type_dict = {'onnx': 'tvmdlr', 'tflite': 'tflitert', 'mxnet': 'tvmdlr'}

    if runtime_type == 'onnxrt':
        session_type = ONNXRTSession
        session_type_dict = {'onnx': 'onnxrt', 'tflite': 'tflitert', 'mxnet': 'tvmdlr'}

    prebuilts_dir = 'prebuilt-models'

    settings_dict = {
        'dataset_loading' : False,
        'tidl_tensor_bits' : num_quant_bits,
        'task_selection' : task_type,
        'session_type_dict' : session_type_dict,
        'run_import' : False
    }

    # This option changes the input sizes of these selected models to high resolution
    # This is only used for performance measurement. For more details, refer to:
    # https://github.com/TexasInstruments/edgeai-benchmark/blob/master/scripts/benchmark_resolution.py
    if high_resolution:
        # only these models are supported in high_resolution setting
        model_selection_high_resolution = [
                           'edgeai-tv/mobilenet_v1_20190906.onnx',
                           'edgeai-tv/mobilenet_v2_20191224.onnx',
                           'edgeai-tv/mobilenet_v2_1p4_qat-p2_20210112.onnx',
                           'torchvision/resnet18.onnx',
                           'torchvision/resnet50.onnx',
                           'fbr-pycls/regnetx-400mf.onnx',
                           'fbr-pycls/regnetx-800mf.onnx',
                           'fbr-pycls/regnetx-1.6gf.onnx'
                          ]
        # these artifacts are meant for only performance measurement - just do a quick import with simple calibration
        # also set the high_resolution_optimization flag for improved performance at high resolution
        runtime_options = {'accuracy_level': 0, 'advanced_options:high_resolution_optimization': 1}
        # the transformations that needs to be applied to the model itself. Note: this is different from pre-processing transforms
        high_resolution_input_sizes = [512, 1024]
        model_transformation_dict = {'input_sizes': high_resolution_input_sizes}
        settings_update_high_resolution = dict(model_selection=model_selection_high_resolution,
            num_frames=100, calibration_iterations=1, runtime_options=runtime_options,
            model_transformation_dict=model_transformation_dict)
        settings_dict.update(settings_update_high_resolution)
    #
    settings = ConfigSettings(settings_dict)
    prebuilt_configs = select_configs(settings,os.path.join(prebuilts_dir, f'{num_quant_bits}bits'), runtime_type)
    merged_list = get_name_key_pair_list(prebuilt_configs.keys(), runtime_type)
    #print("merged_list: ", prebuilt_configs.keys())

    model_selection_artifacts_key = get_selected_artifacts_id()
    if not model_selection_artifacts_key is None:
        last_artifacts_id = model_selection_artifacts_key
    elif last_artifacts_id is None:
        last_artifacts_id = merged_list[0][1]
    #print("last_artifacts_id: ", last_artifacts_id)
    selected_model_id = widgets.Dropdown(
    options=merged_list,
    value=last_artifacts_id,
    description='Select Model:',
    disabled=False,)
    last_artifacts_name = selected_model_id.value
    return prebuilt_configs, selected_model_id

def get_preproc_props(pipeline_config):
    size = pipeline_config['preprocess'].get_param('crop')
    mean = list(pipeline_config['preprocess'].get_param('mean'))
    scale = list(pipeline_config['preprocess'].get_param('scale')) 
    layout = pipeline_config['preprocess'].get_param('data_layout')
    reverse_channels = pipeline_config['preprocess'].get_param('reverse_channels')
    
    if type(size) is tuple:
        size = list(size)

    return size, mean, scale, layout, reverse_channels


def task_type_to_dataset_list(task_type):
    assert task_type == 'classification'
    return ['imagenet']

"""
class loggerWritter():
    # Redirect c- stdout and stderr to a couple of files.
"""

log_dir = Path('./')

class loggerWritter():
    def __init__(self, logname):
        self.logname = logname
        sys.stdout.flush()
        sys.stderr.flush()

        if self.logname == None:
            self.logpath_out = os.devnull
            self.logpath_err = os.devnull
        else:
            self.logpath_out = log_dir / (logname + "_out.log")
            self.logpath_err = log_dir / (logname + "_err.log")

        self.logfile_out = os.open(self.logpath_out, os.O_WRONLY|os.O_TRUNC|os.O_CREAT)
        self.logfile_err = os.open(self.logpath_err, os.O_WRONLY|os.O_TRUNC|os.O_CREAT)
    def __enter__(self):
        self.orig_stdout = sys.stdout # save original stdout
        self.orig_stderr = sys.stderr # save original stderr

        self.new_stdout = os.dup(1)
        self.new_stderr = os.dup(2)

        os.dup2(self.logfile_out, 1)
        os.dup2(self.logfile_err, 2)

        sys.stdout = os.fdopen(self.new_stdout, 'w')
        sys.stderr = os.fdopen(self.new_stderr, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        sys.stderr.flush()

        sys.stdout = self.orig_stdout # restore original stdout
        sys.stderr = self.orig_stderr # restore original stderr
        os.close(self.logfile_out)
        os.close(self.logfile_err)
