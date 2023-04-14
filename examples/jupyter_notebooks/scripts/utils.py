import ipywidgets as widgets
import itertools
import matplotlib.patches as mpatches
import re
import numpy as np
import math
import cv2
import copy
from munkres import Munkres
from numpy.lib.stride_tricks import as_strided
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
"""
Jacinto_ai_benchmark for getting preprocessing configurations
for all supported prebuilt models
"""
from edgeai_benchmark.config_settings import ConfigSettings
from edgeai_benchmark.utils import get_name_key_pair_list
from configs import select_configs
from edgeai_benchmark.sessions.tflitert_session import TFLiteRTSession
from edgeai_benchmark.sessions.tvmdlr_session import TVMDLRSession
from edgeai_benchmark.sessions.onnxrt_session import ONNXRTSession
from edgeai_benchmark.utils.artifacts_id_to_model_name import model_id_artifacts_pair

import os
import sys
from pathlib import Path
import platform

_CLASS_COLOR_MAP = [
    (0, 0, 255) , # Person (blue).
    (255, 0, 0) ,  # Bear (red).
    (0, 255, 0) ,  # Tree (lime).
    (255, 0, 255) ,  # Bird (fuchsia).
    (0, 255, 255) ,  # Sky (aqua).
    (255, 255, 0) ,  # Cat (yellow).
]

palette = np.array(
    [[255, 128, 0],
     [255, 153, 51],
     [255, 178, 102],
     [230, 230, 0],
     [255, 153, 255],
     [153, 204, 255],
     [255, 102, 255],
     [255, 51, 255],
     [102, 178, 255],
     [51, 153, 255],
     [255, 153, 153],
     [255, 102, 102],
     [255, 51, 51],
     [153, 255, 153],
     [102, 255, 102],
     [51, 255, 51],
     [0, 255, 0], [0, 0, 255],
     [255, 0, 0],
     [255, 255, 255]])
skeleton = [[16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13], [6, 12],
            [7, 13], [6, 7],
            [6, 8], [7, 9],
            [8, 10], [9, 11],
            [2, 3],
            [1, 2], [1, 3],
            [2, 4], [3, 5],
            [4, 6], [5, 7]]

pose_limb_color = palette[[
    0, 0, 0, 0, 7, 7, 7, 9, 9,
    9, 9, 9, 16, 16, 16, 16,
    16, 16, 16
]]
pose_kpt_color = palette[[
    16, 16, 16, 16, 16, 9, 9,
    9, 9, 9, 9, 0, 0, 0, 0, 0,
    0
]]

if platform.machine() != 'aarch64':
    import requests
    import onnx


models = {
    'models/public/onnx/resnet18_opset9.onnx': {'model_url': 'https://git.ti.com/cgit/jacinto-ai/jacinto-ai-modelzoo/plain/models/vision/classification/imagenet1k/torchvision/resnet18_opset9.onnx', 'type': 'onnx'},
    'models/public/tflite/mobilenet_v1_1.0_224.tflite': {'model_url': 'https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_224/1/default/1?lite-format=tflite', 'type': 'tflite'},

}

def download_model(mpath):

    headers = {
    'User-Agent': 'My User Agent 1.0',
    'From': 'aid@ti.com'  # This is another valid field
    }

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
            r = requests.get(url, allow_redirects=True, headers=headers)
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

def print_soc_info():
    soc = os.getenv('SOC')
    if soc == 'am68a':
        print(f'SoC: J721S2/AM68A')
        print(f'OPP:')
        print(f'    Cortex-A72 @2GHZ')
        print(f'    DSP C7x-MMA @1GHZ')
        print(f'    2xDDR @4266 MT/s\n')
    elif soc == 'am68pa':
        print(f'SoC: J721E/AM68PA')
        print(f'OPP:')
        print(f'    Cortex-A72 @2GHZ')
        print(f'    DSP C7x-MMA @1GHZ')
        print(f'    DDR @4266 MT/s\n')
    elif soc == 'am69a':
        print(f'SoC: J784S4/AM69A')
        print(f'OPP:')
        print(f'    Cortex-A72 @2GHZ')
        print(f'    4xDSP C7x-MMA @1GHZ')
        print(f'    DDR @4266 MT/s\n')
    elif soc == 'am62a':
        print(f'SoC: AM62A')
        print(f'OPP:')
        print(f'    Cortex-A53 @1.4GHZ')
        print(f'    DSP C7x-MMA @1GHZ')
        print(f'    DDR @4266 MT/s\n')

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
    read_total  = benchmark_dict['ddr:read_end'] - benchmark_dict['ddr:read_start']
    write_total   = benchmark_dict['ddr:write_end'] - benchmark_dict['ddr:write_start']

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

def get_eval_configs(task_type, runtime_type, num_quant_bits, last_artifacts_id=None, model_selection = None, experimental_models = False):

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
        'run_import' : False,
        'model_selection': model_selection,
        'experimental_models' : experimental_models,
        'tidl_offload' : True,
        'modelartifacts_path': prebuilts_dir,
    }
    settings = ConfigSettings(settings_dict)
    prebuilt_configs = select_configs(settings, os.path.join(prebuilts_dir, f'{num_quant_bits}bits'), runtime_type, remove_models=True)
    merged_list = get_name_key_pair_list(prebuilt_configs.keys(), runtime_type)
    #print("merged_list: ", prebuilt_configs.keys())

    model_selection_artifacts_key = get_selected_artifacts_id()
    if not model_selection_artifacts_key is None:
        last_artifacts_id = model_selection_artifacts_key if len(merged_list) > model_selection_artifacts_key else None
    elif last_artifacts_id is None:
        last_artifacts_id = merged_list[0][1] if len(merged_list) > 0 else None
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
    layout = pipeline_config['preprocess'].get_param('data_layout')
    reverse_channels = pipeline_config['preprocess'].get_param('reverse_channels')
    mean = pipeline_config['session'].get_param('input_mean')
    scale = pipeline_config['session'].get_param('input_scale')

    if type(size) is tuple:
        size = list(size)

    if mean is not None:
        mean = list(mean)

    if scale is not None:
        scale = list(scale)

    return size, mean, scale, layout, reverse_channels


def task_type_to_dataset_list(task_type):
    assert task_type == 'classification'
    return ['imagenet']

"""
class loggerWritter():
    # Redirect c- stdout and stderr to a couple of files.
"""

class loggerWritter():
    def __init__(self, logname):
        self.logname = logname
        sys.stdout.flush()
        sys.stderr.flush()

        if self.logname == None:
            self.logpath_out = os.devnull
            self.logpath_err = os.devnull
        else:
            self.logpath_out = (logname + "_out.log")
            self.logpath_err = (logname + "_err.log")

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

"""
def get_svg_path(model_dir):
    # Finds all *.svg inside a model artifact folder and return *.svg's files paths inside a list.
"""

def get_svg_path(model_dir):
    inputNetFile = []
    artifacts_root = os.path.abspath(model_dir)
    for subdir, dirs, files in sorted(os.walk(artifacts_root)):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension in ['.svg']:
                netFile = os.path.join(subdir, file)
                inputNetFile.append(os.path.join(*Path(netFile).parts[3:]))
    return inputNetFile

# ------------------------------------------------------------------------------
# Adapted from https://github.com/princeton-vl/pose-ae-train/
# Original licence: Copyright (c) 2017, umich-vl, under BSD 3-Clause License.
# ------------------------------------------------------------------------------


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        batch size: B
        num keypoints: K
        num persons: N
        hight of heatmaps: H
        width of heatmaps: W
        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        res (np.ndarray[N, K, 2]): Refined coordinates.
    """

    batch_heatmaps = copy.deepcopy(batch_heatmaps)
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)
    batch_heatmaps = np.transpose(batch_heatmaps,
                                  (2, 3, 0, 1)).reshape(H, W, -1)
    batch_heatmaps_pad = cv2.copyMakeBorder(
        batch_heatmaps, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
    batch_heatmaps_pad = np.transpose(
        batch_heatmaps_pad.reshape(H + 2, W + 2, B, K),
        (2, 3, 0, 1)).flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(np.int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords

def gather(a, dim, index):
    expanded_index = [index if dim==i else np.arange(a.shape[i]).reshape([-1 if i==j else 1 for j in range(a.ndim)]) for i in range(a.ndim)]
    return a[tuple(expanded_index)]

def _py_max_match(scores):
    """Apply munkres algorithm to get the best match.

    Args:
        scores(np.ndarray): cost matrix.

    Returns:
        np.ndarray: best match.
    """
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(int)
    return tmp


def _match_by_tag(inp, params):
    """Match joints by tags. Use Munkres algorithm to calculate the best match
    for keypoints grouping.

    Note:
        number of keypoints: K
        max number of people in an image: M (M=30 by default)
        dim of tags: L
            If use flip testing, L=2; else L=1.

    Args:
        inp(tuple):
            tag_k (np.ndarray[KxMxL]): tag corresponding to the
                top k values of feature map per keypoint.
            loc_k (np.ndarray[KxMx2]): top k locations of the
                feature maps for keypoint.
            val_k (np.ndarray[KxM]): top k value of the
                feature maps per keypoint.
        params(Params): class Params().

    Returns:
        np.ndarray: result of pose groups.
    """
    assert isinstance(params, _Params), 'params should be class _Params()'

    tag_k, loc_k, val_k = inp

    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2]),
                        dtype=np.float32)

    joint_dict = {}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = params.joint_order[i]

        tags = tag_k[idx]
        joints = np.concatenate((loc_k[idx], val_k[idx, :, None], tags), 1)
        mask = joints[:, 2] > params.detection_threshold
        tags = tags[mask]
        joints = joints[mask]

        if joints.shape[0] == 0:
            continue

        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0]
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            grouped_keys = list(joint_dict.keys())[:params.max_num_people]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            if (params.ignore_too_much
                    and len(grouped_keys) == params.max_num_people):
                continue

            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)

            if params.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (diff_normed,
                     np.zeros((num_added, num_added - num_grouped),
                              dtype=np.float32) + 1e10),
                    axis=1)

            pairs = _py_max_match(diff_normed)
            for row, col in pairs:
                if (row < num_added and col < num_grouped
                        and diff_saved[row][col] < params.tag_threshold):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]

    ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32)
    return ans


def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)


class _Params:
    """A class of parameter.

    Args:
        cfg(Config): config.
    """

    def __init__(self, cfg):
        self.num_joints = cfg['num_joints']
        self.max_num_people = cfg['max_num_people']

        self.detection_threshold = cfg['detection_threshold']
        self.tag_threshold = cfg['tag_threshold']
        self.use_detection_val = cfg['use_detection_val']
        self.ignore_too_much = cfg['ignore_too_much']

        if self.num_joints == 17:
            self.joint_order = [
                i - 1 for i in
                [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]
        else:
            self.joint_order = list(np.arange(self.num_joints))


class HeatmapParser:
    """The heatmap parser for post processing."""

    def __init__(self, cfg):
        self.params = _Params(cfg)
        self.tag_per_joint = cfg['tag_per_joint']
        self.nms_kernel = cfg['nms_kernel']
        self.nms_padding = cfg['nms_padding']
        self.use_udp = cfg.get('use_udp', False)

    def nms(self, heatmaps):
        """Non-Maximum Suppression for heatmaps.
        """
        for i,heatmap in enumerate(heatmaps[0]):
            maxm = pool2d(
                heatmap,
                kernel_size=self.nms_kernel,
                stride=1,
                padding=self.nms_padding,
                pool_mode='max')

            maxm = np.equal(maxm,heatmap)
            heatmap = heatmap * maxm

            heatmaps[0][i] = heatmap

        return heatmaps

    def match(self, tag_k, loc_k, val_k):
        """Group keypoints to human poses in a batch.

        Args:
            tag_k (np.ndarray[NxKxMxL]): tag corresponding to the
                top k values of feature map per keypoint.
            loc_k (np.ndarray[NxKxMx2]): top k locations of the
                feature maps for keypoint.
            val_k (np.ndarray[NxKxM]): top k value of the
                feature maps per keypoint.

        Returns:
            list
        """

        def _match(x):
            return _match_by_tag(x, self.params)

        return list(map(_match, zip(tag_k, loc_k, val_k)))

    def top_k(self, heatmaps, tags):
        """Find top_k values in an image.

        Note:
            batch size: N ==1
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            max number of people: M
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (torch.Tensor[NxKxHxW])
            tags (torch.Tensor[NxKxHxWxL])

        Return:
            dict: A dict containing top_k values.

            - tag_k (np.ndarray[NxKxMxL]):
                tag corresponding to the top k values of
                feature map per keypoint.
            - loc_k (np.ndarray[NxKxMx2]):
                top k location of feature map per keypoint.
            - val_k (np.ndarray[NxKxM]):
                top k value of feature map per keypoint.
        """
        heatmaps = self.nms(heatmaps)
        N, K, H, W = heatmaps.shape
        heatmaps = np.reshape(heatmaps,[N,K,-1])

        ind = np.zeros((N,K,self.params.max_num_people),int)
        val_k = np.zeros((N,K,self.params.max_num_people))
        for i,heatmap in enumerate(heatmaps[0]):
            ind[0][i] = heatmap.argsort()[-self.params.max_num_people:][::-1]
            val_k[0][i] = heatmap[ind[0][i]]

        tags = np.reshape(tags,(tags.shape[0], tags.shape[1], W * H, -1))
        tag_k = np.concatenate([np.expand_dims(gather(tags[...,i],2,ind),axis=3) for i in range(tags.shape[3])],axis=3)

        x = ind % W
        y = ind // W

        ind_k = np.concatenate((np.expand_dims(x,axis=3),np.expand_dims(y,axis=3)), axis=3)

        ans = {
            'tag_k': tag_k,
            'loc_k': ind_k,
            'val_k': val_k
        }

        return ans

    @staticmethod
    def adjust(ans, heatmaps):
        """Adjust the coordinates for better accuracy.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            ans (list(np.ndarray)): Keypoint predictions.
            heatmaps (torch.Tensor[NxKxHxW]): Heatmaps.
        """
        _, _, H, W = heatmaps.shape
        for batch_id, people in enumerate(ans):
            for people_id, people_i in enumerate(people):
                for joint_id, joint in enumerate(people_i):
                    if joint[2] > 0:
                        x, y = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = heatmaps[batch_id][joint_id]
                        if tmp[min(H - 1, yy + 1), xx] > tmp[max(0, yy - 1),
                                                             xx]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[yy, min(W - 1, xx + 1)] > tmp[yy,
                                                             max(0, xx - 1)]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[batch_id][people_id, joint_id,
                                      0:2] = (x + 0.5, y + 0.5)
        return ans

    @staticmethod
    def refine(heatmap, tag, keypoints, use_udp=False):
        """Given initial keypoint predictions, we identify missing joints.

        Note:
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmap: np.ndarray(K, H, W).
            tag: np.ndarray(K, H, W) |  np.ndarray(K, H, W, L)
            keypoints: np.ndarray of size (K, 3 + L)
                        last dim is (x, y, score, tag).
            use_udp: bool-unbiased data processing

        Returns:
            np.ndarray: The refined keypoints.
        """

        K, H, W = heatmap.shape
        if len(tag.shape) == 3:
            tag = tag[..., None]

        tags = []
        for i in range(K):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(int)
                x = np.clip(x, 0, W - 1)
                y = np.clip(y, 0, H - 1)
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0)
        ans = []

        for _heatmap, _tag in zip(heatmap, tag):
            # distance of all tag values with mean tag of
            # current detected people
            distance_tag = (((_tag -
                              prev_tag[None, None, :])**2).sum(axis=2)**0.5)
            norm_heatmap = _heatmap - np.round(distance_tag)

            # find maximum position
            y, x = np.unravel_index(np.argmax(norm_heatmap), _heatmap.shape)
            xx = x.copy()
            yy = y.copy()
            # detection score at maximum position
            val = _heatmap[y, x]
            if not use_udp:
                # offset by 0.5
                x += 0.5
                y += 0.5

            # add a quarter offset
            if _heatmap[yy, min(W - 1, xx + 1)] > _heatmap[yy, max(0, xx - 1)]:
                x += 0.25
            else:
                x -= 0.25

            if _heatmap[min(H - 1, yy + 1), xx] > _heatmap[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = np.array(ans)

        if ans is not None:
            for i in range(K):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                    keypoints[i, :3] = ans[i, :3]

        return keypoints

    def parse(self, heatmaps, tags, adjust=True, refine=True):
        """Group keypoints into poses given heatmap and tag.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (torch.Tensor[NxKxHxW]): model output heatmaps.
            tags (torch.Tensor[NxKxHxWxL]): model output tagmaps.

        Returns:
            tuple: A tuple containing keypoint grouping results.

            - ans (list(np.ndarray)): Pose results.
            - scores (list): Score of people.
        """
        ans = self.match(**self.top_k(heatmaps, tags))

        if adjust:
            if self.use_udp:
                for i in range(len(ans)):
                    if ans[i].shape[0] > 0:
                        ans[i][..., :2] = post_dark_udp(
                            ans[i][..., :2].copy(), heatmaps[i:i + 1, :])
            else:
                ans = self.adjust(ans, heatmaps)

        scores = [i[:, 2].mean() for i in ans[0]]

        if refine:
            ans = ans[0]
            # for every detected person
            for i in range(len(ans)):
                heatmap_numpy = heatmaps[0]
                tag_numpy = tags[0]
                if not self.tag_per_joint:
                    tag_numpy = np.tile(tag_numpy,
                                        (self.params.num_joints, 1, 1, 1))
                ans[i] = self.refine(
                    heatmap_numpy, tag_numpy, ans[i], use_udp=self.use_udp)
            ans = [ans]

        return ans, scores
#####################################################################################################################
#
#Human Pose Estimation Postprocessing functions
#
#####################################################################################################################
def define_cfg(udp=False):

    cfg = {}

    cfg['higher_hr'] = False
    if not(udp):
        cfg['project2image'] = True
        cfg['use_udp'] = False
    else:
        cfg['project2image'] = False
        cfg['use_udp'] = True

    cfg['num_joints'] = 17
    cfg['max_num_people'] = 30
    cfg['with_heatmaps'] = [True, True]
    cfg['with_ae'] = [True, False]
    cfg['tag_per_joint'] = True
    cfg['flip_index'] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    cfg['s'] = 1
    cfg['test_scale_factor'] = [1]
    cfg['flip_test'] = False

    cfg['adjust'] = True
    cfg['refine'] = True

    cfg['detection_threshold'] = 0.1
    cfg['tag_threshold'] = 1
    cfg['use_detection_val'] = True
    cfg['ignore_too_much'] = False

    cfg['nms_kernel'] = 5
    cfg['nms_padding'] = 2

    return cfg


# def _ceil_to_multiples_of(x, base=64):
#     """Transform x to the integral multiple of the base."""
#     return int(np.ceil(x / base)) * base

# def _get_multi_scale_size(image_to_read,
#                           input_size,
#                           current_scale,
#                           min_scale):
#     """Get the size for multi-scale training.

#     Args:
#         image: Input image.
#         input_size (int): Size of the image input.
#         current_scale (float): Scale factor.
#         min_scale (float): Minimal scale.
#         use_udp (bool): To use unbiased data processing.
#             Paper ref: Huang et al. The Devil is in the Details: Delving into
#             Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

#     Returns:
#         tuple: A tuple containing multi-scale sizes.

#         - (w_resized, h_resized) (tuple(int)): resized width/height
#         - center (np.ndarray)image center
#         - scale (np.ndarray): scales wrt width/height
#     """

#     image = mmcv.imread(image_to_read)
#     h, w, _ = image.shape

#     # calculate the size for min_scale
#     min_input_size = _ceil_to_multiples_of(min_scale * input_size, 64)
#     if w < h:
#         w_resized = int(min_input_size * current_scale / min_scale)
#         h_resized = int(
#             _ceil_to_multiples_of(min_input_size / w * h, 64) * current_scale /
#             min_scale)
#         if cfg['use_udp']:
#             scale_w = w - 1.0
#             scale_h = (h_resized - 1.0) / (w_resized - 1.0) * (w - 1.0)
#         else:
#             scale_w = w / 200.0
#             scale_h = h_resized / w_resized * w / 200.0
#     else:
#         h_resized = int(min_input_size * current_scale / min_scale)
#         w_resized = int(
#             _ceil_to_multiples_of(min_input_size / h * w, 64) * current_scale /
#             min_scale)
#         if cfg['use_udp']:
#             scale_h = h - 1.0
#             scale_w = (w_resized - 1.0) / (h_resized - 1.0) * (h - 1.0)
#         else:
#             scale_h = h / 200.0
#             scale_w = w_resized / h_resized * h / 200.0
#     if cfg['use_udp']:
#         center = (scale_w / 2.0, scale_h / 2.0)
#     else:
#         center = np.array([round(w / 2.0), round(h / 2.0)])

#     # print("\n")
#     # print("h, w : {}, {}".format(h,w))
#     # print("min_scale,input_size: {}, {}".format(min_scale,input_size))
#     # print("min_input_size : {}".format(min_input_size))
#     # print("h_resized, w_resized: {},{}".format(h_resized, w_resized))
#     # print("center : {}".format(center))
#     # print("scale_h, scale_w : {}, {}".format(scale_h, scale_w))


#     return (w_resized, h_resized), center, np.array([scale_w, scale_h])

def get_multi_stage_outputs(outputs,
                            outputs_flip,
                            num_joints,
                            with_heatmaps,
                            with_ae,
                            tag_per_joint=True,
                            flip_index=None,
                            project2image=True,
                            size_projected=None,
                            align_corners=False):
    """Inference the model to get multi-stage outputs (heatmaps & tags), and
    resize them to base sizes.
    Also to aggregate them.
    """

    heatmaps_avg = 0
    heatmaps = []
    tags = []

    aggregated_heatmaps = None
    tags_list = []

    flip_test = outputs_flip is not None

    offset_feat = num_joints

    heatmaps_avg += outputs[0][:, :num_joints]
    tags.append(outputs[0][:, offset_feat:])

    heatmaps.append(heatmaps_avg)

    if flip_test and flip_index:
        # perform flip testing
        heatmaps_avg = 0
        offset_feat = num_joints

        heatmaps_avg += outputs_flip[0][:, :num_joints][:, flip_index, :, :]
        tags.append(outputs_flip[0][:, offset_feat:])
        if tag_per_joint:
            tags[-1] = tags[-1][:, flip_index, :, :]

        heatmaps.append(heatmaps_avg)

    #align corners on the basis of udp, if udp true then align
    #remember project2image is true mostly when udp is False
    if project2image and size_projected:

        dim = (size_projected[1], size_projected[0])

        final_heatmaps =[]
        final_tags = []

        new_heatmaps = np.empty((0,dim[0],dim[1]),int)
        for hms in heatmaps[0][0]:
            new_hms = cv2.resize(
                hms,
                dim,
                interpolation=cv2.INTER_LINEAR)
            new_heatmaps = np.append(new_heatmaps,[new_hms],axis=0)

        final_heatmaps.append(np.expand_dims(new_heatmaps,0))

        if flip_test:
            new_heatmaps_flipped = np.empty((0,dim[0],dim[1]),int)
            for hms in heatmaps[1][0]:
                new_hms = cv2.resize(
                    hms,
                    dim,
                    interpolation=cv2.INTER_LINEAR)
                new_heatmaps_flipped = np.append(new_heatmaps_flipped,[new_hms],axis=0)

            final_heatmaps.append(np.expand_dims(new_heatmaps_flipped,0))

        new_tags = np.empty((0,dim[0],dim[1]),int)
        for tms in tags[0][0]:
            new_tms =  cv2.resize(
                tms,
                dim,
                interpolation=cv2.INTER_LINEAR)
            new_tags = np.append(new_tags,[new_tms],axis=0)


        final_tags.append(np.expand_dims(new_tags,0))

        if flip_test:
            new_tags_flipped = np.empty((0,dim[0],dim[1]),int)
            for tms in tags[1][0]:
                new_tms = cv2.resize(
                    tms,
                    dim,
                    interpolation=cv2.INTER_LINEAR)
                new_tags_flipped = np.append(new_tags_flipped,[new_tms],axis=0)

            final_tags.append(np.expand_dims(new_tags_flipped,0))

    else:
        final_tags = tags
        final_heatmaps = heatmaps

    for tms in final_tags:
        tags_list.append(np.expand_dims(tms,axis=4))

    aggregated_heatmaps = (final_heatmaps[0] +
                    final_heatmaps[1]) / 2.0 if flip_test else final_heatmaps[0]

    tags = np.concatenate(tags_list,axis=4)

    return aggregated_heatmaps, tags


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords

def get_warp_matrix(theta, size_input, size_dst, size_target):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        matrix (np.ndarray): A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                              0.5 * size_input[1] * math.sin(theta) +
                              0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
                              0.5 * size_input[1] * math.cos(theta) +
                              0.5 * size_target[1])
    return matrix

def warp_affine_joints(joints, mat):
    """Apply affine transformation defined by the transform matrix on the
    joints.

    Args:
        joints (np.ndarray[..., 2]): Origin coordinate of joints.
        mat (np.ndarray[3, 2]): The affine matrix.

    Returns:
        matrix (np.ndarray[..., 2]): Result coordinate of joints.
    """
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return np.dot(
        np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
        mat.T).reshape(shape)

def get_group_preds(grouped_joints,
                    center,
                    scale,
                    heatmap_size,
                    use_udp=False):
    """Transform the grouped joints back to the image.

    Args:
        grouped_joints (list): Grouped person joints.
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        heatmap_size (np.ndarray[2, ]): Size of the destination heatmaps.
        use_udp (bool): Unbiased data processing.
             Paper ref: Huang et al. The Devil is in the Details: Delving into
             Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        list: List of the pose result for each person.
    """
    if use_udp:
        if grouped_joints[0].shape[0] > 0:
            heatmap_size_t = np.array(heatmap_size, dtype=np.float32) - 1.0
            trans = get_warp_matrix(
                theta=0,
                size_input=heatmap_size_t,
                size_dst=scale,
                size_target=heatmap_size_t)
            grouped_joints[0][..., :2] = \
                warp_affine_joints(grouped_joints[0][..., :2], trans)
        results = [person for person in grouped_joints[0]]
    else:
        results = []
        for person in grouped_joints[0]:
            joints = transform_preds(person, center, scale, heatmap_size)
            results.append(joints)
    return results


def oks_iou(g, d, a_g, a_d, sigmas=None, vis_thr=None):
    """Calculate oks ious.

    Args:
        g: Ground truth keypoints.
        d: Detected keypoints.
        a_g: Area of the ground truth object.
        a_d: Area of the detected object.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.

    Returns:
        list: The oks ious.
    """
    if sigmas is None:
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list(vg > vis_thr) and list(vd > vis_thr)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious

def oks_nms(kpts_db, thr, sigmas=None, vis_thr=None):
    """OKS NMS implementations.

    Args:
        kpts_db: keypoints.
        thr: Retain overlap < thr.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([k['score'] for k in kpts_db])
    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        inds = np.where(oks_ovr <= thr)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep

def show_result(img,
                result,
                skeleton=None,
                kpt_score_thr=0.3,
                bbox_color=None,
                pose_kpt_color=None,
                pose_limb_color=None,
                radius=4,
                thickness=1,
                font_scale=0.5,
                win_name='',
                show=False,
                show_keypoint_weight=False,
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (list[dict]): The results to draw over `img`
            (bbox_result, pose_result).
        skeleton (list[list]): The connection of keypoints.
        kpt_score_thr (float, optional): Minimum score of keypoints
            to be shown. Default: 0.3.
        pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
            If None, do not draw keypoints.
        pose_limb_color (np.array[Mx3]): Color of M limbs.
            If None, do not draw limbs.
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        win_name (str): The window name.
        show (bool): Whether to show the image. Default: False.
        show_keypoint_weight (bool): Whether to change the transparency
            using the predicted confidence scores of keypoints.
        wait_time (int): Value of waitKey param.
            Default: 0.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        Tensor: Visualized image only if not `show` or `out_file`
    """

    img = cv2.imread(img)
    img = img[:,:,::-1]
    img = img.copy()
    img_h, img_w, _ = img.shape

    pose_result = []
    for res in result:
        pose_result.append(res['keypoints'])

    for _, kpts in enumerate(pose_result):
        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)
            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(
                    kpt[1]), kpt[2]
                if kpt_score > kpt_score_thr:
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        r, g, b = pose_kpt_color[kid]
                        cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                   radius, (int(r), int(g), int(b)), -1)
                        transparency = max(0, min(1, kpt_score))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        r, g, b = pose_kpt_color[kid]
                        cv2.circle(img, (int(x_coord), int(y_coord)),
                                   radius, (int(r), int(g), int(b)), -1)

        # draw limbs
        if skeleton is not None and pose_limb_color is not None:
            assert len(pose_limb_color) == len(skeleton)
            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
                pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
                if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                        and pos1[1] < img_h and pos2[0] > 0
                        and pos2[0] < img_w and pos2[1] > 0
                        and pos2[1] < img_h
                        and kpts[sk[0] - 1, 2] > kpt_score_thr
                        and kpts[sk[1] - 1, 2] > kpt_score_thr):
                    r, g, b = pose_limb_color[sk_id]
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        X = (pos1[0], pos2[0])
                        Y = (pos1[1], pos2[1])
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                        angle = math.degrees(
                            math.atan2(Y[0] - Y[1], X[0] - X[1]))
                        stickwidth = 2
                        polygon = cv2.ellipse2Poly(
                            (int(mX), int(mY)),
                            (int(length / 2), int(stickwidth)), int(angle),
                            0, 360, 1)
                        cv2.fillConvexPoly(img_copy, polygon,
                                           (int(r), int(g), int(b)))
                        transparency = max(
                            0,
                            min(
                                1, 0.5 *
                                (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        cv2.line(
                            img,
                            pos1,
                            pos2, (int(r), int(g), int(b)),
                            thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)

    if out_file is not None:
        imwrite(img, out_file)

    return img

def vis_pose_result(img,
                    result,
                    kpt_score_thr=0.3,
                    show=False,
                    out_file=None,
                    thickness=1,
                    radius=4):

    img = show_result(
        img,
        result,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color,
        kpt_score_thr=kpt_score_thr,
        show=show,
        out_file=out_file)
    return img

def single_img_visualise(output, image_size, img_name, out_file, top, left, ratio, udp=False, thickness=2, radius=5, label='ae'):
    print(label)
    if 'ae' in label and 'yolo' not in label:
        cfg = define_cfg(udp)

        if cfg['use_udp']:
            base_size = (image_size,image_size)
            center = np.array([(image_size-1)/2.0,(image_size-1)/2.0])
            scale = np.array([image_size-1,image_size-1])
        else:
            base_size = (image_size,image_size)
            center = np.array([image_size/2,image_size/2])
            scale = np.array([image_size/200,image_size/200])

        parser = HeatmapParser(cfg)

        result = {}

        outputs_flipped = None

        aggregated_heatmaps, tags = get_multi_stage_outputs(
            output,
            outputs_flipped,
            cfg['num_joints'],
            cfg['with_heatmaps'],
            cfg['with_ae'],
            cfg['tag_per_joint'],
            cfg['flip_index'],
            cfg['project2image'],
            base_size,
            align_corners=cfg['use_udp'])

        grouped, scores = parser.parse(aggregated_heatmaps, tags,
            cfg['adjust'],
            cfg['refine'])

        preds = get_group_preds(
            grouped,
            center,
            scale, [aggregated_heatmaps.shape[3],
                    aggregated_heatmaps.shape[2]],
            use_udp=cfg['use_udp'])

        actual_size = cv2.imread(img_name).shape
        k = [actual_size[1],actual_size[0]]
        final_size = image_size
        # for converting the keypoints back to the original image

        if k[1]<k[0]:
            scale_it = final_size/k[0]
            value_it = (final_size - scale_it*k[1])/2
            for i in range(len(preds)):
                for j in range(len(preds[0])):
                    preds[i][j][0] = preds[i][j][0]/scale_it
                    preds[i][j][1] = (preds[i][j][1]-value_it)/scale_it
        else:
            scale_it = final_size/k[1]
            value_it = (final_size - scale_it*k[0])/2
            for i in range(len(preds)):
                for j in range(len(preds[0])):
                    preds[i][j][1] = preds[i][j][1]/scale_it
                    preds[i][j][0] = (preds[i][j][0]-value_it)/scale_it

        image_paths = []
        image_paths.append(img_name)

        output_heatmap = None

        result['preds'] = preds
        result['scores'] = scores
        result['image_paths'] = img_name
        result['output_heatmap'] = output_heatmap

        pose_results = []
        for idx, pred in enumerate(result['preds']):
            area = (np.max(pred[:, 0]) - np.min(pred[:, 0])) * (
                np.max(pred[:, 1]) - np.min(pred[:, 1]))
            pose_results.append({
                'keypoints': pred[:, :3],
                'score': result['scores'][idx],
                'area': area,
            })

        pose_nms_thr=0.9
        keep = oks_nms(pose_results, pose_nms_thr, sigmas=None)
        pose_results = [pose_results[_keep] for _keep in keep]

        output_image = vis_pose_result(
            img_name,
            pose_results,
            kpt_score_thr=0.3,
            show=False,
            out_file=out_file,
            thickness=thickness,
            radius=radius)
    else:
        pose_results = np.squeeze(output[0])
        output_image = vis_box_pose_result(
            img_name,
            pose_results,
            top, left, ratio,
            score_threshold=0.6,
            thickness=thickness,
            radius=radius)


    return output_image

def vis_box_pose_result(img_file, output, top, left, ratio, score_threshold=0.3,thickness=2, radius=5):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    det_bboxes, det_scores, det_labels, kpts = output[:, 0:4], output[:, 4], output[:, 5], output[:, 6:]
    det_bboxes[:, 0::2] = det_bboxes[:, 0::2]*ratio - left
    det_bboxes[:, 1::2] = det_bboxes[:, 1::2]*ratio - top
    kpts[:, 0::3] = kpts[:, 0::3]*ratio - left
    kpts[:, 1::3] = kpts[:, 1::3]*ratio - top
    #adjust the coordinates here
    img = cv2.imread(img_file)[:, :, ::-1]
    img = np.ascontiguousarray(img)
    #To generate color based on det_label, to look into the codebase of Tensorflow object detection api.
    for idx in range(len(det_bboxes)):
        det_bbox = det_bboxes[idx]
        kpt = kpts[idx]
        if det_scores[idx]>score_threshold:
            color_map = _CLASS_COLOR_MAP[int(det_labels[idx])]
            img = cv2.rectangle(img, (det_bbox[0], det_bbox[1]), (det_bbox[2], det_bbox[3]), color_map[::-1], 1)
            plot_skeleton_kpts(img, kpt, thickness=thickness, radius=radius)
    return img

def plot_skeleton_kpts(im, kpts, steps=3, thickness=2, radius=5):
    num_kpts = len(kpts) // steps
    #plot keypoints
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        if conf > 0.5: #Confidence of a keypoint has to be greater than 0.5
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
    #plot skeleton
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        conf1 = kpts[(sk[0]-1)*steps+2]
        conf2 = kpts[(sk[1]-1)*steps+2]
        if conf1>0.5 and conf2>0.5: # For a limb, both the keypoint confidence must be greater than 0.5
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=thickness)


