# Copyright (c) {2015 - 2021} Texas Instruments Incorporated
#
# All rights reserved not granted herein.
#
# Limited License.
#
# Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
# license under copyrights and patents it now or hereafter owns or controls to make,
# have made, use, import, offer to sell and sell ("Utilize") this software subject to the
# terms herein.  With respect to the foregoing patent license, such license is granted
# solely to the extent that any such patent is necessary to Utilize the software alone.
# The patent license shall not apply to any combinations which include this software,
# other than combinations with devices manufactured by or for TI ("TI Devices").
# No hardware patent is licensed hereunder.
#
# Redistributions must preserve existing copyright notices and reproduce this license
# (including the above copyright notice and the disclaimer and (if applicable) source
# code license limitations below) in the documentation and/or other materials provided
# with the distribution
#
# Redistribution and use in binary form, without modification, are permitted provided
# that the following conditions are met:
#
# *       No reverse engineering, decompilation, or disassembly of this software is
# permitted with respect to any software provided in binary form.
#
# *       any redistribution and use are licensed by TI for use only with TI Devices.
#
# *       Nothing shall obligate TI to provide you with source code for the software
# licensed and provided to you in object code.
#
# If software source code is provided to you, modification and redistribution of the
# source code are permitted provided that the following conditions are met:
#
# *       any redistribution and use of the source code, including any resulting derivative
# works, are licensed by TI for use only with TI Devices.
#
# *       any redistribution and use of any object code compiled from the source code
# and any resulting derivative works, are licensed by TI for use only with TI Devices.
#
# Neither the name of Texas Instruments Incorporated nor the names of its suppliers
#
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# DISCLAIMER.
#
# THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
import onnx
from onnx import helper
from onnx import TensorProto,shape_inference 
import numpy as np
import sys
import getopt
import os

###########Function description#############
# This Function takes a RGB trained model and update the inputs to the model to accept 
# YUV(NV12) image format
# Expected NV12 format is as follows for a 224x224 RGB 
# input1 = 224x224 Y data in uint8 format
# input2 = 112x224 UV interleaved data in uint8 format
###########Function description#############
def addYUVConv(in_model_path, out_model_path):
    #Read Model
    model = onnx.load_model(in_model_path)
    op = onnx.OperatorSetIdProto()
    #Track orginal opset:
    op.version = model.opset_import[0].version
    #Get Graph:
    originalGraph = model.graph
    #Get Nodes:
    originalNodes = originalGraph.node
    #Get Initializers:
    originalInitializers = originalGraph.initializer
    #Create Lists
    nodeList = [node for node in originalNodes]
    initList = [init for init in originalInitializers]

    nInCh = int(originalGraph.input[0].type.tensor_type.shape.dim[1].dim_value)

    #Input & Output Dimensions:
    inDims = tuple([x.dim_value for x in originalGraph.input[0].type.tensor_type.shape.dim])
    outDims = tuple([x.dim_value for x in originalGraph.output[0].type.tensor_type.shape.dim])
    
    #Construct bias & scale tensors
    weights = [1.164, 0.0, 1.596,
                1.164, -0.391, -0.813,
                1.164, 2.018, 0.0 ]
    bias= [-222.912, 135.488, -276.928]
    scales = np.array([1, 2, 2, 1], dtype=np.int64)
    weight_init = onnx.helper.make_tensor(
            name='TIDL_preProc_YUV_RGB_weights',
            data_type=TensorProto.FLOAT,
            dims=[3,3,1,1],
            vals=np.array(weights,dtype=np.float32))
    bias_init = onnx.helper.make_tensor(
            name='TIDL_preProc_YUV_RGB_bias',
            data_type=TensorProto.FLOAT,
            dims=[3,1],
            vals=np.array(bias,dtype=np.float32))
    resize_scales = onnx.helper.make_tensor(
            name='TIDL_preProc_UV_resize_sizes',
            data_type=TensorProto.FLOAT,
            dims=[4],
            vals=scales)
    dummy = onnx.helper.make_tensor(
            name='roi',
            data_type=TensorProto.FLOAT,
            dims=(0,),
            vals=[])   

    #Conv Node:
    concat = onnx.helper.make_node('Concat',name="Concat_YUV",inputs=[originalGraph.input[0].name+"Y_IN","Transpose_UV_2_out"], axis=1, outputs=["YUV_RGB_Concat_out"],)
    conv = onnx.helper.make_node('Conv',name="Conv_YUV_RGB",inputs=["YUV_RGB_Concat_out","TIDL_preProc_YUV_RGB_weights","TIDL_preProc_YUV_RGB_bias"],outputs=[originalGraph.input[0].name])
    transpose = onnx.helper.make_node('Transpose',name="Transpose_UV",inputs=[originalGraph.input[0].name+"UV_IN"], perm = [0, 2, 3, 1], outputs=["Transpose_UV_out"])
    resize = onnx.helper.make_node('Resize',name="Resize_UV",inputs=["Transpose_UV_out", "const_roi_node", "TIDL_preProc_UV_resize_sizes"] , mode = "nearest", outputs=["Resize_UV_out"])
    const_roi_node = helper.make_node("Constant", [], ["const_roi_node"], value=dummy, name="const_roi_node")
    transpose_2 = onnx.helper.make_node('Transpose',name="Transpose_UV_2",inputs=["Resize_UV_out"], perm = [0, 3, 1, 2], outputs=["Transpose_UV_2_out"])
       
    # nodeList = [cast, addNode, scaleNode] + nodeList #Toplogically Sorted   
    nodeList = [transpose, const_roi_node, resize, transpose_2, concat, conv] + nodeList #Toplogically Sorted

    initList = [  resize_scales,  weight_init, bias_init] + initList
    outSequence = originalGraph.output
    #Construct Graph:
    newGraph = helper.make_graph(
        nodeList,
        'Rev_Model',
        [helper.make_tensor_value_info(originalGraph.input[0].name+"Y_IN", TensorProto.FLOAT, [inDims[0], 1, inDims[2], inDims[3]]),
         helper.make_tensor_value_info(originalGraph.input[0].name+"UV_IN", TensorProto.FLOAT, [inDims[0], 2, int(inDims[2]/2), int(inDims[3]/2)]),
        ],
        outSequence,
        initList,
        )
    #Construct Model:
    op.version = 11
    model_def_noShape = helper.make_model(newGraph, producer_name='onnx-TIDL', opset_imports=[op])
    model_def = shape_inference.infer_shapes(model_def_noShape)    
    try:
        onnx.checker.check_model(model_def)
    except onnx.checker.ValidationError as e:
        print('Converted model is invalid: %s' % e)
    else:
        print('Converted model is valid!')
        onnx.save_model(model_def, out_model_path)


###########Function description#############
# This Function helps to seperate Y and UV data from a NV12 format image
# ne can convert a jpg image to NV12 image by mentioning the required size 
# ffmpeg -y -colorspace bt470bg -i airshow.jpg -s 224x224 -pix_fmt nv12 airshow.yuv 
# output generated for 224x224 NV12 format is as follows
# creates 224x224 Y data in uint8 format
# creates 112x224 UV interleaved data in uint8 format
###########Function description#############
def createInputYUVData(input_file, width, height):
    yuv_file = input_file.replace(".jpg",".yuv")
    cmd = "ffmpeg -y -colorspace bt470bg -i "+ input_file+ " -s "+str(height)+"x"+ str(width)+" -pix_fmt nv12 "+ yuv_file
    os.system(cmd)
    input_data = np.fromfile(yuv_file,dtype=np.uint8,count=width*height,offset=0)
    input_file = yuv_file.replace('.yuv', '')
    input_data.tofile(input_file + "_Y_uint8.bin")
    input_data = np.fromfile(yuv_file,dtype=np.uint8,count=width*int(height/2),offset=width*height)
    input_data.tofile(input_file + "_UV_uint8.bin")

# if __name__ == "__main__":
#     in_model_path= "/home/a0496663/work/edgeaitidltools/rel86/edgeai-tidl-tools/models/public/resnet18_opset9.onnx"
#     out_model_path= "/home/a0496663/work/edgeaitidltools/rel86/edgeai-tidl-tools/models/public/resnet18_opset9_yuv.onnx"    
#     addYUVConv(in_model_path, out_model_path)
    
def main(argv):
   inputfile = ''
   outputfile = ''
   gen_yuv_data = 0
   width = 224
   height = 224
   opts, args = getopt.getopt(argv,"hi:o:g:w:l:",["ifile=","ofile=","gen_yuv_data=","width=","height="])
   for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile> -o <outputfile> -g <gen_yuv_data> -w <width> -l <heigh>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
      elif opt in ("-g", "--gen_yuv_data"):
         gen_yuv_data = int(arg )
      elif opt in ("-w", "--width"):
         width = int(arg )
      elif opt in ("-l", "--height"):
         height = int(arg )                  
   if(inputfile == ''):
        print ('test.py -i <inputfile> -o <outputfile>')
        sys.exit()
   if(outputfile == ''):
        temp = inputfile
        outputfile = temp.replace('.onnx', '_yuv.onnx')
   print("inputfile: "+ inputfile)
   print("outputfile: "+ outputfile)
   if(gen_yuv_data == 1):
       print("generating YUV data")
       print("width: "+ str(width) )
       print("height: "+ str(height) )
       createInputYUVData(input_file=inputfile,width=width,height=height)
   else:
       print("generating YUV model")
       addYUVConv(inputfile, outputfile)
if __name__ == "__main__":
   main(sys.argv[1:])