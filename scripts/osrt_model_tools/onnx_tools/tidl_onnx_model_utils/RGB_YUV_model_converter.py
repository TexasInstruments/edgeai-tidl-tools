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
import os
import argparse
import onnx
from onnx import helper
from onnx import TensorProto,shape_inference 
import numpy as np

SUPPORTED_MODES = ("YUV420SP", "YUV420P")

###########Function description#############
# This Function takes a RGB trained model and update the inputs to the model to accept 
# YUV(NV12) image format
# Expected NV12 format is as follows for a 224x224 RGB 
# input1 = 224x224 Y data in uint8 format
# input2 = 112x224 UV interleaved data in uint8 format
###########Function description#############
def addYUVConv(in_model_path, out_model_path, args):
   mode = args.mode
   assert mode in SUPPORTED_MODES

   #Read Model
   model = onnx.load_model(in_model_path)
   graph = model.graph

   inDims = []
   for inp_idx in range(len(graph.input)):
      if args.input_names is not None and len(args.input_names) > 0 and graph.input[inp_idx].name not in args.input_names:
         continue
      inDims.append(tuple([x.dim_value for x in graph.input[inp_idx].type.tensor_type.shape.dim]))
   
   gNodes = [] # container to hold the newly added nodes in topologically sorted order
   gInitList = []

   for inp_idx in range(len(graph.input)):
      if args.input_names is not None and len(args.input_names) > 0 and graph.input[inp_idx].name not in args.input_names:
         continue
      B, _, H, W = inDims[inp_idx]
      UV_shape = [B, 2, H//2, W//2] if mode == "YUV420P" else [B, H//2, W//2, 2]
      inTensors = [
         helper.make_tensor_value_info(
               graph.input[inp_idx].name + "_Y_IN",
               TensorProto.FLOAT,
               [B, 1, H, W]        
         ),
         helper.make_tensor_value_info(
               graph.input[inp_idx].name + "_UV_IN",
               TensorProto.FLOAT,
               UV_shape
         ),
      ]

      curr_output_layer = inTensors[-1].name
      new_nodes = []
      if mode == "YUV420SP":
         transpose = onnx.helper.make_node("Transpose", name=f"Transpose_UV", inputs=[graph.input[inp_idx].name + "_UV_IN"], perm=[0, 3, 1, 2], outputs=[f"Transpose_UV_output_{inp_idx}"])
         new_nodes.append(transpose)
         curr_output_layer = f"Transpose_UV_output_{inp_idx}"

      scales = np.array([1, 1, 2, 2], dtype=np.int64)
      resize_uv_scales = onnx.helper.make_tensor(name="Resize_uv_scales", data_type=TensorProto.FLOAT, dims=[4], vals=scales)
      dummy_uv = onnx.helper.make_tensor(
               name='roi_uv',
               data_type=TensorProto.FLOAT,
               dims=(0,),
               vals=[])
      roi_uv_node = helper.make_node("Constant", [], [f"roi_uv_output_{inp_idx}"], value=dummy_uv, name="roi_uv")
      resize_uv = onnx.helper.make_node("Resize", name="Resize_uv", inputs=[curr_output_layer, f"roi_uv_output_{inp_idx}", "Resize_uv_scales"], mode="nearest", outputs=[f"Resized_uv_output_{inp_idx}"])
      concat = onnx.helper.make_node("Concat", name="Concat_YUV", inputs=[graph.input[inp_idx].name + "_Y_IN", f"Resized_uv_output_{inp_idx}"], axis=1, outputs=[f"Concat_YUV_output_{inp_idx}"])
      new_nodes.extend([roi_uv_node, resize_uv, concat])

      # adding conv to convert YUV to RGB
      weights = [1.164, 0.0, 1.596,
                  1.164, -0.391, -0.813,
                  1.164, 2.018, 0.0 ]
      bias= [-222.912, 135.488, -276.928]

      weight_init = onnx.helper.make_tensor(
               name=f'TIDL_preProc_YUV_RGB_weights_{inp_idx}',
               data_type=TensorProto.FLOAT,
               dims=[3,3,1,1],
               vals=np.array(weights,dtype=np.float32))
      bias_init = onnx.helper.make_tensor(
               name=f'TIDL_preProc_YUV_RGB_bias_{inp_idx}',
               data_type=TensorProto.FLOAT,
               dims=[3,1],
               vals=np.array(bias,dtype=np.float32))

      conv = onnx.helper.make_node(
         'Conv',
         name="Conv_YUV_RGB",
         inputs=[
               f"Concat_YUV_output_{inp_idx}",
               f"TIDL_preProc_YUV_RGB_weights_{inp_idx}",
               f"TIDL_preProc_YUV_RGB_bias_{inp_idx}"
         ],
         outputs=[graph.input[inp_idx].name]
      )
      new_nodes.append(conv)
      gNodes = new_nodes + gNodes
      gInitList = [dummy_uv, resize_uv_scales, weight_init, bias_init] + gInitList

   yuv_graph = helper.make_graph(
      gNodes + [node for node in graph.node],
      "YUV_model",
      inTensors,
      graph.output,
      gInitList + [init for init in graph.initializer]
   )

   #Construct Model:
   op = onnx.OperatorSetIdProto()
   op.version = 18
   model_def_noShape = helper.make_model(yuv_graph, producer_name='onnx-TIDL', opset_imports=[op])
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


def parse():
   parser = argparse.ArgumentParser()
   parser.add_argument("-i", "--input", type=str, help="Path to model or input (if you want to generate yuv input)") 
   parser.add_argument("-o", "--output", type=str, help="Path to save the output model") 
   parser.add_argument("-g", "--gen_yuv_data", action="store_true", help="Generate YUV input")
   parser.add_argument("-w", "--width", type=int, default=224, help="Width of the input data")
   parser.add_argument("-h", "--height", type=int, default=224, help="Height of the input data")
   parser.add_argument("-m", "--mode", choices=SUPPORTED_MODES, help="Layout of the Input Data", default="YUV420SP")
   parser.add_argument("--input_names", type=str, nargs="+", help="Names of the input to convert. Sometimes the model may have multiple inputs coming from different sources. With this flag you can define specific inputs to convert into YUV")

   return parser.parse_args()

def main():
   args = parse()

   if args.gen_yuv_data:
      print("Generating YUV input data")
      createInputYUVData(input_file=args.input, width=args.width, height=args.height)
   else:
      if args.output == "":
         args.output = args.input.replace(".onnx", "_yuv.onnx")
      print("Adding YUV input data convert layer")
      addYUVConv(args.input, args.output, args)


if __name__ == "__main__":
   main()