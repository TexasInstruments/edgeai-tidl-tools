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
import sys

SUPPORTED_MODES = ("YUV420SP",)
def convert_model_to_yuv(model_path, output_path, input_names=None, mean=None, scale=None, mode="YUV420SP",):

   '''
   Function to convert RGB trained model and update the inputs to the model to accept YUV(NV12) image format
   
   seperate Y and UV data from a jpg image or .yuv image (NV12 format) and saves it.
   Creates Y and UV interleaved data in uint8 format. Uses ffmpeg library to perform
   the conversion so make sure ffmpeg is installed.

   Args:
        model_path: Path to the input ONNX model
        output_path: Path to save the modified ONNX model
        (Optional)input_names: List of input names to convert in case of multiple inputs. Default: All inputs of model
        (Optional)mean: Input means if applicable. Default: None
        (Optional)scale: Input scale if applicable. Default: None
        (Optional) mode: YUV Mode. Default: YUV420SP. Allowed: [YUV420SP]
   '''

   assert mode in SUPPORTED_MODES, f"Only {','.join(SUPPORTED_MODES)} are supported"

   if mean is not None and scale is not None:
      print(f"Adding mean and scale to {model_path}...")
      from osrt_model_tools.onnx_tools.tidl_onnx_model_utils import optimize_model_input 
      optimize_model_input(model_path, output_path, scale=scale, mean=mean)
      model_path = output_path
      print(f"Mean and scale added to model and saved in {output_path}...")

   print(f"Loading model from {model_path}...")
   model = onnx.load_model(model_path)
   graph = model.graph

   inDims = []
   gNodes = [] # container to hold the newly added nodes in topologically sorted order
   gInitList = []

   print("Converting to accept YUV format...")

   for inp_idx in range(len(graph.input)):
      if input_names is not None and len(input_names) > 0 and graph.input[inp_idx].name not in input_names:
         continue
      inDims.append(tuple([x.dim_value for x in graph.input[inp_idx].type.tensor_type.shape.dim]))
      B, _, H, W = inDims[inp_idx]
      UV_shape = [B, H//2, W//2, 2]
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

      transpose = onnx.helper.make_node("Transpose", name=f"Transpose_UV_{inp_idx}", inputs=[graph.input[inp_idx].name + "_UV_IN"], perm=[0, 3, 1, 2], outputs=[f"Transpose_UV_output_{inp_idx}"])
      new_nodes.append(transpose)
      curr_output_layer = f"Transpose_UV_output_{inp_idx}"

      scales = np.array([1, 1, 2, 2], dtype=np.int64)
      resize_uv_scales = onnx.helper.make_tensor(name=f"Resize_uv_scales_{inp_idx}", data_type=TensorProto.FLOAT, dims=[4], vals=scales)
      dummy_uv = onnx.helper.make_tensor(
               name='roi_uv',
               data_type=TensorProto.FLOAT,
               dims=(0,),
               vals=[])
      roi_uv_node = helper.make_node("Constant", [], [f"roi_uv_output_{inp_idx}"], value=dummy_uv, name=f"roi_uv_{inp_idx}")
      resize_uv = onnx.helper.make_node("Resize", name=f"Resize_uv_{inp_idx}", inputs=[curr_output_layer, f"roi_uv_output_{inp_idx}", f"Resize_uv_scales_{inp_idx}"], mode="nearest", outputs=[f"Resized_uv_output_{inp_idx}"])
      concat = onnx.helper.make_node("Concat", name=f"Concat_YUV_{inp_idx}", inputs=[graph.input[inp_idx].name + "_Y_IN", f"Resized_uv_output_{inp_idx}"], axis=1, outputs=[f"Concat_YUV_output_{inp_idx}"])
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
         name=f"Conv_YUV_RGB_{inp_idx}",
         inputs=[
               f"Concat_YUV_output_{inp_idx}",
               f"TIDL_preProc_YUV_RGB_weights_{inp_idx}",
               f"TIDL_preProc_YUV_RGB_bias_{inp_idx}"
         ],
         outputs=[graph.input[inp_idx].name]
      )
      new_nodes.append(conv)
      initList = [dummy_uv, resize_uv_scales, weight_init, bias_init]

      gNodes = new_nodes + gNodes
      gInitList = initList + gInitList

   yuv_graph = helper.make_graph(
      gNodes + [node for node in graph.node],
      "YUV_model",
      inTensors,
      graph.output,
      gInitList + [init for init in graph.initializer]
   )

   #Construct Model:
   op = onnx.OperatorSetIdProto()
   op.version = 11
   model_def_noShape = helper.make_model(yuv_graph, producer_name='onnx-TIDL', opset_imports=[op])
   model_def = shape_inference.infer_shapes(model_def_noShape)    

   try:
      onnx.checker.check_model(model_def)
   except onnx.checker.ValidationError as e:
      print('Converted model is invalid: %s' % e)
   else:
      print('Converted model is valid!')
      onnx.save_model(model_def, output_path)
      print(f"Converted model saved to {output_path}")

def convert_image_to_yuv(input_path, input_width, input_height):
   '''
   Function to seperate Y and UV data from a jpg image or .yuv image (NV12 format) and saves it.
   Creates Y and UV interleaved data in uint8 format. Uses ffmpeg library to perform
   the conversion so make sure ffmpeg is installed.

   Args:
        input_path: Path to the input
        input_width: Width of the input
        input_height: Height of the input
   '''
   input_path = input_path.strip()
   print("Generating Y and UV inputs from {input_path}...")
   yuv_file = input_path.replace(".jpg",".yuv")
   cmd = "ffmpeg -y -colorspace bt470bg -i " + input_path + " -s "+str(input_height) + "x" + str(input_width) + " -pix_fmt nv12 " + yuv_file
   ret = os.system(cmd)
   if ret == 0:
      input_data = np.fromfile(yuv_file,dtype=np.uint8,count=input_width*input_height,offset=0)
      input_path = yuv_file.replace('.yuv', '')
      y_path = input_path + "_Y_uint8.bin"
      input_data.tofile(y_path)
      input_data = np.fromfile(yuv_file,dtype=np.uint8,count=input_width*int(input_height/2),offset=input_width*input_height)
      uv_path = input_path + "_UV_uint8.bin"
      input_data.tofile(uv_path)
      print("Y data saved to {y_path}")
      print("UV data saved to {uv_path}")
   else:
      print(f"Command: '{cmd}' returned with exit code {ret}")

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("-t", "--task", type=str, help="convert_model_to_yuv or convert_image_to_yuv") 
   parser.add_argument("-i", "--input", type=str, help="Path to input") 

   parser.add_argument("-o", "--output", type=str, help="Path to save the output model (applicable for convert_model_to_yuv)")
   parser.add_argument("-m", "--mode", choices=SUPPORTED_MODES, default="YUV420SP", help="Layout of the input data (applicable for convert_model_to_yuv)")
   parser.add_argument("-in","--input_names", type=str, nargs="+", help="Names of the input to convert in case model may have multiple inputs coming from different sources (applicable for convert_model_to_yuv)")
   parser.add_argument("--mean", type=float, nargs="+", help="Mean for normalizing the input (applicable for convert_model_to_yuv)")
   parser.add_argument("--scale", type=float, nargs="+", help="Scale for normalizing the input (applicable for convert_model_to_yuv)")

   parser.add_argument("-w", "--width", type=int, default=224, help="Width of the input data (applicable for convert_image_to_yuv)")
   parser.add_argument("-l", "--height", type=int, default=224, help="Height of the input data (applicable for convert_image_to_yuv)")

   if args.task not in ["convert_image_to_yuv", "convert_model_to_yuv"]:
      print("Only convert_model_to_yuv or convert_image_to_yuv task type is allowed.")
      sys.exit(-1)

   if args.task == "convert_image_to_yuv":
      convert_jpg_to_yuv(args.input, args.width, args.height)
   else:
      convert_model_to_yuv(args.input, args.output, args.input_names, args.mean, args.scale, args.mode)