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

onesList = [1.0,1.0,1.0]

def tidlOnnxModelOptimize(in_model_path, out_model_path, scaleList=[0.0078125,0.0078125,0.0078125], meanList=[128.0, 128.0, 128.0]):
    #Read Model
    meanList = [x * -1 for x in meanList]
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
    biasTensor = helper.make_tensor("TIDL_preProc_Bias",TensorProto.FLOAT,[1,nInCh, 1, 1],np.array(meanList,dtype=np.float32))
    scaleTensor = helper.make_tensor("TIDL_preProc_Scale",TensorProto.FLOAT,[1, nInCh, 1, 1],np.array(scaleList,dtype=np.float32))

    #Add these tensors to initList:
    initList.append(biasTensor)
    initList.append(scaleTensor)

    #Cast Node:
    attrib_dict = {"to":TensorProto.FLOAT}
    cast = onnx.helper.make_node('Cast',inputs=[originalGraph.input[0].name+"Net_IN"],outputs=['TIDL_cast_in'], **attrib_dict)

    #Add Node:
    addNode = onnx.helper.make_node('Add',inputs=["TIDL_cast_in","TIDL_preProc_Bias"],outputs=["TIDL_Scale_In"])

    #Scale Node:
    scaleNode = onnx.helper.make_node('Mul',inputs=["TIDL_Scale_In","TIDL_preProc_Scale"],outputs=[originalGraph.input[0].name]) #Assumption that input[0].name is the input node

    nodeList = [cast, addNode, scaleNode] + nodeList #Toplogically Sorted

    outSequence = originalGraph.output
    #Check for Argmax:
    for node in nodeList:
        if node.op_type == "ArgMax":
            #Check if it is final output:
            if node.output[0] == originalGraph.output[0].name:
                #Argmax Output is final output:
                attrib_dict_1 = {"to":TensorProto.UINT8}
                cast_out = onnx.helper.make_node('Cast',inputs=[originalGraph.output[0].name],outputs=[originalGraph.output[0].name+'TIDL_cast_out'], **attrib_dict_1)
                nodeList = nodeList + [cast_out] #Toplogically Sorted
                outSequence = [helper.make_tensor_value_info(originalGraph.output[0].name+'TIDL_cast_out', TensorProto.UINT8,outDims)]

    #Construct Graph:
    newGraph = helper.make_graph(
        nodeList,
        'Rev_Model',
        [helper.make_tensor_value_info(originalGraph.input[0].name+"Net_IN", TensorProto.UINT8, inDims)],
        outSequence,
        initList
        )
    #Construct Model:
    #op.version = 11
    model_def_noShape = helper.make_model(newGraph, producer_name='onnx-TIDL', opset_imports=[op])
    model_def = shape_inference.infer_shapes(model_def_noShape)

    try:
        onnx.checker.check_model(model_def)
    except onnx.checker.ValidationError as e:
        print('Converted model is invalid: %s' % e)
    else:
        print('Converted model is valid!')
        onnx.save_model(model_def, out_model_path)
