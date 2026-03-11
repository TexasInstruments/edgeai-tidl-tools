# Copyright (c) {2024 - 2024} Texas Instruments Incorporated
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
"""
Module containing Slice layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np
from .common import get_all_deformal_convolution_nodes

def tidl_convert_patch_merging_to_reshp_tr_reshp(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    '''
                        inp (NCHW)
                         |
    --------------------------------------------
    |              |             |             |
 Slice(1,2)   Slice(1,2)     Slice(1,2)   Slice(1,2)
    |              |             |             |
    --------------------------------------------
                         |
                     Concat(3)
                         |
                        out
    Note the numbers inside () are axes, Each Slice has step 2 and their starts are (0,0), (0,1), (1,0), (1,1)
    
    '''
    
    start_end = []
    for inp in graph.tensors().values():
        if isinstance(inp, gs.Constant):
            continue
        if inp.shape is None:
            continue
        if len(inp.shape) != 4:
            continue
        if len(inp.outputs)!=4 or any(o.op != 'Slice' for o in inp.outputs):
            continue
        slices = list(inp.outputs)
        for ind, slice in enumerate(slices):
            nodes = []
            node = slice
            while node.op == 'Slice':
                if len(node.inputs)!= 5:
                    break
                if any(not isinstance(i , gs.Constant) for i in node.inputs[1:]):
                    break
                step  = node.inputs[-1].values 
                if step!=2:
                    break
                if len(node.outputs[0].outputs) != 1:
                    break
                nodes.append(node)
                node = node.outputs[0].outputs[0]
            slices[ind] = nodes+[node] if node.op == 'Concat'  and len(nodes) == 2 else []
        if any(slice_list[-1] is not slices[0][-1] for slice_list in slices[1:]):
            continue
        if slices[0][-1].attrs['axis'] not in (-1, len(inp.shape)-1):
            continue
        starts = [[s.inputs[1].values[0] for s in slice_list[:-1]] for slice_list in slices]
        if any(start_list not in starts for start_list in ([[0, 0], [1, 0], [0, 1], [1, 1]])):
            continue
        # ends = [[s.inputs[2] for s in slice_list] for slice_list in slices]
        axes = [[s.inputs[3].values[0] for s in slice_list[:-1]] for slice_list in slices]
        if not all(1 in axes_list and 2 in axes_list for axes_list in axes):
            continue
        start_end.append((inp,slices[0][-1].outputs[0]))
        
    for inp, out in start_end:
        inp.outputs.clear()
        out.inputs.clear()
        N,C,H,W = inp.shape
        if C%2 or H%2:
            continue
        shape1 = [N,C//2,2,H,W]
        reshape1_out = gs.Variable(f'{inp.name}_reshape1_out', inp.dtype, shape1)
        shape1 = gs.Constant(f'{inp.name}_shape1', values=np.array(shape1).astype(np.int64))
        reshape1 = gs.Node('Reshape', f'{inp.name}_reshape1', {}, [inp, shape1], [reshape1_out])
        graph.nodes.append(reshape1)
        transout = gs.Variable(f'{inp.name}_transpose_out', inp.dtype, [N,C//2,H,2,W])
        tranpose = gs.Node('Transpose',f'{inp.name}_transpose',{'perm' :[0,1,3,2,4]},[reshape1_out], [transout])
        graph.nodes.append(tranpose)        
        shape2 = [N,C//2,H//2,W*4]
        shape2 = gs.Constant(f'{inp.name}_shape2', values=np.array(shape2).astype(np.int64))
        reshape2 = gs.Node('Reshape', f'{inp.name}_reshape2', {}, [transout, shape2], [out])
        graph.nodes.append(reshape2)
        
def tidl_convert_nonsingular_strided_slice_to_gather(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Replace Slice nodes with stride > 1 with Gather nodes.
    Adds transpose operations when needed to satisfy hardware constraints.
    """
    
    deform_convs = get_all_deformal_convolution_nodes(graph)
    
    for node in graph.nodes:
        # Only process Slice nodes with strides
        if node.op != "Slice" or len(node.inputs) < 5:
            continue
        
        # Skip deformable convolution slices
        if any(node in dc for dc in deform_convs):
            continue
        
        # STEP 1: Extract slice parameters
        steps_tensor = node.inputs[4]
        if not isinstance(steps_tensor, gs.Constant):
            continue
        
        step = steps_tensor.values
        step = int(step.flat[0]) if isinstance(step, np.ndarray) else int(step)
        
        # # Only optimize if stride > 1
        if step == 1 or step == 0:
            continue
        
        # Get axis
        axis = 0
        if len(node.inputs) > 3 and isinstance(node.inputs[3], gs.Constant):
            axis_val = node.inputs[3].values
            axis = int(axis_val.flat[0]) if isinstance(axis_val, np.ndarray) else int(axis_val)
        
        # Get start and end
        if not isinstance(node.inputs[1], gs.Constant) or not isinstance(node.inputs[2], gs.Constant):
            continue
        
        start = node.inputs[1].values
        start = int(start.flat[0]) if isinstance(start, np.ndarray) else int(start)
        
        end = node.inputs[2].values
        end = int(end.flat[0]) if isinstance(end, np.ndarray) else int(end)
        
        # Get input shape
        input_shape = node.inputs[0].shape
        if not input_shape:
            continue
        
        # Handle negative axis
        if axis < 0:
            axis += len(input_shape)
        
        # Handle negative start/end
        dim_size = input_shape[axis]
        if start < 0:
            start += dim_size
        if end < 0:
            end += dim_size
        
        start = max(0, min(start, dim_size))
        end = max(0, min(end, dim_size))

        # Get the indices 
        if start <= end and step > 0:
            indice_list = list(range(start, end, step))
        elif start > end and step < 0:
            indice_list = list(range(start, end-1, step))
        else:
            continue

        # STEP 2: Create gather indices
        indices = gs.Constant(
            name=f"{node.name}_indices",
            values=np.array(indice_list, dtype=np.int64)
        )
        
        # STEP 3: Find valid gather position
        # Find last consecutive dimension=1 from start, target is next position
        last_consecutive_one = -1
        for i in range(axis):
            if input_shape[i] == 1:
                last_consecutive_one = i
            else:
                break  # Stop at first non-1
        
        # Determine target position
        if all(input_shape[i] == 1 for i in range(axis)):
            # All dimensions before axis are 1 - no transpose needed
            target_pos = axis
        elif last_consecutive_one >= 0:
            # Found consecutive 1s - place after them
            target_pos = last_consecutive_one + 1
        else:
            # No consecutive 1s - move to position 0
            target_pos = 0
        
        # ============================================================
        # STEP 4: Create gather or transpose-gather-transpose pattern
        # ============================================================
        if target_pos == axis:
            # Case A: Direct gather (no transpose needed)
            gather = gs.Node(
                op="Gather",
                name=f"{node.name}_gather",
                attrs={"axis": axis},
                inputs=[node.inputs[0], indices],
                outputs=node.outputs
            )
            graph.nodes.append(gather)
        else:
            # Case B: Need transpose -> gather -> transpose back
            ndim = len(input_shape)
            
            # Create permutation: move axis to target_pos
            perm = list(range(ndim))
            perm.insert(target_pos, perm.pop(axis))
            
            # Inverse permutation to restore original layout
            inv_perm = [perm.index(i) for i in range(ndim)]
            
            # Create intermediate variables
            trans1_out = gs.Variable(
                name=f"{node.name}_t1",
                dtype=node.inputs[0].dtype
            )
            gather_out = gs.Variable(
                name=f"{node.name}_g",
                dtype=node.inputs[0].dtype
            )
            
            # Transpose 1: Move gather axis to valid position
            trans1 = gs.Node(
                op="Transpose",
                name=f"{node.name}_transpose1",
                attrs={"perm": perm},
                inputs=[node.inputs[0]],
                outputs=[trans1_out]
            )
            
            # Gather at target position
            gather = gs.Node(
                op="Gather",
                name=f"{node.name}_gather",
                attrs={"axis": target_pos},
                inputs=[trans1_out, indices],
                outputs=[gather_out]
            )
            
            # Transpose 2: Restore original dimension order
            trans2 = gs.Node(
                op="Transpose",
                name=f"{node.name}_transpose2",
                attrs={"perm": inv_perm},
                inputs=[gather_out],
                outputs=node.outputs
            )
            
            # Add all three nodes
            graph.nodes.extend([trans1, gather, trans2])
        
        # ============================================================
        # STEP 5: Remove original slice node
        # ============================================================
        node.inputs.clear()
        node.outputs.clear()
    
    # Cleanup disconnected nodes
    graph.cleanup().toposort()


def tidl_expand_slice_across_multiple_axis (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert the Slice across multiple axis to multiple slices in series
    """
    nodes = graph.nodes
    # node_iter = 0
    
    for node_iter, node in enumerate(nodes):
        if (node.op == "Slice"):
            if len(node.inputs)<2:
                # Slice-1 implementation (old opset)
                len_starts = len(node.attrs['starts'])
                if len_starts > 1:
                    logging.warning(f"Slice-1 implementation is not supported as of now. We suggest using higher opset version.")
                    continue
            elif isinstance(node.inputs[1], gs.Constant):
                len_starts = node.inputs[1].values.size
            else:
                continue
            
            if len_starts > 1:
                # slice is across multiple axis
                node_name = node.inputs[1].name if node.name=='' else node.name 
                
                len_inputs = len(node.inputs)
                if len_inputs<4: # all_axes isn't defined
                    logging.warning(f"All Axes isn't defined in a multi-axis slice node : {node_name}. Skipping the conversion")
                    continue
                
                all_starts = node.inputs[1].values
                len_axes = len(all_starts)
                all_ends = node.inputs[2].values
                all_axes = node.inputs[3].values
                if len_inputs > 4: # steps are provided
                    all_steps = node.inputs[4].values
                else:
                    all_steps = np.ones(len_axes, dtype=np.int64)
    
                prev_slice_node = None
                for i in range(len_axes):
                    curr_start = gs.Constant(name= node_name + "_start_" + str(node_iter) + "_" + str(i), values=np.array([all_starts[i]]))
                    curr_end = gs.Constant(name= node_name + "_end_" + str(node_iter) + "_" + str(i), values=np.array([all_ends[i]]))
                    curr_axis = gs.Constant(name= node_name + "_axis_" + str(node_iter) + "_" + str(i), values=np.array([all_axes[i]]))
                    curr_step = gs.Constant(name= node_name + "_step_" + str(node_iter) + "_" + str(i), values=np.array([all_steps[i]]))
                    interim_output = gs.Variable(name= node_name + "_out_" + str(node_iter) + "_" + str(i), dtype=np.float32)
                    
                    node_output = node.outputs[0] if i==(len_axes-1) else interim_output
                    node_input = node.inputs[0] if i==0 else prev_slice_node.outputs[0]

                    slice_node = gs.Node(name= node_name + "_" + str(node_iter) + "_" + str(all_axes[i]), op= "Slice",
                            inputs= [node_input, curr_start, curr_end, curr_axis, curr_step], outputs = [node_output])
                    
                    logging.debug(f"Adding Node {slice_node.name}")
                    graph.nodes.append(slice_node)
                    prev_slice_node = slice_node 
                    
                node.outputs.clear()
                # node_iter += 1


def tidl_convert_2_dimension_slice_to_maxpool (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert the Slice present in 2 dimensions to a maxpool layer. A slice with steps=[2,2] with axes=[2,3] can be converted 
    to maxpool with a stride of 2. There could be 2 different possibilities i.e. NCHW format
    and NHWC format. In the second part, we will have to introduce transpose -> maxpool -> transpose. 
    """
    nodes = graph.nodes
    node_iter = 0
    
    for node in nodes:
        if (node.op == "Slice") and len(node.inputs) > 3 and isinstance(node.inputs[3], gs.Constant) and (node.inputs[3].values.size == 2): 
            # slice has axes input and axes is across 2 axis
            if len(node.inputs) < 5:
                # does not have access to the steps input , continuing
                continue

            input_shape = node.inputs[0].shape
            if input_shape is None or len(input_shape) != 4:
                logging.warning(f"Shape Inference is not done, or the dimensions are not equal to 4")
                continue

            # Normalize negative axes to positive values
            axes = node.inputs[3].values.copy()
            axes = np.where(axes < 0, axes + len(input_shape), axes)
            

            if abs(axes[0] - axes[1]) == 1 and (node.inputs[4].values[0] == node.inputs[4].values[1]):
                # slice axes are consecutive and both axes has the same step
                stride = node.inputs[4].values[0]
                # stride == 2 is supported in TIDL, if greater than that, it will be broken into 2 by other optimizations
                node_name = node.name if node.name else f"slice_to_maxpool_{node_iter}"

                end_axes = sorted(axes)[-1]
                if end_axes != (len(input_shape) - 1):
                    # would require a transpose before and after the maxpool
                    transpose_output = gs.Variable(name=f"{node_name}_transpose_out")
                    maxpool_output = gs.Variable(name=f"{node_name}_maxpool_out")
                    perm1 = [0, 3, 1, 2] if end_axes==2 else [2, 3, 0, 1]
                    perm2 = [0, 2, 3, 1] if end_axes==2 else [2, 3, 0, 1]

                    transpose1_node = gs.Node(
                        name=f"{node_name}_transpose1",
                        op="Transpose",
                        attrs={"perm": perm1},
                        inputs=[node.inputs[0]],
                        outputs=[transpose_output]
                    )
                    logging.debug(f"Adding Node {transpose1_node.name}")

                    # MaxPool2D
                    maxpool_node = gs.Node(
                        name=f"{node_name}_maxpool",
                        op="MaxPool",
                        attrs={
                            "kernel_shape": [1, 1],
                            "strides": [stride, stride]
                        },
                        inputs=[transpose_output],
                        outputs=[maxpool_output]
                    )
                    logging.debug(f"Adding Node {maxpool_node.name}")
                    
                    # NCHW to NHWC transpose
                    transpose2_node = gs.Node(
                        name=f"{node_name}_transpose2",
                        op="Transpose",
                        attrs={"perm": perm2},
                        inputs=[maxpool_output],
                        outputs=[node.outputs[0]]
                    )
                    logging.debug(f"Adding Node {transpose2_node.name}")
                    
                    graph.nodes.extend([transpose1_node, maxpool_node, transpose2_node])
                #
                else:
                    maxpool_node = gs.Node(
                        name=f"{node_name}_maxpool",
                        op="MaxPool",
                        attrs={
                            "kernel_shape": [1, 1],
                            "strides": [stride, stride],
                            "pads": [0, 0, 0, 0]
                        },
                        inputs=[node.inputs[0]],
                        outputs=[node.outputs[0]]
                    )
                    logging.debug(f"Adding Node {maxpool_node.name}")
                    graph.nodes.append(maxpool_node)
                #
            #
            node.outputs.clear()
            node_iter += 1
        #
    #


def tidl_add_slice_step_axis (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    TIDL has some bug with supporting slice when step axis is not present, adding the default value
    """
    nodes = graph.nodes
    node_iter = 0
    
    for node in nodes:
        if (node.op == "Slice") and isinstance(node.inputs[1], gs.Constant) and len(node.inputs)<5: 
            # slice is found, however, it has missing axes and/or step
            node_name = node.inputs[1].name if node.name=='' else node.name 
            len_inputs = len(node.inputs)
            node_input = node.inputs[0]
            slice_start =  node.inputs[1]
            slice_end =  node.inputs[2]
            slice_axes = node.inputs[3]
            slice_steps = gs.Constant(name= node_name + "_step_" + str(node_iter), values=np.ones(slice_start.values.shape[0], dtype=np.int64))
            
            slice_node = gs.Node(name= node_name + "_" + str(node_iter), op= "Slice",
                inputs= [node_input, slice_start, slice_end, slice_axes, slice_steps], outputs = [node.outputs[0]])
            
            logging.debug(f"Adding Node {slice_node.name}")
            graph.nodes.append(slice_node)
                
            node.outputs.clear()
            node_iter += 1
            
            
def tidl_eliminate_noop_slice(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Eliminate Slice nodes that are no-ops (i.e., they do not change the input tensor).
    """
    nodes = graph.nodes

    for node in nodes:
        if node.op != "Slice":
            continue

        # Get starts, ends, axes, steps as node.inputs
        if len(node.inputs) < 4:
            logging.debug(f"Skipping Slice node '{node.name}': not enough inputs for starts/ends/axes.")
            continue  # Need at least input, starts, ends, axes

        input_var = node.inputs[0]
        starts = node.inputs[1]
        ends = node.inputs[2]
        axes = node.inputs[3]
        steps = node.inputs[4] if len(node.inputs) > 4 else None

        if not (isinstance(starts, gs.Constant) and isinstance(ends, gs.Constant) and isinstance(axes, gs.Constant)):
            logging.debug(f"Skipping Slice node '{node.name}': starts/ends/axes are not all constants.")
            continue
        if steps is not None and not isinstance(steps, gs.Constant):
            logging.debug(f"Skipping Slice node '{node.name}': steps is not a constant.")
            continue

        starts_v = starts.values
        ends_v = ends.values
        axes_v = axes.values
        steps_v = steps.values if steps is not None else np.ones_like(starts_v)

        # Only support <=4D input and batch size 1
        input_shape = input_var.shape
        if input_shape is not None and (len(input_shape) > 4 or input_shape[0] != 1):
            logging.debug(f"Skipping Slice node '{node.name}': input shape not supported ({input_shape}).")
            continue

        # All steps must be 1
        if not np.all(steps_v == 1):
            logging.debug(f"Skipping Slice node '{node.name}': steps are not all 1 ({steps_v}).")
            continue

        # Check if slice is a no-op: starts==0, ends==input_shape[axis]
        is_noop = True
        for i, axis in enumerate(axes_v):
            axis = int(axis)
            start = starts_v[i]
            end = ends_v[i]
            dim = input_shape[axis] if input_shape is not None and axis < len(input_shape) else None
            if start != 0:
                logging.debug(f"Slice node '{node.name}' is not a no-op: start[{i}]={start} != 0.")
                is_noop = False
                break
            if dim is not None:
                if end<0:
                    end += dim
                if end < dim:
                    logging.debug(f"Slice node '{node.name}' is not a no-op: end[{i}]={end} != dim[{axis}]={dim}.")
                    is_noop = False
                    break

        if is_noop:
            logging.info(f"Eliminating no-op Slice node: {node.name}")
            # If the output of this slice is a graph output, update graph.outputs
            for out in node.outputs:
                for idx, graph_out in enumerate(graph.outputs):
                    if out is graph_out:
                        logging.debug(f"Updating graph output from '{out.name}' to '{input_var.name}' for node '{node.name}'.")
                        graph.outputs[idx] = input_var
            # Replace all consumers of this node's output with the input
            for out in node.outputs:
                outs = list(out.outputs)
                for consumer in outs:
                    for idx, inp in enumerate(consumer.inputs):
                        if inp is out:
                            logging.debug(f"Redirecting consumer '{consumer.name}' input from '{out.name}' to '{input_var.name}'.")
                            consumer.inputs[idx] = input_var
            node.outputs.clear()