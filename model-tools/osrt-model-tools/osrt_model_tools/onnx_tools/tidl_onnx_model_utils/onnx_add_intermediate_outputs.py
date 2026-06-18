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
import argparse
import sys

def add_intermediate_outputs(model_path, output_path):
    """
    Add all intermediate outputs to an ONNX model.
    
    Args:
        model_path: Path to the input ONNX model
        output_path: Path to save the modified ONNX model
    """
    print(f"Loading model from {model_path}...")
    onnx_model = onnx.load(model_path)
    
    # Keep track of outputs we've already added to avoid duplicates
    existing_outputs = set(output.name for output in onnx_model.graph.output)
    added_count = 0
    
    print("Adding intermediate outputs...")
    for i in range(len(onnx_model.graph.node)):
        node = onnx_model.graph.node[i]
        for j in range(len(node.output)):
            output_name = node.output[j]
            
            # Skip if this output is already in the model outputs
            if output_name in existing_outputs:
                continue
                
            # Create a new ValueInfoProto for each output
            intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            intermediate_layer_value_info.name = output_name
            
            # Add to model outputs and track it
            onnx_model.graph.output.append(intermediate_layer_value_info)
            existing_outputs.add(output_name)
            added_count += 1
    
    print(f"Added {added_count} intermediate outputs to the model")
    onnx.save(onnx_model, output_path)
    print(f"Modified model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add intermediate outputs to an ONNX model')
    parser.add_argument('--input', '-i', required=True, help='Path to input ONNX model')
    parser.add_argument('--output', '-o', required=True, help='Path to save the modified ONNX model')
    
    # Add example usage to help text
    parser.epilog = '''
    Example usage:
        python onnx_add_intermediate_outputs.py -i /path/to/model.onnx -o /path/to/output_model.onnx
    '''
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    
    args = parser.parse_args()
    
    try:
        add_intermediate_outputs(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
