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

def extract_model(model_path, output_path, input_names, output_names):
    """
    Extract a subgraph from an ONNX model.
    
    Args:
        model_path: Path to the input ONNX model
        output_path: Path to save the extracted ONNX model
        input_names: List of input names to extract
        output_names: List of output names to extract
    """
    print(f"Loading model from {model_path}...")
    print(f"Extracting subgraph with inputs: {input_names} and outputs: {output_names}")
    onnx.utils.extract_model(model_path, output_path, input_names, output_names)
    print(f"Extracted model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract a subgraph from an ONNX model')
    parser.add_argument('--input', '-i', required=True, help='Path to input ONNX model')
    parser.add_argument('--output', '-o', required=True, help='Path to save the extracted ONNX model')
    parser.add_argument('--input-names', '-in', required=True, nargs='+', help='List of input names to extract')
    parser.add_argument('--output-names', '-on', required=True, nargs='+', help='List of output names to extract')
    
    # Add example usage to help text
    parser.epilog = '''
    Example usage:
        python onnx_extract_model.py -i /path/to/model.onnx -o /path/to/extracted_model.onnx -in input1 input2 -on output1 output2
    '''
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    
    args = parser.parse_args()
    
    try:
        extract_model(args.input, args.output, args.input_names, args.output_names)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
