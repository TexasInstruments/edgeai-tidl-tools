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

import subprocess
import csv
import filecmp
import os
import platform
import sys
final_report = []

enable_debug = True

ref_outputs_base_dir = 'test_data' 
rt_base_dir_py = 'examples/osrt_python/'
rt_base_dir_bash = 'scripts'

test_configs = [
                {'script_name':'tflrt_delegate.py', 'script_dir':'tfl','lang':'py', 'rt_type':'tfl'},
                {'script_name':'onnxrt_ep.py', 'script_dir':'ort','lang':'py', 'rt_type':'ort'},
                {'script_name':'dlr_inference_example.py', 'script_dir':'tvm_dlr','lang':'py', 'rt_type':'dlr'},
                {'script_name':'run_tfl_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'tfl'}
    ]
currIdx = 0
if platform.machine() == 'aarch64':
    device = 'j7es'
else:
    device = 'pc'


def run_cmd(cmd, dir):
    try:
        msg = f'Command : {cmd} in Dir : {dir} Started'
        print(msg)
        lines = []
        process = subprocess.Popen([cmd], shell=True, cwd=dir, stdout=subprocess.PIPE,  universal_newlines=True, stderr=subprocess.STDOUT)
        while True:
            b = process.stdout.readline()
            if not len(b):
                break
            sys.stdout.write(b)
            sys.stdout.flush()
            lines.append(b.rstrip())

        if process.returncode is not None and process.returncode != 0 :
            msg = f'Command : {cmd} in Dir : {dir} Failed with Error code {process.returncode}'
            print(msg)
            exit(-1)
        return lines
    except Exception as e:
        raise e

for test_config in test_configs:
    script_name = test_config['script_name']
    rt_type = test_config['rt_type']
    if(test_config['lang'] == 'bash'):
        rt_base_dir = rt_base_dir_bash
        curr_rt_base_dir= os.path.join(rt_base_dir,test_config['script_dir'])
        curr_ref_outputs_base_dir = ref_outputs_base_dir+'/'+rt_type+'-refs-'+device+'/'
        cmd = ('bash '+ script_name)

    elif(test_config['lang'] == 'py'):
        rt_base_dir = rt_base_dir_py

        curr_rt_base_dir= os.path.join(rt_base_dir,test_config['script_dir'])
        curr_ref_outputs_base_dir = ref_outputs_base_dir+'/'+rt_type+'-refs-'+device+'/'
        cmd = ('python3 '+ script_name)
    msg = f'Command : {cmd} in Dir : {curr_rt_base_dir} Started'
    print(msg)

    #result = subprocess.run(cmd, cwd=curr_rt_base_dir, shell=True, stdout=subprocess.PIPE, check=True, universal_newlines=True)
    #lines = result.stdout.splitlines()
    
    lines = run_cmd(cmd=cmd, dir=curr_rt_base_dir)

    if enable_debug:
        for i in lines:
            print(i)

    rt_report = []
    golden_ref_file= ""
    if(test_config['lang'] == 'py'):
        golden_ref_file = curr_ref_outputs_base_dir+'/golden_ref.csv'
    
    elif(test_config['lang'] == 'bash'):
        golden_ref_file = curr_ref_outputs_base_dir+'/golden_ref_cpp.csv'
    with open(golden_ref_file, 'r') as f:
        ref_report = [{k:v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
    if enable_debug:
        print(ref_report)

    for i in lines:
        if i.startswith('Completed_Model : '):
            curr = i.split(',')
            tc_dict = {}
            for pair in curr:
                pair = pair.strip()
                pair = pair.split(':')
                tc_dict[pair[0].strip()] = pair[1].strip()
            rt_report.append(tc_dict)
    if enable_debug:
        print(rt_report)
    for r in ref_report:
        curr = [item for item in rt_report if item["Name"] == r['Name']]
        if enable_debug:
            print(curr)
        if len(curr) == 0:
            r['Offload Time'] = '0'
            r['Functional'] = 'FAIL'
            final_report.append(r)
        else:
            final_report.append(curr[0])
            out_file_name = os.path.join(os.path.join('./model-artifacts/'+rt_type+'/', r['Name']),final_report[-1]['Output File'])
            ref_file_name = os.path.join(os.path.join(curr_ref_outputs_base_dir, r['Name']),final_report[-1]['Output File'])
            if filecmp.cmp(out_file_name, ref_file_name) == True:
                final_report[-1]['Functional'] = 'PASS'
            else:
                final_report[-1]['Functional'] = 'FAIL'
            if platform.machine() == 'aarch64':
                final_report[-1]['Ref Total Time']   =  r['Total time']
                final_report[-1]['Ref Offload Time'] =  r['Offload Time']
                diff_in_total_time = float(final_report[-1]['Total time']) - float(final_report[-1]['Ref Total Time'])
                diff_in_total_time = (diff_in_total_time/float(final_report[-1]['Ref Total Time']))*100.0
                final_report[-1]['Diff in Total Time %']  = f'{diff_in_total_time:5.2f}'
                if(diff_in_total_time > 2.0):
                    final_report[-1]['Performance Status']  = "FAIL"
                else :
                    final_report[-1]['Performance Status']  = "PASS"

                
        final_report[-1]['Completed_Model'] = currIdx
        final_report[-1]['rt type'] = rt_type
        currIdx+= 1

print(final_report)
keys =final_report[0].keys()
with open('test_report_'+device+'.csv', 'w', newline='')  as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(final_report)