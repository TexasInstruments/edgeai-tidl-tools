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

enable_debug = False

ref_outputs_base_dir = 'test_data' 
rt_base_dir_py = 'examples/osrt_python/'
rt_base_dir_bash = 'scripts'

try:
    SOC = os.environ['SOC']
except:
    print('SOC env variable not found')
    exit(-1)

global test_configs
if SOC == "am62":
    device = 'am62'
    test_configs = [
                    {'script_name':'tflrt_delegate.py', 'script_dir':'tfl','lang':'py', 'rt_type':'tfl-py'},
                    {'script_name':'onnxrt_ep.py', 'script_dir':'ort','lang':'py', 'rt_type':'ort-py'},
                    #{'script_name':'dlr_inference_example.py', 'script_dir':'tvm_dlr','lang':'py', 'rt_type':'dlr-py'},
                    {'script_name':'run_tfl_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'tfl-cpp'},
                    {'script_name':'run_onnx_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'ort-cpp'},
                    #{'script_name':'run_dlr_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'dlr-cpp'},
        ]
elif SOC == "am68pa" :
    device = 'am68pa'
    test_configs = [
                    {'script_name':'tflrt_delegate.py', 'script_dir':'tfl','lang':'py', 'rt_type':'tfl-py'},
                    {'script_name':'onnxrt_ep.py', 'script_dir':'ort','lang':'py', 'rt_type':'ort-py'},
                    {'script_name':'dlr_inference_example.py', 'script_dir':'tvm_dlr','lang':'py', 'rt_type':'dlr-py'},
                    {'script_name':'run_tfl_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'tfl-cpp'},
                    {'script_name':'run_onnx_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'ort-cpp'},
                    {'script_name':'run_dlr_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'dlr-cpp'},
        ]
elif SOC == "am68a" :
    device = 'am68a'
    test_configs = [
                    {'script_name':'tflrt_delegate.py', 'script_dir':'tfl','lang':'py', 'rt_type':'tfl-py'},
                    {'script_name':'onnxrt_ep.py', 'script_dir':'ort','lang':'py', 'rt_type':'ort-py'},
                    {'script_name':'dlr_inference_example.py', 'script_dir':'tvm_dlr','lang':'py', 'rt_type':'dlr-py'},
                    {'script_name':'run_tfl_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'tfl-cpp'},
                    {'script_name':'run_onnx_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'ort-cpp'},
                    {'script_name':'run_dlr_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'dlr-cpp'},
        ]
elif SOC == "am69a" :
    device = 'am69a'
    test_configs = [
                    {'script_name':'tflrt_delegate.py', 'script_dir':'tfl','lang':'py', 'rt_type':'tfl-py'},
                    {'script_name':'onnxrt_ep.py', 'script_dir':'ort','lang':'py', 'rt_type':'ort-py'},
                    {'script_name':'dlr_inference_example.py', 'script_dir':'tvm_dlr','lang':'py', 'rt_type':'dlr-py'},
                    {'script_name':'run_tfl_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'tfl-cpp'},
                    {'script_name':'run_onnx_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'ort-cpp'},
                    {'script_name':'run_dlr_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'dlr-cpp'},
        ]                 
elif SOC == "am62a" :
    device = 'am62a'
    test_configs = [
                    {'script_name':'tflrt_delegate.py', 'script_dir':'tfl','lang':'py', 'rt_type':'tfl-py'},
                    {'script_name':'onnxrt_ep.py', 'script_dir':'ort','lang':'py', 'rt_type':'ort-py'},
                    {'script_name':'dlr_inference_example.py', 'script_dir':'tvm_dlr','lang':'py', 'rt_type':'dlr-py'},
                    {'script_name':'run_tfl_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'tfl-cpp'},
                    {'script_name':'run_onnx_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'ort-cpp'},
                    {'script_name':'run_dlr_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'dlr-cpp'},
        ]        
elif SOC == "am67a" :
    device = 'am67a'
    test_configs = [
                    {'script_name':'tflrt_delegate.py', 'script_dir':'tfl','lang':'py', 'rt_type':'tfl-py'},
                    {'script_name':'onnxrt_ep.py', 'script_dir':'ort','lang':'py', 'rt_type':'ort-py'},
                    #{'script_name':'dlr_inference_example.py', 'script_dir':'tvm_dlr','lang':'py', 'rt_type':'dlr-py'},
                    {'script_name':'run_tfl_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'tfl-cpp'},
                    {'script_name':'run_onnx_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'ort-cpp'},
                    #{'script_name':'run_dlr_models.sh', 'script_dir':'osrt_cpp_scripts/','lang':'bash','rt_type':'dlr-cpp'},
        ]
else:
    print( "Set SOC variable in your shell")
    exit(-1)

currIdx = 0
if platform.machine() != 'aarch64':
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

num_func_pass = 0
num_func_fail = 0
num_perf_pass = 0
num_perf_fail = 0

for test_config in test_configs:
    script_name = test_config['script_name']
    rt_type = test_config['rt_type']
    if(test_config['lang'] == 'bash'):
        rt_base_dir = rt_base_dir_bash
        curr_rt_base_dir= os.path.join(rt_base_dir,test_config['script_dir'])
        cmd = ('bash '+ script_name)

    elif(test_config['lang'] == 'py'):
        rt_base_dir = rt_base_dir_py
        curr_rt_base_dir= os.path.join(rt_base_dir,test_config['script_dir'])
        cmd = ('python3 '+ script_name)

    if device != 'pc':
        curr_ref_base_dir = ref_outputs_base_dir+'/refs-'+device+'/'
    else:
        curr_ref_base_dir = ref_outputs_base_dir+'/refs-'+device+'-'+SOC+'/'
    
    curr_ref_output_base_dir = os.path.join(curr_ref_base_dir,"bin")

    lines = run_cmd(cmd=cmd, dir=curr_rt_base_dir)

    if enable_debug:
        for i in lines:
            print(i)

    rt_report = []
    golden_ref_file= ""
    if device != 'pc':
        golden_ref_file = curr_ref_base_dir+'/golden_ref_'+device+'.csv'
    else:
        golden_ref_file = curr_ref_base_dir+'/golden_ref_'+device+'_'+SOC+'.csv'
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
        curr = [item for item in rt_report if ((item["Name"] == r['Name']) and (rt_type == r['rt type']) )]
        if enable_debug:
            print(curr)
        if (len(curr) == 0 and rt_type == r['rt type']):
            r['Offload Time'] = '0'
            r['Functional Status'] = 'FAIL'
            r['Info'] = 'Output Not Detected'
            final_report.append(r)
            final_report[-1]['Sl No.'] = currIdx
            final_report[-1]['Runtime'] = rt_type
            currIdx+= 1
            num_func_fail += 1
        elif(len(curr) != 0 and rt_type == r['rt type']):
            final_report.append(curr[0])
            out_file_name = os.path.join('./output_binaries',final_report[-1]['Output Bin File'])
            ref_file_name = os.path.join(curr_ref_output_base_dir, final_report[-1]['Output Bin File'])
            if not os.path.exists(out_file_name) or not os.path.exists(ref_file_name):
                if not os.path.exists(out_file_name):
                    final_report[-1]['Functional Status'] = 'FAIL'
                    final_report[-1]['Info'] = 'Output Bin File Not Found'
                elif not os.path.exists(ref_file_name):
                    final_report[-1]['Functional Status'] = 'FAIL'
                    final_report[-1]['Info'] = 'Output Ref File Not Found'
                num_func_fail += 1
            else:
                if filecmp.cmp(out_file_name, ref_file_name) == True:
                    final_report[-1]['Functional Status'] = 'PASS'
                    final_report[-1]['Info'] = ''
                    num_func_pass += 1
                else:
                    final_report[-1]['Functional Status'] = 'FAIL'
                    final_report[-1]['Info'] = 'Output Bin File Mismatch'
                    num_func_fail += 1

            if platform.machine() == 'aarch64':
                final_report[-1]['Ref Total Time']   =  r['Total time']
                final_report[-1]['Ref Offload Time'] =  r['Offload Time']

                diff_in_total_time = float(final_report[-1]['Total time']) - float(final_report[-1]['Ref Total Time'])
                diff_in_total_time_pct = (diff_in_total_time/float(final_report[-1]['Ref Total Time']))*100.0

                final_report[-1]['Total Time Diff']  = f'{diff_in_total_time:5.2f}'
                final_report[-1]['Total Time Diff(%)']  = f'{diff_in_total_time_pct:5.2f}%'
    
                if(diff_in_total_time_pct > 2.0 and diff_in_total_time > 0.02):
                    final_report[-1]['Performance Status']  = "FAIL"
                    final_report[-1]['Info'] ="Actual time more than Ref time by 0.02ms or 2%"
                    num_perf_fail += 1
                else :
                    final_report[-1]['Performance Status']  = "PASS"
                    num_perf_pass += 1
            final_report[-1]['Sl No.'] = currIdx
            final_report[-1]['Runtime'] = rt_type
            currIdx+= 1

if(len(final_report) > 0):
    if device == 'pc':
        for i in range(len(final_report)):
            final_report[i].pop('Total time',None)
            final_report[i].pop('Offload Time',None)
            final_report[i].pop('DDR RW MBs',None)

    for i in range(len(final_report)):
        if 'Output Bin File' not in final_report[i]:
            final_report[i]['Output Bin File'] = ''
        if 'Output Image File' not in final_report[i]:
            final_report[i]['Output Image File'] = ''
    
    # Sort the dictionary
    sequence = ["Sl No.","Runtime","Name","Output Image File","Output Bin File",
                "Total time","Offload Time","Ref Total Time","Ref Offload Time",
                "Total Time Diff","Total Time Diff(%)","DDR RW MBs", "Functional Status",
                "Performance Status","Info"]
    for i in range(len(final_report)):
        temp_dict = {}
        for p in sequence:
            found = False
            for q in final_report[i]:
                if p.strip() == q.strip():
                    key = q
                    found = True
                    break
            if (found):
                temp_dict[key] = final_report[i][key]
        final_report[i] = temp_dict

    keys =final_report[0].keys()
    if device != 'pc':
        report_file = 'test_report_'+device+'.csv'
    else:
        report_file = 'test_report_'+device+'_'+ SOC+ '.csv'
    with open(report_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(final_report)
else:
    print("No test_report is generated.")

print(final_report)
print("\nFunc Pass: {}\nFunc Fail: {}".format(num_func_pass, num_func_fail))
if platform.machine() == 'aarch64':
    print("\nPerf Pass: {}\nPerf Fail: {}".format(num_perf_pass, num_perf_fail))

print("\nPlease refer to the output_images and output_binaries directory for generated outputs")
print("TEST DONE!")