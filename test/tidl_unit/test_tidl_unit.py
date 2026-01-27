import os
import sys
import re
import pytest
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from typing import List, Dict, Tuple, Any, Union
import yaml
import platform

# Add paths to import basic_example.py
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../runtimes/examples/python/basic_example')))

from unit_test_utils import *
from basic_example import run, parse_and_validate_config

'''
Pytest file for TIDL Unit tests
Available test suites:
- test_tidl_unit: Run tests for models defined in config files
'''

import logging
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

@pytest.fixture(scope="session")
def configs(pytestconfig):
    return pytestconfig.getoption("configs")

@pytest.fixture(scope="session")
def models(pytestconfig):
    return pytestconfig.getoption("models")

@pytest.fixture(scope="session")
def artifacts_dir(pytestconfig):
    return pytestconfig.getoption("artifacts_dir")

@pytest.fixture(scope="session")
def run_infer(pytestconfig):
    return pytestconfig.getoption("run_infer")

@pytest.fixture(scope="session")
def disable_tidl_offload(pytestconfig):
    return pytestconfig.getoption("disable_tidl_offload")

@pytest.fixture(scope="session")
def force_runtime(pytestconfig):
    return pytestconfig.getoption("force_runtime")

@pytest.fixture(scope="session")
def options(pytestconfig):
     return pytestconfig.getoption("options")

@pytest.fixture(scope="session")
def nmse_threshold(pytestconfig):
    return pytestconfig.getoption("nmse_threshold")

@pytest.fixture(scope="session")
def expected_fails(pytestconfig):
     return pytestconfig.getoption("expected_fails")

@pytest.fixture(scope="session")
def disable_plot(pytestconfig):
    return pytestconfig.getoption("disable_plot")

@pytest.fixture(scope="session")
def no_subprocess(pytestconfig):
    return pytestconfig.getoption("no_subprocess")

@pytest.fixture(scope="session")
def exit_on_critical_error(pytestconfig):
    return pytestconfig.getoption("exit_on_critical_error")

@pytest.fixture(scope="session")
def timeout(pytestconfig):
    return pytestconfig.getoption("timeout")

@pytest.fixture(scope="session")
def reports_dir(pytestconfig):
    return pytestconfig.getoption("reports_dir")

def get_models_from_configs(config_files: List[str], model_filters: List[str] = None) -> List[str]:
    """
    Extract model names from config files with optional filtering.
    
    Args:
        config_files: List of config file paths
        model_filters: Optional list of model names to filter
        
    Returns:
        List[str]: List of model names
    """
    model_names = []
    
    # If no config files provided, return empty list
    if not config_files:
        return model_names
    
    # Ensure model_filters is a list
    if model_filters is None:
        model_filters = []
    
    # Process each config file
    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"[WARNING] Config file {config_file} not found, skipping")
            continue
            
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            if 'models' not in config:
                print(f"[WARNING] Config file {config_file} does not have any models defined, skipping")
                continue
                
            # Add models to the list
            for model_name in config['models'].keys():
                # Skip if model doesn't match filter
                if model_filters and model_name not in model_filters:
                    continue
                    
                model_names.append(model_name)
                
        except Exception as e:
            print(f"[ERROR] Failed to parse config file {config_file}: {e}")
    
    return model_names

def pytest_generate_tests(metafunc):
    """
    Dynamically generate tests based on models in config files.
    """
    if 'model_name' in metafunc.fixturenames:
        # Get config files and model filter
        config_files = metafunc.config.getoption("configs")
        model_filter = metafunc.config.getoption("models")
        expected_fails = metafunc.config.getoption("expected_fails")
        
        # Get models from config files
        model_names = get_models_from_configs(config_files, model_filter)
        
        # If no models found, skip tests
        if not model_names:
            pytest.skip("No models found in config files")
        
        # Mark expected failures
        if expected_fails:
            model_params = []
            for model in model_names:
                if model in expected_fails:
                    model_params.append(pytest.param(model, marks=pytest.mark.xfail))
                else:
                    model_params.append(model)
            metafunc.parametrize("model_name", model_params)
        else:
            metafunc.parametrize("model_name", model_names)

def test_tidl_unit(model_name: str,
                  configs: List[str],
                  models: List[str],
                  artifacts_dir: str,
                  run_infer: bool,
                  disable_tidl_offload: bool,
                  force_runtime: str,
                  options: List[List[str]],
                  nmse_threshold: float,
                  disable_plot: bool,
                  no_subprocess: bool,
                  exit_on_critical_error: bool,
                  timeout: int):
    """
    Test function that runs tests for models defined in config files using basic_example.py's run() function.
    
    Args:
        model_name: Name of the model to test
        configs: List of config file paths
        models: List of model names to filter
        artifacts_dir: Directory to store/use compiled models artifacts
        run_infer: Whether to run inference
        disable_tidl_offload: Whether to disable TIDL offload
        force_runtime: Runtime to use (overrides config)
        options: Additional options
        nmse_threshold: NMSE threshold for inference testing
        disable_plot: Whether to disable output plots
        no_subprocess: Whether to disable running as subprocess
        exit_on_critical_error: Whether to exit on critical error
        timeout: Timeout for test
    """
    # Force no_subprocess=True on aarch64 platform, also prevent model compilation on SoC
    if platform.machine() == 'aarch64':
        print("[INFO] Running on aarch64 platform. Disabling subprocess and setting num_threads=1.")
        no_subprocess = True

        if not disable_tidl_offload and not run_infer:
            pytest.skip("Model compilation is not supported on SoC. Please compile models on x86 and transfer/mount artifacts to SoC.")

    # Check if SOC environment variable is set
    soc = os.environ.get('SOC')
    if not soc:
        pytest.fail("SOC environment variable not set")
    soc = soc.upper()

    # Set up TIDL tools path
    tidl_tools_path = os.environ.get('TIDL_TOOLS_PATH')
    if not disable_tidl_offload and not run_infer and tidl_tools_path is None:
        pytest.fail("TIDL_TOOLS_PATH not set but TIDL offload is enabled")
    
    # Print test information
    print(f"\nRunning test for model: {model_name}")
    print(f"Config file(s): {configs}")
    print(f"Runtime: {force_runtime if force_runtime else 'Using from config'}")
    print(f"Mode: {'Inference' if run_infer else 'Compilation'}")
    print(f"TIDL Offload: {'Disabled' if disable_tidl_offload else 'Enabled'}")

    # Convert options list to a dictionary
    options = [i for sublist in options for i in sublist]
    options_dict = {}
    for option in options:
        key_val = option.strip().split('=', 1)
        if len(key_val) == 2:
            val = key_val[1]
            try:
                val = float(val)
                if val.is_integer():
                    val = int(val)
            except:
                pass
            options_dict[key_val[0]] = val

    # Run the test using basic_example.py's run() function
    if no_subprocess:
        perform_test_oneprocess(
            model_name=model_name,
            config_files=configs,
            soc=soc,
            run_infer=run_infer,
            disable_tidl_offload=disable_tidl_offload,
            force_runtime=force_runtime,
            tidl_tools_path=tidl_tools_path,
            artifacts_base_path=artifacts_dir,
            options=options_dict,
            nmse_threshold=nmse_threshold,
            disable_plot=disable_plot,
            timeout=timeout
        )
    else:
        perform_test_subprocess(
            model_name=model_name,
            config_files=configs,
            soc=soc,
            run_infer=run_infer,
            disable_tidl_offload=disable_tidl_offload,
            force_runtime=force_runtime,
            tidl_tools_path=tidl_tools_path,
            artifacts_base_path=artifacts_dir,
            options=options_dict,
            nmse_threshold=nmse_threshold,
            disable_plot=disable_plot,
            timeout=timeout,
            exit_on_critical_error=exit_on_critical_error
        )

def perform_test_subprocess(**kwargs):
    """
    Perform a TIDL unit test using a subprocess (to properly capture output for fatal errors).
    """
    # Set default temp buffer directory
    temp_buffer_dir = "/dev/shm"
    
    # Check if a custom temp buffer directory is specified in options
    options = kwargs.get('options', {})
    if "advanced_options:temp_buffer_dir" in options:
        temp_buffer_dir = options["advanced_options:temp_buffer_dir"]

    p = Process(target=perform_test_oneprocess, kwargs=kwargs)
    p.start()
    
    timeout = kwargs.get('timeout', 300)
    p.join(timeout=timeout*0.90)
    if p.is_alive():
        p.terminate()
        print("PROCESS TIMED OUT")
        for f in glob.glob(f"{temp_buffer_dir}/vashm_buff_*"):
            os.remove(f)

    assert p.exitcode == 0, f"Received nonzero exit code: {p.exitcode}"

def perform_test_oneprocess(**kwargs):
    """
    Perform a TIDL unit test using basic_example.py's run() function.
    """
    model_name = kwargs['model_name']
    config_files = kwargs['config_files']
    soc = kwargs['soc']
    run_infer = kwargs['run_infer']
    disable_tidl_offload = kwargs['disable_tidl_offload']
    force_runtime = kwargs['force_runtime']
    tidl_tools_path = kwargs['tidl_tools_path']
    artifacts_base_path = kwargs['artifacts_base_path']
    options_dict = kwargs.get('options', {})
    nmse_threshold = kwargs.get('nmse_threshold', -1)
    disable_plot = kwargs.get('disable_plot', False)
    
    # Determine which config file contains the model
    config_file = None
    config_data = None
    model_info = None
    validated_config = None
    
    for cf in config_files:
        try:
            with open(cf, 'r') as f:
                config = yaml.safe_load(f)
                if 'models' in config and model_name in config['models']:
                    config_file = cf
                    
                    validated_config = parse_and_validate_config(
                        file=cf,
                        soc=soc,
                        model_filter=(model_name,),
                        runtime_filter=None
                    )
                    
                    if validated_config is not None:
                        config_data = validated_config
                        model_info = validated_config['models'][model_name]
                        break
                    else:
                        print(f"[WARNING] Failed to validate config file {cf} for model {model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to parse config file {cf}: {e}")
    
    if config_file is None or validated_config is None:
        pytest.fail(f"Model {model_name} not found in any of the provided config files or validation failed")
    
    try:
        # Create a new config with only the model under test
        new_config = {
            'models': {
                model_name: config_data['models'][model_name]
            }
        }

        # Check for multiple inputs and outputs and warn the user
        model_config = new_config['models'][model_name]
        if 'inputs' in model_config:
            inputs = model_config['inputs']
            if isinstance(inputs, list) and len(inputs) > 1:
                print(f"[WARNING] Model {model_name} has multiple inputs defined. Only the first input will be used for testing.")
                model_config['inputs'] = inputs[0]
        
        if 'expected_outputs' in model_config:
            expected_outputs = model_config['expected_outputs']
            if isinstance(expected_outputs, list) and len(expected_outputs) > 1:
                print(f"[WARNING] Model {model_name} has multiple expected outputs defined. Only the first output will be used for testing.")
                model_config['expected_outputs'] = expected_outputs[0]
            if not os.path.isabs(model_config['expected_outputs']):
                model_config['expected_outputs'] = os.path.abspath(os.path.join(os.path.dirname(config_file), model_config['expected_outputs']))

        # Copy any global settings from the original config
        for key, value in config_data.items():
            if key != 'models':
                new_config[key] = value
        
        # Apply any modifications to the config data
        if options_dict:
            # Ensure the model has compile_options and infer_options sections
            model_config = new_config['models'][model_name]
            
            # Add compile options
            if 'compile_options' not in model_config:
                model_config['compile_options'] = {}
            
            # Add infer options
            if 'infer_options' not in model_config:
                model_config['infer_options'] = {}
            
            # Apply options to both compile and infer options
            for key, value in options_dict.items():
                model_config['compile_options'][key] = value
                model_config['infer_options'][key] = value
        
        # If force_runtime is provided, overwrite the runtime in the config
        if force_runtime:
            model_config = new_config['models'][model_name]
            model_path = model_config['path']
            
            # Check if the forced runtime is compatible with the model format
            if model_path.endswith('.onnx'):
                if force_runtime not in ['onnxrt', 'tidlrt', 'tvmrt']:
                    pytest.fail(f"Invalid runtime '{force_runtime}' for ONNX model. Only 'onnxrt', 'tidlrt', or 'tvmrt' are supported for ONNX models.")
            elif model_path.endswith('.tflite'):
                if force_runtime != 'tflitert':
                    pytest.fail(f"Invalid runtime '{force_runtime}' for TFLite model. Only 'tflitert' is supported for TFLite models.")
            else:
                print(f"[WARNING] Unrecognized model format for {model_path}")
            
            model_config['runtime'] = force_runtime
            print(f"[INFO] Overwriting runtime with forced runtime: {force_runtime}")

        # Print runtime information for pytest_runtest_makereport to parse
        runtime = new_config['models'][model_name]['runtime']
        print(f"\nRUNTIME: {runtime}")
        
        # Set up artifacts directory
        if not artifacts_base_path:
            tensor_bits = 8
            if 'compile_options' in new_config:
                tensor_bits = new_config['compile_options'].get('tensor_bits', tensor_bits)
            if 'compile_options' in model_config:
                tensor_bits = model_config['compile_options'].get('tensor_bits', tensor_bits)
            artifacts_base_path = os.path.join(os.path.dirname(__file__), "model-artifacts", f'{soc}', f'{runtime}', f'{tensor_bits}bits')
        artifacts_parent_path = os.path.join(artifacts_base_path, model_name)
        artifacts_path = os.path.join(artifacts_parent_path, "artifacts")
        
        # Clean up artifacts directory for compilation
        if not run_infer and not disable_tidl_offload:
            try:
                shutil.rmtree(artifacts_parent_path, ignore_errors=True)
            except Exception as e:
                print(f"[WARNING] Failed to clean up artifacts directory: {e}")

        # Run the model using basic_example.py's run() function with the modified config
        status, outputs = run(
            config=new_config,
            soc=soc,
            compile=not run_infer,
            tidl_tools_path=tidl_tools_path,
            artifacts_base_path=artifacts_base_path,
            disable_tidl_offload=disable_tidl_offload,
            verbose=True
        )
        
        # Clean up temporary directory in artifacts folder
        if not run_infer and not disable_tidl_offload:
            '''
            Can't remove tempDir in TVMRT since "tempDir" is used to determine
            offload during inference for TVMRT in basic examples. This
            condition can be removed when this dependency on tempDir is removed
            for TVMRT
            '''
            if runtime != "tvmrt":
                try:
                    temp_dir_path = os.path.join(artifacts_path, "tempDir")
                    if os.path.exists(temp_dir_path):
                        shutil.rmtree(temp_dir_path)
                except Exception as e:
                    print(f"[WARNING] Failed to clean up temporary directory: {e}")
        
        assert status == 0, f"Error running model {model_name}"
        if new_config['models'][model_name]['runtime'] != "tvmrt" or run_infer:
            assert outputs is not None, f"No outputs generated for model {model_name}"
            assert model_name in outputs, f"No outputs found for model {model_name}"
            assert len(outputs[model_name]) > 0, f"Empty outputs for model {model_name}"

        # For inference mode, check outputs and validate against reference
        if run_infer:
            binary_outputs = outputs[model_name][0][0]
            post_processed_outputs = outputs[model_name][0][1]

            use_binary_outputs = post_processed_outputs is None or len(post_processed_outputs) < 1

            ref_binary_outputs = {}
            ref_post_processed_outputs = None
            if "expected_outputs" in model_info:
                expected_output_file = model_info["expected_outputs"]
                if use_binary_outputs:
                    if os.path.isfile(expected_output_file) and expected_output_file.endswith('.npz'):
                        ref_binary_outputs = np.load(expected_output_file)
                    else:
                        print(f"[WARN] Expected output file {expected_output_file} not found or not an npz file.")

            # Generate reference outputs on the fly without TIDL offload
            if len(ref_binary_outputs) == 0:
                print()
                if use_binary_outputs:
                    print("[WARN] Expected outputs not defined in config file.")
                print("[INFO] Generating reference outputs on the fly without TIDL offload...")

                # For TVM or TIDLRT use ONNXRT/TFLITERT to generate golden reference
                if new_config['models'][model_name]['runtime'] == "tvmrt" or new_config['models'][model_name]['runtime'] == "tidlrt":
                    if new_config['models'][model_name]['path'].endswith('.onnx'):
                        new_config['models'][model_name]['runtime'] = "onnxrt"
                    else:
                        new_config['models'][model_name]['runtime'] = "tflitert"
      
                _, reference_outputs = run(
                    config=new_config,
                    soc=soc,
                    compile=False,
                    tidl_tools_path=tidl_tools_path,
                    artifacts_base_path=artifacts_base_path,
                    disable_tidl_offload=True,
                    verbose=False
                )
                ref_binary_outputs = reference_outputs[model_name][0][0]
                ref_post_processed_outputs = reference_outputs[model_name][0][1]

            '''
            If post processing is not done use nmse based method
            '''
            if use_binary_outputs:

                print("[INFO] Using binary outputs for evaluation.")

                
                if nmse_threshold < 0:
                    if 'extra_options' in config_data:
                        nmse_threshold = config_data['extra_options'].get("nmse_threshold", nmse_threshold)
                    nmse_threshold = model_info.get("nmse_threshold", nmse_threshold)
                # Set default NMSE Threshold value in case of negative
                if nmse_threshold < 0:
                    nmse_threshold = 0.5
                print(f"NMSE Threshold: {nmse_threshold}\n")

                # Compute output metrics
                max_delta = float('-inf')
                max_nmse = float('-inf')
                max_mse = float('-inf')
                epsilon = 1e-10
                
                eval_results = {
                    'nmse': [],
                    'mse': [],
                    'delta': [],
                    'expected_outputs': [],
                    'outputs': []
                }

                for out_name, output in binary_outputs.items():
                    expected_output = ref_binary_outputs.get(out_name)
                    assert expected_output is not None, f" No expected output for output named {out_name}"
        
                    output = np.squeeze(output.astype(float))
                    expected_output = np.squeeze(expected_output.astype(float))

                    assert expected_output.shape == output.shape, f" Shape mismatch! Expected {expected_output.shape} got {output.shape}"
                            
                    curr_mse = np.mean((expected_output - output)**2)
                    curr_var = np.var(expected_output)
                    
                    if curr_var < epsilon:
                        curr_nmse = None
                    else:
                        curr_nmse = curr_mse / curr_var
                    
                    if max_mse is None or curr_mse is None or np.isnan(curr_mse):
                        max_mse = None
                    elif curr_mse > max_mse:
                        max_mse = curr_mse
                    
                    if max_nmse is None or curr_nmse is None or np.isnan(curr_nmse):
                        max_nmse = None
                    elif curr_nmse > max_nmse:
                        max_nmse = curr_nmse
                    
                    delta = np.abs(expected_output - output)
                    curr_max_delta = np.max(delta)
                    
                    if max_delta is None or curr_max_delta is None or np.isnan(curr_max_delta):
                        max_delta = None
                    elif curr_max_delta > max_delta:
                        max_delta = curr_max_delta
                    
                    eval_results['nmse'].append(curr_nmse)
                    eval_results['mse'].append(curr_mse)
                    eval_results['delta'].append(curr_max_delta)
                    eval_results['expected_outputs'].append(expected_output)
                    eval_results['outputs'].append(output)
                    
                    '''
                    For Internal Purpose:
                    For TopK we only consider 1st output (Values) and not
                    the 2nd output (Indices) because of internal implementation
                    of TopK output buffer datatype for indices in TIDL
                    '''
                    if model_name.startswith("TopK"):
                        break
                        
                # Print metrics
                if max_nmse is None:
                    print("MAX_NMSE: None")
                else:
                    print(f"MAX_NMSE: {max_nmse:.7f}")
                
                if max_mse is None:
                    print("MAX_MSE: None")
                else:
                    print(f"MAX_MSE: {max_mse:.7f}")
                
                if max_delta is None:
                    print("MAX_DELTA: None")
                else:
                    print(f"MAX_DELTA: {max_delta:.7f}")
                
                # Generate plots binary
                try:
                    if not disable_plot and len(eval_results['outputs']) > 0:
                        plots_dir = os.path.join(artifacts_parent_path, 'plots')
                        os.makedirs(plots_dir, exist_ok=True)
                        _, plot_base64_path = generate_plot(
                            binary_results=eval_results,
                            image_results=None,
                            output_dir=plots_dir, 
                            plot_name=model_name, 
                            save_image=False
                        )
                        print(f"\nPLOT_BASE_64_PATH: {plot_base64_path}")
                    print()
                except Exception as e:
                    print(f"[WARN] Cannot generate output plot : {e}")

                if max_nmse is None and max_mse is None:
                    pytest.fail("Could not calculate NMSE")
                elif max_nmse is not None:
                    if max_nmse > nmse_threshold:
                        pytest.fail(f"max_nmse of {max_nmse} is higher than threshold {nmse_threshold}")
                elif max_mse is not None:
                    if max_mse > nmse_threshold:
                        pytest.fail(f"max_mse of {max_mse} is higher than threshold {nmse_threshold}")

            else:
                print("[INFO] Post-processed images found. Using metadata for specialized evaluation.")
                
                status = 0
                reason = ""
                task_type = None
                if 'post_process_info' in model_info and 'task_type' in model_info['post_process_info']:
                    task_type = model_info['post_process_info']['task_type']
                
                assert task_type is not None, f" No 'task_type' defined in 'post_process_info'"

                task_type = task_type.strip().lower()
                assert task_type in ["classification", "detection", "segmentation"], f"task_type {task_type} unrecognized. Allowed values are classification, detection or segmentation"

                if task_type == 'classification':
                    '''
                    For classification check if top n classes match and in correct order
                    Count how many classes match out of all classes compared
                    '''
                    print(f"[INFO] Evaluating classification model based on classes...")
                    top_n = 3
                    match_count = 0
                    for image_name, (metadata, _) in post_processed_outputs.items():
                        expected_output = ref_post_processed_outputs.get(image_name)
                        assert expected_output is not None, f" No expected post-processed output for {image_name}"
                        
                        # Parse classes from metadata
                        expected_metadata = expected_output[0].strip().split('\n')
                        output_metadata = metadata.strip().split('\n')
                        
                        expected_classes = []
                        output_classes = []
                        for i, c in enumerate(expected_metadata):
                            expected_classes.append(c.split('-',1)[-1].strip())
        
                        for i, c in enumerate(output_metadata):
                            output_classes.append(c.split('-',1)[-1].strip())
                        
                        class_matches = 0
                        classes_compared = 0
                        for i, (exp_class, out_class) in enumerate(zip(expected_classes, output_classes)):
                            if i < top_n and exp_class == out_class:
                                class_matches += 1
                            classes_compared += 1
                        
                        print(f"[INFO] Top {class_matches}/{classes_compared} classes match for {image_name}. Top {top_n} classes should match.")

                        if class_matches >= top_n:
                            match_count += 1
                        else:
                            status = -1
                            reason = f"Only {class_matches}/{classes_compared} top classes match for {image_name}. Top {top_n} classes should match."

                    print(f"\nPOST-PROC METRICS: {match_count}/{len(post_processed_outputs)} images match")

                elif task_type == 'segmentation':
                    '''
                    Check class ID for each pixel is same while allowing some
                    percentage of pixels to differ.
                    '''                    
                    print("[INFO] Evaluating segmentation model based on output pixel classes..")
                    allowed_deviation = 0.05     # 5% allowed deviation
                    match_count = 0
                    for image_name, (metadata, _) in post_processed_outputs.items():
                        expected_output = ref_post_processed_outputs.get(image_name)
                        assert expected_output is not None, f" No expected post-processed output for {image_name}"
                        expected_metadata = expected_output[0].strip().replace('\n',' ').split(' ')
                        output_metadata = metadata.strip().replace('\n',' ').split(' ')

                        assert len(expected_metadata) == len(output_metadata), f" Length of expected output does not match actual output"
                        
                        diff_cnt = 0
                        for i in range(len(output_metadata)):
                            if output_metadata[i] != expected_metadata[i]:
                                diff_cnt += 1

                        match_pct = (len(output_metadata) - diff_cnt) / len(output_metadata)

                        print(f"[INFO] {(match_pct * 100):.2f}% out of output matches for {image_name}. Allowed deviation {(allowed_deviation * 100):.2f}%.")

                        if match_pct < (1.0 - allowed_deviation):
                            status = -1
                            reason = f"More than {(allowed_deviation * 100):.4f}% output mismatches for {image_name}"
                        else:
                            match_count += 1
                    
                    print(f"\nPOST-PROC METRICS: {match_count}/{len(post_processed_outputs)} images match")

                elif task_type == 'detection':
                    '''
                    Check IOU for each bounding boxes.
                    '''
                    print("[INFO] Evaluating detection model based on bounding boxes IoU...")                    
                    
                    iou_threshold = 0.8
                    match_count = 0
                    for image_name, (metadata, _) in post_processed_outputs.items():

                        expected_output = ref_post_processed_outputs.get(image_name)
                        assert expected_output is not None, f" No expected post-processed output for {image_name}"
                            
                        expected_metadata = expected_output[0].strip().split('\n')
                        output_metadata = metadata.strip().split('\n')
                        
                        # Parse bounding boxes and labels from metadata
                        actual_boxes = []
                        reference_boxes = []
                        
                        for det in output_metadata:
                            parts = det.strip().split('-')
                            if len(parts) >= 2:
                                conf = float(parts[0])
                                box_str = parts[1].strip().strip('()[]')
                                box_coords = [int(x.strip()) for x in box_str.split(',')]
                                label = parts[2] if len(parts) > 2 else ""
                                actual_boxes.append((conf, box_coords, label))
                        
                        for det in expected_metadata:
                            parts = det.strip().split('-')
                            if len(parts) >= 2:
                                conf = float(parts[0])
                                box_str = parts[1].strip().strip('()[]')
                                box_coords = [int(x.strip()) for x in box_str.split(',')]
                                label = parts[2] if len(parts) > 2 else ""
                                reference_boxes.append((conf, box_coords, label))
                        
                        # Sort boxes by confidence (highest first)
                        actual_boxes.sort(key=lambda x: x[0], reverse=True)
                        reference_boxes.sort(key=lambda x: x[0], reverse=True)
                        
                        # Match detections using IoU
                        matched_refs = set()
                        matched_acts = set()
                        matches = []
                        for i, (_, act_box, act_label) in enumerate(actual_boxes):
                            best_iou = 0
                            best_match = -1
                            for j, (_, ref_box, ref_label) in enumerate(reference_boxes):
                                if j in matched_refs:
                                    continue

                                # Skip if labels don't match (when both have labels)
                                if act_label and ref_label and act_label != ref_label:
                                    continue
                                    
                                iou = calculate_iou(act_box, ref_box)                                
                                if iou > best_iou:
                                    best_iou = iou
                                    best_match = j
                            
                            # If we found a match above threshold
                            if best_match >= 0:
                                matched_refs.add(best_match)
                                matched_acts.add(i)
                                matches.append((i, best_match, best_iou))

                        true_positives = len(matches)
                        false_positives = len(actual_boxes) - true_positives
                        false_negatives = len(reference_boxes) - true_positives

                        mismatch = False
                        if true_positives <= 0:
                            mismatch = True
                            reason = f"No matching detections found for {image_name}"
                        
                        elif false_positives > 0:
                            mismatch = True
                            reason = f"{false_positives} box does not match for reference and actual output in {image_name}"
                        
                        elif false_negatives > 0:
                            mismatch = True
                            reason = f"{false_negatives} box does not match for reference and actual output in {image_name}"

                        for match in matches:
                            if match[2] < iou_threshold:
                                mismatch = True
                                reason = f"IoU ({match[2]:.4f}) < Threshold IoU ({iou_threshold}) for box {match[0]} in {image_name}"
                                break

                        if mismatch == True:
                            status = -1
                        else:
                            match_count += 1
                        
                        print(f"[INFO] All {true_positives} detection have IoU more that threshold({iou_threshold}) for {image_name}.")

                    print(f"\nPOST-PROC METRICS: {match_count}/{len(post_processed_outputs)} images match")
        
                # Generate plots for post-processed images
                try:
                    if not disable_plot:
                        plots_dir = os.path.join(artifacts_parent_path, 'plots')
                        os.makedirs(plots_dir, exist_ok=True)
                        
                        image_results = {
                            'outputs': post_processed_outputs,
                            'expected_outputs': ref_post_processed_outputs
                        }
                        
                        _, plot_base64_path = generate_plot(
                            binary_results=None,
                            image_results=image_results,
                            output_dir=plots_dir,
                            plot_name=model_name,
                            save_image=False
                        )
                        print(f"\nPLOT_BASE_64_PATH: {plot_base64_path}")
                except Exception as e:
                    print(f"[WARN] Cannot generate output plot : {e}")
                
                # Check if evaluation failed
                if status != 0:
                    pytest.fail(reason)
                    
    except Exception as e:
        pytest.fail(f"Error running model {model_name}: {str(e)}")
