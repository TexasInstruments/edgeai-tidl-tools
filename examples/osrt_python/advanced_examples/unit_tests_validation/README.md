# Advanced examples : Unit test validation scripts

## Introduction 

The main purpose of these scripts is to validate TIDL 32 bit floating point reference implementation with ARM implementation for floating point models.
These scripts can be used for any generic models but also contains scripts to generate unit test cases for specific layers - generate unit test case, get ARM reference output (output_ref), do model compilation for TIDL and corresponding inference (output_test) followed by comparison script for output_ref with output_test.

**Disclaimer** : These scripts are mainly written/maintained for debug/development validation and may not receive active support in case of issues. Also all example unit test cases may not be necessarily supported on TIDL

## Usage

The usage is specified for tfite runtime, follow corresponding steps for ONNX runtime

1. Perform the setup steps are required for setting up the python examples

2. Install additional requirements :
```
cd examples/osrt_python/advanced_examples/unit_tests_validation
pip3 install -r ./requirements.txt
```

3. Create required directories
```
mkdir unit_test_models
mkdir outputs/output_ref/tflite/  ## outputs/output_ref/onnx for onnx models
mkdir outputs/output_test/tflite/  ## outputs/output_test/onnx for onnx models
```

4. Generate test case for required layer, update following script as required
```
cd scripts
python3 generate_unit_test_models_tflite.py
```

5. To run a new model, add the model config as part of models_configs in common_utils.py (refer example configs there), add the model name in "models' list in tflrt_delegate.py and add the same name as part of 'models' list in evaluate_unit_tests.py.

6. Run ARM only reference flow
```
cd tfl
python3 tflrt_delegate.py -d
```
Outputs are saved in outputs/output_ref/tflite/

7. Run TIDL compilation followed by inference
```
python3 tflrt_delegate.py -c
python3 tflrt_delegate.py
```
Outputs are saved in outputs/output_test/tflite/

8. Run evaluation script to find the maximum difference in elements on performing element wise comparison of ref and test outputs
Output format : {model_name_1 : {output_i : max_diff_i}}
```
python3 evaluate_unit_tests.py
```

**Note** : These scripts are meant to be used only on PC and not on target board