# OSRT Troubleshooting Guide


Below are some of the troubleshooting guidelines for OSRT model compilation and inference. This section is common across the all the OSRT (TFlite / ONNX runtime /TVM-DLR )

* Set debug_level > 0 to enable verbose debug log during import and inference
* The import process generates visualization of the model in artifacts folder. This can be used to understand the number sub graphs offloaded to C7x-MMA, layers in each subgraph and layers processed on ARM processor
* If there are any issues observed during model compilation process, then enable ARM only mode by passing “-d” as an argument to the default model compilation script 
As an example
    ```
    python3 onnxrt_ep.py -d
    ```
* If the model works fine with ARM only mode but fails to compile with C7x-MMA offload, try dispatching some of the layers (less commonly used layer type) to ARM by using “deny_list” option
* If accuracy of the model is observed lower than expected, try to import the model with 16-bit inference. This can be enabled by setting tensor_bits=16. If the 16-bit mode works fine, then fine tune the advanced_options:calibration to work with 8-bit mode. If the 16-bit mode also fails to meet accuracy requirement, then try with “deny_list” option as described in previous step.
