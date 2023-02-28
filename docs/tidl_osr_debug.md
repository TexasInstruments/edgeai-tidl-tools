# OSRT Troubleshooting Guide


Below are some of the troubleshooting guidelines for OSRT model compilation and inference. This section is common across the all the OSRT (TFlite / ONNX runtime /TVM-DLR )

* Set debug_level > 0 to enable verbose debug log during import and inference
* If there are any issues observed during model compilation process, then enable ARM only mode by passing “-d” as an argument to the default model compilation script 
As an example
    ```
    python3 onnxrt_ep.py -d
    ```
* If the model works fine with ARM only mode but fails to compile with C7x-MMA offload, try dispatching some of the layers (less commonly used layer type) to ARM by using “deny_list” option
* If accuracy of the model is observed lower than expected, try to import the model with 16-bit inference. This can be enabled by setting tensor_bits=16. If the 16-bit mode works fine, then fine tune the advanced_options:calibration to work with 8-bit mode. If the 16-bit mode also fails to meet accuracy requirement, then try with “deny_list” option as described in previous step.

### Graph visualization
Graph visualization is an important tool to understand subgraphs created as part of TIDL compilation process, understand layers delegated to DSP vs ARM and view the TIDL compiled subgraphs.

* The model compilation process generates visualization of the model in <artifacts_folder>/tempDir
* 2 types of visualizations can be found in this folder (1) Runtimes visualization (runtimes_visualization.svg) (2) TIDL compiled subgraph visualization (\<subgraphId>_tidl_net.bin.svg)
  * Runtimes visualization : 
    * Depicts the original model with a colored box marked around each subgraph delegated to DSP - all the nodes within same subgraph are marked with same color to distinguish from another subgraph
    * Nodes marked with Grey color are the ones not supported on TI DSP and hence delegated to ARM
    * User can hover over individual nodes to get additional info about the node. This is especially useful for ARM delegated nodes as addditional info about why node is not supported on TI-DSP is displayed
  * TIDL compiled subgraphs visualization :
    * Each of the subgraphs depicted in runtimes visualization has its own visualization generated after compilation through TIDL
    * The subgraphs are identified by name/tensor index of output nodes of the subgraph in the original model
    * TIDL imported properties of the node in each TIDL subgraph can be viewed by hovering over the nodes in each of the visualization files
