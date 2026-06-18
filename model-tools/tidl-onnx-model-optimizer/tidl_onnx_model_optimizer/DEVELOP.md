# Developer's Guideline

This document describes how you can add your own optimization function to TIDL ONNX Model Optimizer library.


## Creating a function
You have to write a function that contains the optimization you are intending to do.

### Prototype

`def tidl_abc ( graph: gs.Graph, onnx_graph: onnx.GraphProto):`

`abc` should clearly describe what the function is doing in brief. For e.g., `tidl_convert_resize_params_size_to_scale` is name of the function that changes the sizes input of a Resize layer to appropriate scales.

### Where to put your function
In the following section, changes have to be made inside `tidl_onnx_model_optimizer/src` directory.

#### Case - I
If you are writing a function which handles a layer/block which is already present, you are expected to keep the function in the .py file existing for that layer/block. For e.g., if you are adding a optimization function on resize layer, you are supposed to put `tidl_abc` function in `resize.py`.

#### Case - II
If you are creating a function for a completely new layer/block, you will have to create a new .py file, which should be named as `<layer>.py` or `<block>.py`. Put your function in this file.


## Interface
Make the following changes in the file `tidl_onnx_model_optimizer/ops.py`

### Control Flag
You will require a new flag for enable/disbale control over your optimization. For that purpose you have to change the dict returned in the function `get_optimizers` to have a new entry `'abc'`. You can make the default value here True/False as per your need.

### Dependency graph
Your function might need to be run strictly after some other existing optimization function and before some functions. For e.g., say you are converting a unoptimal MatMul to Conv, you want all the optimizations which converts other layers to MatMul to run before this (as then you don't have to run your function multiple number of times).

1. Add a new entry `'abc': []` in the dict `adj_list`.
2. For any other key, `k`, if you need your function to run before function corresposding to k, modify your entry as `'abc': [k]`. Keep adding to this list like `[k1, k2, k3, ...]` for as many functions you need.
3. If you want your function to strictly run after some other function corresponding to key `k`, modify the entry for `k` as `k: [..., 'abc']`

Please add a single line comment justfying your reason of adding a dependency, as these are good when a strict ordering of functions are necessary but costs time when there are lot of functions i.e., nodes in the dependency graph.

### Call to implemented function
Finally you need to change the dict variable `opt_ops` if you have added a new .py file.
Add a entry `"abc" : tidl_abc` and voila! You are done.


## Debug Your Added Transformation

You can use the test_optimise.py script, where you can specify the path to the original onnx model as well as modify the test_optimizers function in ops.py to specify the transformation that needs to be debugged. 


## Good Practices
For good practices ensure to use the logging library facility and add useful debug logs wherever you seem necessary 🙂.
Also please try to run pylint on the code as this codebase has been developed with pylint coding guidelines. If you are using pylint in your VS Code, you can add these settings to ensure consistency:
```
"pylint.args": [
        "--disable=E1101,W0613,W1203,E0401,W1201,W1514",
        "--max-line-length=160"
    ],
```

## Bucket Functionality

The optimizer supports grouping related optimizations into **buckets**. This allows you to enable or disable sets of optimizations together and ensures that dependencies between groups are respected.
> **Note:** Optimizations do not need to belong to a bucket to work. Any optimization can be enabled and executed independently, even if it is not part of any bucket.

### How to Add or Modify Buckets

1. **Define a Bucket**  
   In [`ops.py`](ops.py), add your bucket to the `BUCKETS` dictionary:
   ```python
   BUCKETS = {
       "BASIC_ALL": [...],
       "EXTENDED_ALL": [...],
       "MY_NEW_BUCKET": ['my_opt1', 'my_opt2'],
   }
   ```

2. **Set Bucket Dependencies**  
   In `BUCKET_ADJ_LIST`, specify dependencies between buckets:
   ```python
   BUCKET_ADJ_LIST = {
       "BASIC_ALL": [],
       "MY_NEW_BUCKET": ["BASIC_ALL"],  
   }
   ```
   > **Note on Dependency Direction:**  
   > - For **individual optimizations** (ops), the adjacency list (`adj_list`) expresses dependencies as:  
   >   `'my_opt': ['other_opt']`  
   >   This means **`my_opt` must run before `other_opt`**.
   >
   > - For **buckets**, the adjacency list (`BUCKET_ADJ_LIST`) expresses dependencies in the opposite direction:  
   >   `'MY_NEW_BUCKET': ['BASIC_ALL']`  
   >   This means **`MY_NEW_BUCKET` depends on `BASIC_ALL`**, so `BASIC_ALL` and its optimizations will be enabled and run before `MY_NEW_BUCKET`.
   >
   > Be careful with this difference when adding new dependencies!
   

3. **Enable Buckets in Optimizer**  
   Use `get_optimizers(bucket_flags=[...])` to enable your bucket for a run.

4. **Bucket Execution Flow**  
   - When a bucket is enabled, all optimizations in that bucket are considered.
   - Dependencies are resolved recursively, so all prerequisite buckets are also enabled.
   - The optimizer runs optimizations in a topologically sorted order based on dependencies.

5. **Logging**  
   The optimizer logs which buckets are enabled and which optimizations are run.

### Example

To enable all layout-related optimizations and their dependencies:
```python
optimizers = get_optimizers(bucket_flags=['LAYOUT_ALL'])
```

### Notes

- Buckets are useful for testing, debugging, or deploying groups of related optimizations.
- You can still enable or disable individual optimizations as needed.
- See [`optimize.py`](optimize.py) and [`ops.py`](ops.py) for implementation details.

## Adding Optimizations Using Subgraph Insertion API

A new, flexible way to add optimizations is by generating a replacement subgraph (for example, using PyTorch), exporting it to ONNX, and inserting it into the main graph using the `insert_subgraph_with_mappings` API. This approach minimizes manual rewiring and makes complex graph surgery much easier.

### Workflow

1. **Identify the node(s) to replace** (e.g., all `Neg` nodes).
2. **Dynamically create a replacement subgraph** using PyTorch (or another framework), export it to ONNX, and import it with ONNX GraphSurgeon.
3. **Map the subgraph’s inputs and outputs** to the original graph’s tensors.
4. **Insert the subgraph** using `insert_subgraph_with_mappings`, which handles most of the rewiring automatically.

### Example

See [1_negToMul.ipynb](example/1_negToMul.ipynb) for a step-by-step tutorial.
In this example, every `Neg` node is replaced by a `Mul` node with `-1`, using a PyTorch module exported to ONNX.

#### Key Advantages

- **Rapid prototyping:** No need to manually construct ONNX nodes or handle all wiring.
- **Reusability:** Easily adapt the approach for other node replacements or subgraph insertions.
- **Reliability:** The API ensures correct mapping and minimizes manual errors.

#### API Reference

```python
from osrt_model_tools.onnx_tools.tidl_onnx_model_optimizer.src.common import insert_subgraph_with_mappings

success = insert_subgraph_with_mappings(
    graph,            # The main ONNX GraphSurgeon graph
    input_mapping,    # Dict: subgraph input name → main graph tensor name
    output_mapping,   # Dict: subgraph output name → main graph tensor name
    subgraph,         # The ONNX GraphSurgeon subgraph to insert
    suffix            # (Optional) Suffix for unique naming
)
```

