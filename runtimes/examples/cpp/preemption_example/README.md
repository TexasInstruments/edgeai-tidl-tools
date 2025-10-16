# Preemption Example

This example demonstrates TIDL's preemption capabilities, allowing multiple neural networks to execute concurrently with different priority levels on DSP cores.

## Overview

The `preemption_example.cpp` showcases:

- Priority-based execution of multiple neural networks
- Context switching between networks based on priority
- Configuration of the `max_preempt_delay` parameter to control preemption behavior
- Performance comparison between preemptive and non-preemptive execution
- Results visualization through formatted tables

## Requirements

- C++17 compatible compiler
- TIDL Runtime
- YAML-CPP library
- pthread library for multi-threading

## Key Concepts

### Priority-Based Execution

When multiple networks are executing on the DSP, TIDL uses a priority-based scheduling mechanism:

- Each network can be assigned a priority value (lower numerical values indicate higher priority)
- Higher priority networks can preempt (interrupt) lower priority networks
- Networks with the same priority are executed sequentially in a round-robin fashion

### Max Preempt Delay

The `max_preempt_delay` property controls when and how preemption occurs:

- This property is applicable only when a lower priority network is being preempted by a higher priority one
- It specifies a "breathing room" time window for the lower priority network
- The system determines the optimal point for context switching within this window

#### Behavior with Different Values

| `max_preempt_delay` Value | Behavior |
|---------------------------|----------|
| 0 | Immediate preemption after the current layer completes |
| Small value (e.g., 2ms) | The lower priority network has a short breathing room to find an optimal switching point |
| Large value | The lower priority network has more breathing room to complete operations before being preempted |
| FLT_MAX | The system will only preempt at the most optimal point in the entire network |

For more detailed information about preemption in TIDL, see the [preemption documentation](../../../docs/preemption.md).

## Example Structure

The example runs a series of tests with different preemption configurations:

### Test 1: Same Priority Networks
- Two networks with the same priority (0)
- Both networks have `max_preempt_delay = FLT_MAX`
- Networks execute sequentially in round-robin order

### Test 2: Different Priority Networks with Maximum Flexibility
- Network A: priority 1, `max_preempt_delay = FLT_MAX`
- Network B: priority 0, `max_preempt_delay = FLT_MAX`
- Network B can preempt Network A, but only at optimal points

### Test 3: Different Priority Networks with Large Breathing Room
- Network A: priority 1, `max_preempt_delay = 7ms`
- Network B: priority 0, `max_preempt_delay = FLT_MAX`
- Network B can preempt Network A after finding an optimal point within 7ms

### Test 4: Different Priority Networks with Medium Breathing Room
- Network A: priority 1, `max_preempt_delay = 3ms`
- Network B: priority 0, `max_preempt_delay = FLT_MAX`
- Network B can preempt Network A after finding an optimal point within 3ms

### Test 5: Different Priority Networks with Immediate Preemption
- Network A: priority 1, `max_preempt_delay = 0`
- Network B: priority 0, `max_preempt_delay = FLT_MAX`
- Network B preempts Network A immediately after the current layer completes

## Test Methodology

For each test configuration, the example:

1. Runs each network individually without preemption to establish a baseline
2. Runs both networks concurrently with preemption enabled
3. Measures and reports the average processing time for each network
4. Saves output tensors for verification
5. Displays results in formatted tables for easy comparison

## Usage

### Building the Example

```bash
cd runtimes
mkdir build && cd build
cmake ..
make
```

### Running the Example

```bash
./examples/cpp/preemption_example/preemption_example
```

## Output

The example outputs:

1. Test configuration details for each test
2. Average processing time for each network in both baseline and preemptive modes
3. Total number of iterations completed during the test duration
4. Output tensors saved to the `outputs/test_<test_number>/<model_name>/<preemption|no_preemption>/` directory structure
5. Results tables showing performance metrics for all tests

### Results Tables

At the end of execution, the example displays two formatted tables:

#### 1. Baseline Performance (Without Parallel Processing)

This table shows the performance of each network when run individually without preemption:

```
+----------+----------+--------------+----------+--------------+-----------------+-----------------+
| Test ID  | N1 - Pri | N1 - Delay   | N2 - Pri | N2 - Delay   | N1 - Time (ms)  | N2 - Time (ms)  |
+----------+----------+--------------+----------+--------------+-----------------+-----------------+
| 1        | 0        | FLT_MAX      | 0        | FLT_MAX      | XX.XXX          | XX.XXX          |
| 2        | 1        | FLT_MAX      | 0        | FLT_MAX      | XX.XXX          | XX.XXX          |
| 3        | 1        | 7.00         | 0        | FLT_MAX      | XX.XXX          | XX.XXX          |
| 4        | 1        | 3.00         | 0        | FLT_MAX      | XX.XXX          | XX.XXX          |
| 5        | 1        | 0.00         | 0        | FLT_MAX      | XX.XXX          | XX.XXX          |
+----------+----------+--------------+----------+--------------+-----------------+-----------------+
```

#### 2. Performance with Parallel Processing (With Preemption)

This table shows the performance when both networks run concurrently with preemption enabled:

```
+----------+----------+--------------+----------+--------------+-----------------+-----------------+
| Test ID  | N1 - Pri | N1 - Delay   | N2 - Pri | N2 - Delay   | N1 - Time (ms)  | N2 - Time (ms)  |
+----------+----------+--------------+----------+--------------+-----------------+-----------------+
| 1        | 0        | FLT_MAX      | 0        | FLT_MAX      | XX.XXX          | XX.XXX          |
| 2        | 1        | FLT_MAX      | 0        | FLT_MAX      | XX.XXX          | XX.XXX          |
| 3        | 1        | 7.00         | 0        | FLT_MAX      | XX.XXX          | XX.XXX          |
| 4        | 1        | 3.00         | 0        | FLT_MAX      | XX.XXX          | XX.XXX          |
| 5        | 1        | 0.00         | 0        | FLT_MAX      | XX.XXX          | XX.XXX          |
+----------+----------+--------------+----------+--------------+-----------------+-----------------+
```

#### Table Column Descriptions:

- **Test ID**: The test number (1-5)
- **N1 - Pri**: Priority value for Network 1 (lower value = higher priority)
- **N1 - Delay**: Max preempt delay value for Network 1 in milliseconds (FLT_MAX = maximum flexibility)
- **N2 - Pri**: Priority value for Network 2
- **N2 - Delay**: Max preempt delay value for Network 2
- **N1 - Time (ms)**: Average inference time for Network 1 in milliseconds
- **N2 - Time (ms)**: Average inference time for Network 2 in milliseconds

### Analyzing the Results

By comparing the two tables, you can observe:

1. **Impact of Preemption**: Compare the inference times between baseline and preemptive execution to see how preemption affects performance
2. **Effect of Priority**: Networks with higher priority (lower numerical value) typically experience less performance degradation
3. **Effect of Max Preempt Delay**: As the delay value decreases, the higher priority network's performance improves, potentially at the expense of the lower priority network
4. **Optimal Settings**: Identify which configuration provides the best balance between responsiveness and throughput for your specific use case

## Implementation Details

### Directory Structure

The example uses the following directory structure:

- **Model Artifacts**: Located at `../../model-artifacts/<model_name>/artifacts/` relative to the executable
- **Output Files**: Saved to `outputs/test_<test_number>/<model_name>/<preemption|no_preemption>/` 

### Multi-threading

The example uses pthreads to create multiple threads, each running a different neural network. Synchronization between threads is handled using:

- `pthread_mutex_t` for protecting critical sections
- `pthread_barrier_t` for synchronizing the start of execution across threads

### Random Input Generation

The example populates inputs with pseudo-random numbers **byte by byte** incrementally from 0-255 and then reset to 0 and start again.
Ex: 0,1,2,3,4.....255,0,1,2,3,4...255,0...255

## Key Files

- `preemption_example.cpp`: Main example source code
- `preemption_example.h`: Header file with declarations and includes

## Customization

To customize the example for your own models:

1. Modify the `gModelsMap` vector in `main()` to include your models
2. For each model, specify:
   - Model name
   - Priority value
   - Max preempt delay value
   - Runtime type

Example:
```cpp
static std::vector<std::vector<ModelInfo>> gModelsMap = 
{
    /* Test 1 */
    {
        /* Threads*/
        {
            "your-model-1", 0, FLT_MAX, "tidlrt" // {Model, Priority, Max Preempt Delay, runtime}
        },
        {
            "your-model-2", 1, 3.0, "tidlrt"
        }
    }
};
```

## Limitations

- The example currently only supports the TIDL runtime
- Preemption occurs only at layer boundaries, not within the execution of a single layer
- Very small networks with few layers may have limited preemption opportunities
