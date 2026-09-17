# Manual Super-Tiling Group Configuration Guide

## Overview

Manual super-tiling groups allow you to intelligently group neural network layers to optimize memory bandwidth utilization and improve inference latency. By analyzing DDR transaction patterns and grouping layers with offchip memory(DDR) access, TIDL can optimize memory planning and exeuction strategies further reducing the memory access bottlenecks.

This document explains the practical workflow for identifying and configuring manual super-tiling groups.

## Why Super-Tiling?

The memory bandwidth in current system for Deep Neural Networks could be very
large. But SoCs have a limited amount of L3 memory (usually 4MB/8MB). This would
make the network un-realizable on the system. One way to approach this problem
is to move the layers with output size greater than L3 memory to external memory.
However this would create a I/O bottleneck and low compute utilization.

TIDL implements a method called supertiling in which processing for each tensor
is split into multiple sub-tensors called supertiles. A supertile is selected
such that output of any given layer that is being processed can fit in L3.

---

### ⚠️ Important Note: Experimental Technique

**Manual super-tiling is an experimental optimization.** Latency may improve, degrade, or remain unchanged depending on:
- Network architecture and layer characteristics
- Grouping decisions accuracy
- Device-specific memory hierarchy
- Actual runtime conditions

**Always profile and validate:**
1. Measure latency **without** manual grouping (baseline)
2. Apply manual grouping and recompile
3. Measure latency **with** manual grouping
4. Compare results - if worse, revert or adjust groups

Manual grouping is not a guaranteed optimization; it requires experimentation and validation to achieve improvements.

---

## Quick Reference

### Key Principle
**Only specify execution orders you want to optimize. TIDL automatically handles all other layers.**

### File Format (Minimal Example)
```
# supertiling_info_groups.txt
startExecOrderNum, endExecOrderNum

1, 10
30, 45
```

### Workflow Summary
1. **Compile** with `high_resolution_optimization = 0`
2. **Analyze CSV** - look at `srcMem-IN` and `dstMem-OUT` columns
3. **Identify** groups with regular DDR transactions
6. **Configure** group information file with start/end execution orders
4. **Check** - verify layers support ST, no branches in Compiled Graph.
5. **Measure** - baseline latency → with grouping → compare

### Tips
- Start simple: only 1-2 groups
- Gaps OK: unspecified layers auto-managed
- Validate: measure before/after latency
- Iterate: if no improvement, adjust or revert

### Common Issues
| Issue | Solution |
|-------|----------|
| "Layer type not supported" | Remove that layer from group |
| No latency improvement | Try different groups or revert |
| Output changed | Non-ST layer in group, or branches exist in groups - remove it |

---

## Step 1: Compile Model Without High Resolution Optimization

Compile your model with the high resolution optimization flag disabled to generate a performance estimate CSV.

### Configure without high_resolution_optimization and compile

In your TIDL config file, set:

```
advanced_options:high_resolution_optimization = 0
```

Or ensure it's not set to enable baseline performance estimation.

### Output Location

Post compilation the performance estimate CSV is generated in the model artifacts subdirectory:

```
<output_dir>/artifacts/tempDir/*.csv
```

Or check the compilation log for the exact path.

---

## Step 2: Analyze Performance Estimate CSV

Open the generated CSV file and examine the columns that describe DDR transaction patterns.

### Key CSV Columns

| Column | Meaning | Values |
|--------|---------|--------|
| `executionId` | Execution order sequence | 1, 2, 3, ... |
| `lyrNum` | Layer number in network | Actual layer ID |
| `LyrType` | Layer type | `TIDL_ConvolutionLayer`, `TIDL_PoolingLayer`, etc. |
| `inVol(KB)` | Input volume in KB | Memory size |
| `outVol(KB)` | Output volume in KB | Memory size |
| `wtVol(KB)` | Weight volume in KB | Memory size |
| **`srcMem-IN`** | Source memory for input | `DDR`, `L2`, `MSMC`, etc. |
| **`dstMem-IN`** | Destination memory for input | `L2`, `MSMC`, `DDR`, `NONE` |
| **`srcMem-OUT`** | Source memory for output | `DDR`, `MSMC`, `L2`, etc. |
| **`dstMem-OUT`** | Destination memory for output | `DDR`, `MSMC`, `L2`, `NONE` |
| `MSMC_Hold_Size` | Memory held from dependencies | Size in KB |
| `MSMC_Hold_Layers` | Which layers are held | Layer references |

### Example CSV

```
executionId, lyrNum, LyrType,                 inVol(KB), outVol(KB), wtVol(KB), srcMem-IN, dstMem-IN, srcMem-OUT, dstMem-OUT
1,          1,      TIDL_ConvolutionLayer,     3072.00,   16384.00,    9.50,     [DDR],     L2,       MSMC,       DDR
2,          2,      TIDL_PoolingLayer,         16384.00,   4096.00,    0.00,     [DDR],     L2,       MSMC,       DDR
3,          3,      TIDL_ConvolutionLayer,      4096.00,   4096.00,    4.25,     [DDR],     L2,       MSMC,       DDR
4,          4,      TIDL_ConvolutionLayer,      4096.00,  12288.00,   108.75,    [DDR],     L2,       MSMC,       DDR
5,          5,      TIDL_PoolingLayer,         12288.00,   3072.00,    0.00,     [DDR],     L2,       MSMC,       DDR
6,          7,      TIDL_PoolingLayer,          3072.12,   3072.00,    0.00,     [DDR],     L2,       MSMC,       DDR
7,         10,      TIDL_ConvolutionLayer,      3072.00,    512.00,    6.12,     [DDR],     L2,       MSMC,       DDR
8,          6,      TIDL_ConvolutionLayer,      3072.00,   1024.00,   12.25,     [DDR],     L2,       MSMC,       DDR
9,          9,      TIDL_ConvolutionLayer,      3072.00,    256.00,    3.12,     [DDR],     L2,       MSMC,      NONE
10,        12,      TIDL_ConvolutionLayer,       256.00,    512.00,    4.62,    [MSMC],     L2,       MSMC,       DDR
...
```

---

## Step 3: Identify Groups with Regular DDR Patterns

Analyze the CSV to find consecutive layers with regular DDR transactions.

### What to Look For

**Regular DDR Read-Write Access:**

Look for consecutive execution orders where:
- `srcMem-IN` is all `[DDR]`
- `dstMem-OUT` is all `[DDR]`
- Memory volumes are similar in magnitude
- No sudden jumps in volume requirements

**Example of Good Group:**

```
executionId 1-6 (lyrNum 1-7):
- All read input from DDR
- All write output to DDR
- Input volumes: 3-16 MB (consistent range)
- Output volumes: 0.5-16 MB (consistent range)
```
---

## Step 4: Validate Layer Types Support Super-Tiling

For each group candidate, verify all layers support super-tiling.

### Non-Tileable Layer Types

The following layer types **DO NOT** support super-tiling:

```
TIDL_SliceLayer
TIDL_CropLayer
TIDL_FlattenLayer
TIDL_TransposeLayer
TIDL_PadLayer
TIDL_CustomLayer
TIDL_ScatterElementsLayer
TIDL_GatherLayer
TIDL_GatherElementsLayer
TIDL_GridSampleLayer
TIDL_GatherNDLayer
TIDL_ReshapeLayer (conditional)
TIDL_InnerProductLayer (if wtVol = 0) --> Only MatMul Layers are supported.
TIDL_ConcatLayer
```

**Warning:** If any layer in your proposed group is in this list, then remove it from the group and start the next group from the next layer

### Verification

For each group candidate:

```
Group 1: executionId 1-6 (lyrNum 1, 2, 3, 4, 5, 7)
- Layer 1 (lyrNum 1): TIDL_ConvolutionLayer ✓
- Layer 2 (lyrNum 2): TIDL_PoolingLayer ✓
- Layer 3 (lyrNum 3): TIDL_ConvolutionLayer ✓
- Layer 4 (lyrNum 4): TIDL_ConvolutionLayer ✓
- Layer 5 (lyrNum 5): TIDL_PoolingLayer ✓
- Layer 7 (lyrNum 7): TIDL_PoolingLayer ✓
→ All supported, group is valid ✓
```

---

## Step 5: Check Compiled Graph Visualization for Branch-Free Execution

When layer grouping is generated, Compiled Graph visualizations show the execution flow. Verify that your proposed groups do not contain branches.

### Finding Compiled Graph Visualization Files

The HTML files are typically generated during compilation:

```
<output_dir>/artifacts/*.html
```

### What to Look For

**Branch-Free Group:**
- Layers execute sequentially without splits
- No alternative paths through the group
- All layers feed into next layer in order

Example (✓ Valid):
```
Layer 1 → Layer 2 → Layer 3 → Layer 4 → Layer 5
 [All in same group - linear execution]
```

---

## Step 6: Configure Group Information File

Once you've identified valid groups, create a group information file.

### ⚠️ Key Point: Only Specify Groups You Want to Supertile

**You only need to specify the execution orders for layers you want to manually supertile.** Layers not mentioned in the configuration file will be automatically handled by TIDL with default settings. This means:

- You can start simple with a few groups
- TIDL handles the rest automatically
- No need to specify every single layer
- Easier to iterate and test

### File Format

Create a file (typically named `supertiling_info_groups.txt` or similar):

```
# Supertiling Group Configuration
# Format: start_execution_id, end_execution_id
# Only specify execution orders you want to manually control
# Other layers will be auto-managed by TIDL

1, 6
10, 15
25, 32
```

**Columns:**
- `start_execution_id` - First execution order in the group you want to supertile
- `end_execution_id` - Last execution order in the group you want to supertile

### Example Configuration

For the CSV example above, specify only the groups with regular DDR patterns:

```
# Group 1: executionId 1-7 (regular DDR pattern, all Conv/Pool)
# These layers have consistent DDR access we want to optimize
1, 6

# executionId 7-9: Omitted - let TIDL auto-manage
# (Mixed DDR patterns, don't need manual grouping)

# executionId 10-19: Omitted

# Group 2: executionId 20-25.
# TIDL will automatically assign this a group ID, but won't do the supertiling optimization
20, 25
```

**Result:**
- executionId 1-6: Manually grouped for supertiling (your optimization)
- executionId 7-9: Auto-managed by TIDL (Default behavior)
- executionId 10-15: Manually grouped for supertiling (your optimization)
- executionId 16-24: Automatically grouped but no supertiling.(Default behaviour)
- executionId 25-32: Manually grouped for supertiling (your optimization)
- executionId 32+: Auto-managed by TIDL (Default behavior)

### Benefits

- **Simpler files** - Only specify the groups you're confident about
- **Flexibility** - TIDL handles uncertain layers automatically
- **Easier iteration** - Try a few groups, measure results, add more if beneficial
- **Lower risk** - Don't need to perfectly group all layers
- **Faster experimentation** - Start with high-impact groups only

### Important Rules

1. **Consecutive executionIds only** - Each group must contain contiguous execution orders
2. **No overlaps** - Each executionId can appear in at most one group (or not at all)
3. **Gaps allowed** - executionIds not specified are auto-managed (this is normal and expected)
4. **All layers in group must be ST-supported** - Check against non-tileable list
5. **No branches** - Verify via Compiled Graph before adding to file

### Validation Checklist

- [ ] Start execution order ≤ End execution order for each group
- [ ] No overlaps (same execution order in multiple lines)
- [ ] All layers in each group are ST-supported
- [ ] Each group has coherent DDR pattern
- [ ] No branches within groups (verified via Compiled Graph)
- [ ] File syntax is correct (comma-separated values)

---

## Troubleshooting

### Issue: Layer Type Not Supported

If compilation fails with "layer type X not supported for tiling":
1. Check if layer is in non-tileable list
2. Skip that layer, and start your grouping from the next ST supported Layer.

### Issue: Performance Not Improved

If latency improvement is minimal:
1. Verify CSV columns were analyzed correctly
2. Check the compiled graph for unexpected branches
3. Consider different grouping that better aligns DDR patterns
4. Profile actual execution to confirm pattern assumptions

---

## Profiling and Validation

After applying manual grouping, **always validate the impact**:

### Baseline Measurement (Without Manual Grouping)


Compile without manual grouping
Run inference and measure latency
Record baseline latency (e.g., 150ms)


### Experimental Measurement (With Manual Grouping)


Compile WITH manual grouping
(high_resolution_group_info_file set in config)

Run inference and measure latency
Record latency with grouping (e.g., 145ms or 160ms)


### Analysis

Compare results:

```
Baseline (no grouping):     150 ms
With manual grouping:       145 ms
Improvement:                 3.3% ✓

OR

Baseline (no grouping):     150 ms
With manual grouping:       165 ms
Regression:                 10% ✗ (revert or adjust groups)
```

### Validation Points

- [ ] Latency improved (expected but not guaranteed)
- [ ] Output correctness unchanged (verify inference results)
- [ ] Compilation successful with no warnings

### If Latency Degrades

1. **Review grouping decisions**
   - Were DDR patterns correctly identified?
   - Did you miss branches in the compiled graph?

2. **Try different grouping**
   - Merge some groups
   - Split others
   - Recompile and test

3. **Accept baseline**
   - Manual grouping isn't always beneficial
   - Revert to automatic grouping or no grouping(baseline performance).

---

## Summary Checklist

**Phase 1: Analysis**
- [ ] Compile model with high_res_optimization disabled
- [ ] Generate performance estimate CSV
- [ ] Analyze srcMem-IN, dstMem-OUT columns for regular transaction patterns
- [ ] Verify all layers support super-tiling (check against non-tileable list)
- [ ] Check the compiled graph visualization for branch-free execution paths

**Phase 2: Configuration (Start Simple)**
- [ ] Identify 1-3 groups with the most regular DDR patterns
- [ ] Create minimal configuration file with only these groups
- [ ] Let TIDL auto-manage all other layers (don't over-specify)

**Phase 3: Testing & Validation**
- [ ] Configure TIDL with `high_resolution_group_info_file` path
- [ ] **Measure baseline latency (without manual grouping)**
- [ ] Recompile with manual grouping
- [ ] **Measure latency with grouping and compare**
- [ ] Validate correctness (output unchanged)

**Phase 4: Iterate or Ship**
- [ ] If latency improved: keep configuration, consider adding more groups
- [ ] If latency degraded: adjust groups or revert to auto-management
- [ ] If neutral: consider if complexity worth it; may revert

---

## When Manual Grouping May Not Help

Manual super-tiling may provide little benefit when:

1. **Network already well-optimized** by TIDL's automatic tiling optimizations.
2. **Compute-bound workloads** where memory access isn't the bottleneck
3. **Small networks** where auto-management is already near-optimal
4. **Complex, branchy execution** that doesn't have clear layer groupings

---

