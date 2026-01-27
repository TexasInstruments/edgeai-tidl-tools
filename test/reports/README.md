# Test Reports

This directory contains comprehensive operator test reports for various TI SoC. These reports are generated using the pytest-based [TIDL Unit Test Framework](../tidl_unit/README.md) and are published for each release of edgeai-tidl-tools.

## Available Reports

The reports follow a consistent naming convention:

```
Operator_Test_Report_[Device]_[bit].xlsx
```

Where:
- `[Device]` is the TI device family/model
- `[bit]` is the quantization precision (8bit or 16bit)

## Report Contents

Each excel report includes the following sheets:

- **Versions** - This sheet contains SDK, TIDL TOOLS and SOC information used to perform and generate the reports. 
- **Summary** - This sheet contains a summary of all tested operators, TIDL offload information, pass/fail inference for both x86 and TI SoC runs 
- **\<Operators\>** - Each operator has a dedicated sheet highlighting its attributes, shapes, pass/fail and TIDL offload information

## Additional Resources
- [Supported Operators](../../docs/operators.md): List of operators supported by TIDL
- [TIDL Unit Test Framework](../tidl_unit/README.md): Learn about the pytest-based framework used to generate these reports
