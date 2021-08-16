# Edgeai TIDL Tools and Examples

This repository contains example developed for Deep learning runtime (DLRT) offering provided by TI’s edge AI solutions. This repository also contains tools that can help in deploying AI applications on TI’s edgeai solutions quickly to achieve most optimal performance.

## Steps to Run

1. Run Setup.sh to install all the dependencies. This would download and intall all the python packages amd PC tools required for compiling models for TI device
2. Set below to environement vairables. If you are building complete SDK from source, then thes path can be set as per the SDK intall paths

```
export TIDL_TOOLS_PATH=/base_repo_path/tidl_tools
export LD_LIBRARY_PATH=$TIDL_TOOLS_PATH 
```
3. Run below script to compile and validate the models in PC
```
./scripts/run_python_examples.sh
```
