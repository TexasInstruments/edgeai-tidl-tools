#!/usr/bin/env bash
skip_setup=0
skip_models_download=0

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --skip_setup)
    skip_setup=1
    ;;
    --skip_models_download)
    skip_models_download=1
    ;;
    -h|--help)
    echo Usage: $0 [options]
    echo
    echo Options,
    echo --skip_setup                      Skip Installing python dependencies. Direclty launch Notebook session
    echo --skip_models_download            Skip Pre-compiled models download
    exit 0
    ;;
esac
shift # past argument
done
set -- "${POSITIONAL[@]}" # restore positional parameters


echo "# ##################################################################"
echo "This script download python modules, edgeai-benchmark, and some precompiled models artifacts.
It also sets other requirements, and, at the end, it launches jupyter notebook server in the EVM
Note: take a note of the EVM's ip address before running this scrip (ifconfig)
and use it in a computer's web browser to access and run the notebooks. ex: http://192.168.1.199:8888"

echo "# ##################################################################"
# target_device - use one of: TDA4VM AM62A AM68A AM69A AM67A

if [ "$SOC" == "am68pa"]
then
    $TARGET_SOC = "TDA4VM"
elif [ "$SOC" == "am68a"]
then
    $TARGET_SOC = "AM68A"
elif [ "$SOC" == "am67a"]
then
    $TARGET_SOC = "AM67A"
elif [ "$SOC" == "am69a"]
then
    $TARGET_SOC = "AM69A"
elif [ "$SOC" == "am62a"]
then
    $TARGET_SOC = "AM62A"
else
    echo "Please set env SOC. Note: am68pa (TDA4VM) is used by default"
    echo "use one of: am62a, am68a, am67a, am69a, or am68pa (TDA4VM)"
    $TARGET_SOC = "TDA4VM"
fi

if [ $skip_setup -eq 0 ]
then
echo "Installing python modules
This step is required only the first time"
pip3 install numpy
pip3 install pycocotools
pip3 install colorama
pip3 install pytest
pip3 install notebook
pip3 install ipywidgets
pip3 install papermill --ignore-installed
pip3 install munkres
pip3 install json_tricks
pip3 install git+https://github.com/jin-s13/xtcocoapi.git
pip3 install h5py
pip3 install scipy
pip3 install plyfile
pip3 install scikit-learn

jupyter nbextension enable --py widgetsnbextension

echo "# ##################################################################"
echo "Clone and install jacinto-ai python module
This could take some time..
This step is required only the first time"
cd ../
git clone --single-branch -b r9.0 https://github.com/TexasInstruments/edgeai-benchmark.git
cd edgeai-benchmark
pip3 install -e ./
cd ../jupyter_notebooks
fi

if [ $skip_models_download -eq 0 ]
then
echo "# ##################################################################"
echo "Download pre-compiled models
For additional models visit: https://software-dl.ti.com/jacinto7/esd/modelzoo/latest/docs/html/index.html
This step is required only the first time"
mkdir prebuilt-models
mkdir prebuilt-models/8bits
cd prebuilt-models/8bits

wget http://software-dl.ti.com/jacinto7/esd/modelzoo/09_00_00/modelartifacts/$TARGET_SOC/8bits/cl-0000_tflitert_imagenet1k_mlperf_mobilenet_v1_1.0_224_tflite.tar.gz
wget http://software-dl.ti.com/jacinto7/esd/modelzoo/09_00_00/modelartifacts/$TARGET_SOC/8bits/3dod-7100_onnxrt_kitti_mmdet3d_lidar_point_pillars_10k_496x432_qat-p2_onnx.tar.gz
wget http://software-dl.ti.com/jacinto7/esd/modelzoo/09_00_00/modelartifacts/$TARGET_SOC/8bits/kd-7060_onnxrt_coco_edgeai-yolox_yolox_s_pose_ti_lite_640_20220301_model_onnx.tar.gz
wget http://software-dl.ti.com/jacinto7/esd/modelzoo/09_00_00/modelartifacts/$TARGET_SOC/8bits/cl-3090_tvmdlr_imagenet1k_torchvision_mobilenet_v2_tv_onnx.tar.gz
wget http://software-dl.ti.com/jacinto7/esd/modelzoo/09_00_00/modelartifacts/$TARGET_SOC/8bits/cl-6090_onnxrt_imagenet1k_torchvision_mobilenet_v2_tv_onnx.tar.gz
wget http://software-dl.ti.com/jacinto7/esd/modelzoo/09_00_00/modelartifacts/$TARGET_SOC/8bits/od-5120_tvmdlr_coco_tf1-models_ssdlite_mobiledet_dsp_320x320_coco_20200519_tflite.tar.gz
wget http://software-dl.ti.com/jacinto7/esd/modelzoo/09_00_00/modelartifacts/$TARGET_SOC/8bits/od-8200_onnxrt_coco_edgeai-mmdet_yolox_nano_lite_416x416_20220214_model_onnx.tar.gz
wget http://software-dl.ti.com/jacinto7/esd/modelzoo/09_00_00/modelartifacts/$TARGET_SOC/8bits/od-2000_tflitert_coco_mlperf_ssd_mobilenet_v1_coco_20180128_tflite.tar.gz
wget http://software-dl.ti.com/jacinto7/esd/modelzoo/09_00_00/modelartifacts/$TARGET_SOC/8bits/ss-5710_tvmdlr_cocoseg21_edgeai-tv_deeplabv3plus_mobilenetv2_edgeailite_512x512_20210405_onnx.tar.gz
wget http://software-dl.ti.com/jacinto7/esd/modelzoo/09_00_00/modelartifacts/$TARGET_SOC/8bits/ss-8710_onnxrt_cocoseg21_edgeai-tv_deeplabv3plus_mobilenetv2_edgeailite_512x512_20210405_onnx.tar.gz
wget http://software-dl.ti.com/jacinto7/esd/modelzoo/09_00_00/modelartifacts/$TARGET_SOC/8bits/ss-2580_tflitert_ade20k32_mlperf_deeplabv3_mnv2_ade20k32_float_tflite.tar.gz

find . -name "*.tar.gz" -exec tar --one-top-level -zxvf "{}" \;
cd ../../
fi

echo "# ##################################################################"
echo "Setup the environment
This step is required everytime notebook server is launched"

if [[ -z "$TIDL_TOOLS_PATH" ]]
then
echo "Setting TIDL_TOOLS_PATH. Note: TIDL_TOOLS_PATH needs to exist for jacinto-ai module, but, this is a dummy path.."
export TIDL_TOOLS_PATH="/opt/jai_tidl_notebooks"
echo "TIDL_TOOLS_PATH=${TIDL_TOOLS_PATH}"
fi

export TIDL_RT_DDR_STATS="1"
export TIDL_RT_PERFSTATS="1"
echo "TIDL_RT_PERFSTATS=${TIDL_RT_PERFSTATS}"

echo "# ##################################################################"
echo "Launch notebook server"
jupyter notebook --allow-root --no-browser --ip=0.0.0.0

