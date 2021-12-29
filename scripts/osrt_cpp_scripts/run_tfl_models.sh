#!/bin/bash
cd ../../
Field_Separator=$IFS
IFS=,

#od tflite-modelzoo-recommended
# DataList="od-2000_tflitert_mlperf_ssd_mobilenet_v1_coco_20180128_tflite,od-2010_tflitert_mlperf_ssd_mobilenet_v2_300_float_tflite,od-2020_tflitert_tf1-models_ssdlite_mobiledet_dsp_320x320_coco_20200519_tflite,od-2030_tflitert_tf1-models_ssdlite_mobiledet_edgetpu_320x320_coco_20200519_tflite,od-2060_tflitert_tf1-models_ssdlite_mobilenet_v2_coco_20180509_tflite,od-2070_tflitert_tf2-models_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8_tflite,od-2080_tflitert_tf2-models_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8_tflite,od-2090_tflitert_tf2-models_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8_tflite,od-2110_tflitert_google-automl_efficientdet-lite0_bifpn_maxpool2x2_relu_ti-lite_tflite,od-2130_tflitert_tf2-models_ssd_mobilenet_v2_320x320_coco17_tpu-8_tflite"
# for val in $DataList;
# do
#  echo $val
#  ./bin/Release/tfl_main -z  ../edgeai-modelzoo/modelartifacts/8bits/${val}/ -v 1 -i "test_data/ADE_val_00001801.jpg"
# done

# #cl tflite modelzoo recommended
# DataList="cl-0000_tflitert_mlperf_mobilenet_v1_1.0_224_tflite,cl-0020_tflitert_tf1-models_squeezenet_tflite,cl-0038_tflitert_tf1-models_inception_v1_224_quant_tflite,cl-0040_tflitert_tf1-models_inception_v3_tflite,cl-0050_tflitert_tf1-models_resnet50_v1_tflite,cl-0070_tflitert_tf1-models_mnasnet_1.0_224_tflite,cl-0080_tflitert_mlperf_mobilenet_edgetpu_224_1.0_float_tflite,cl-0090_tflitert_tf-tpu_efficientnet-edgetpu-S_float_tflite,cl-0100_tflitert_tf-tpu_efficientnet-edgetpu-M_float_tflite,cl-0130_tflitert_tf-tpu_efficientnet-lite0-fp32_tflite,cl-0140_tflitert_tf-tpu_efficientnet-lite4-fp32_tflite,cl-0160_tflitert_mlperf_resnet50_v1.5_tflite,cl-0170_tflitert_tf-tpu_efficientnet-lite1-fp32_tflite,cl-0190_tflitert_tf-tpu_efficientnet-edgetpu-L_float_tflite,cl-0260_tflitert_tf1-models_mobilenet_v3-large-minimalistic_224_1.0_float_tflite"

# for val in $DataList;
# do
#  echo $val
#  ./bin/Release/tfl_main -z  ../edgeai-modelzoo/modelartifacts/8bits/${val}/ -v 1 -i "test_data/airshow.jpg"
# done

# #ss-tflite-modelzoo-recommeneded
# DataList="ss-2580_tflitert_mlperf_deeplabv3_mnv2_ade20k32_float_tflite"
# for val in $DataList;
# do
#  echo $val
#  ./bin/Release/tfl_main -z  ../edgeai-modelzoo/modelartifacts/8bits/${val}/ -v 1 -i "test_data/ADE_val_00001801.jpg"
# done

#running default models

#cl-tflite-default models
DataList="cl-tfl-mobilenet_v1_1.0_224"
for val in $DataList;
do
 echo $val
 ./bin/Release/tfl_main -f "model-artifacts/${val}/" -v 1 -i "test_data/airshow.jpg"
done

#od,ss-tflite-deafult models
DataList="od-tfl-ssd_mobilenet_v2_300_float,ss-tfl-deeplabv3_mnv2_ade20k_float"
for val in $DataList;
do
 echo $val
 ./bin/Release/tfl_main -f "model-artifacts/${val}/" -v 1 -i "test_data/ADE_val_00001801.jpg"
done

cd -