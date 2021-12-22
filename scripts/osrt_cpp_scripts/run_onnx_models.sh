#!/bin/bash
cd ../../
#od-onnx-modelzoo-recommended
DataList="od-8000_onnxrt_mlperf_ssd_resnet34-ssd1200_onnx,od-8020_onnxrt_edgeai-mmdet_ssd-lite_mobilenetv2_512x512_20201214_model_onnx,od-8030_onnxrt_edgeai-mmdet_ssd-lite_mobilenetv2_fpn_512x512_20201110_model_onnx,od-8040_onnxrt_edgeai-mmdet_ssd-lite_regnetx-200mf_fpn_bgr_320x320_20201010_model_onnx,od-8050_onnxrt_edgeai-mmdet_ssd-lite_regnetx-800mf_fpn_bgr_512x512_20200919_model_onnx,od-8060_onnxrt_edgeai-mmdet_ssd-lite_regnetx-1.6gf_bifpn168x4_bgr_768x768_20201026_model_onnx,od-8080_onnxrt_edgeai-mmdet_yolov3-lite_regnetx-1.6gf_bgr_512x512_20210202_model_onnx,od-8090_onnxrt_edgeai-mmdet_retinanet-lite_regnetx-800mf_fpn_bgr_512x512_20200908_model_onnx"
Field_Separator=$IFS
IFS=,
for val in $DataList;
do
 echo $val
 ./bin/Release/ort_main -z "/home/a0496663/edgeai-tidl-apps/samples/${val}/" -v 1 -i "test_data/ADE_val_00001801.jpg" -l "test_data/labels.txt" -a 1  
done

#cl-onnx-modelzoo-recommended
DataList="cl-6060_onnxrt_edgeai-tv_mobilenet_v1_20190906_onnx,cl-6070_onnxrt_edgeai-tv_mobilenet_v2_20191224_onnx,cl-6078_onnxrt_edgeai-tv_mobilenet_v2_qat-p2_20201213_onnx,cl-6080_onnxrt_torchvision_shufflenet_v2_x1.0_onnx,cl-6090_onnxrt_torchvision_mobilenet_v2_tv_onnx,cl-6098_onnxrt_torchvision_mobilenet_v2_tv_qat-p2_onnx,cl-6100_onnxrt_torchvision_resnet18_onnx,cl-6110_onnxrt_torchvision_resnet50_onnx,cl-6120_onnxrt_fbr-pycls_regnetx-400mf_onnx,cl-6130_onnxrt_fbr-pycls_regnetx-800mf_onnx,cl-6140_onnxrt_fbr-pycls_regnetx-1.6gf_onnx,cl-6150_onnxrt_edgeai-tv_mobilenet_v2_1p4_qat-p2_20210112_onnx,cl-6360_onnxrt_fbr-pycls_regnetx-200mf_onnx,cl-6440_onnxrt_pingolh-hardnet_hardnet68_onnx,cl-6450_onnxrt_pingolh-hardnet_hardnet85_onnx,cl-6460_onnxrt_pingolh-hardnet_hardnet68ds_onnx,cl-6470_onnxrt_pingolh-hardnet_hardnet39ds_onnx,cl-6480_onnxrt_edgeai-tv_mobilenet_v3_lite_small_20210429_onnx,cl-6488_onnxrt_edgeai-tv_mobilenet_v3_lite_small_qat-p2_20210429_onnx,cl-6490_onnxrt_edgeai-tv_mobilenet_v3_lite_large_20210507_onnx"
for val in $DataList;
do
 echo $val
 ./bin/Release/ort_main -z "/home/a0496663/edgeai-tidl-apps/samples/${val}/" -v 1 -i "test_data/airshow.jpg" -l "test_data/labels.txt"  -a 1
done

ss-onnx-modelzoo-recommended
DataList="ss-8610_onnxrt_edgeai-tv_deeplabv3lite_mobilenetv2_512x512_ade20k32_outby4_onnx,ss-8630_onnxrt_edgeai-tv_unetlite_aspp_mobilenetv2_512x512_ade20k32_outby2_onnx,ss-8650_onnxrt_edgeai-tv_fpnlite_aspp_mobilenetv2_512x512_ade20k32_outby4_onnx,ss-8670_onnxrt_edgeai-tv_fpnlite_aspp_mobilenetv2_1p4_512x512_ade20k32_outby4_onnx,ss-8690_onnxrt_edgeai-tv_fpnlite_aspp_regnetx400mf_ade20k32_384x384_outby4_onnx,ss-8700_onnxrt_edgeai-tv_fpnlite_aspp_regnetx800mf_ade20k32_512x512_outby4_onnx,ss-8710_onnxrt_edgeai-tv_deeplabv3lite_mobilenetv2_cocoseg21_512x512_20210405_onnx,ss-8720_onnxrt_edgeai-tv_fpnlite_aspp_regnetx800mf_cocoseg21_512x512_20210405_onnx,ss-8730_onnxrt_edgeai-tv_deeplabv3_mobilenet_v3_lite_large_512x512_20210527_onnx"
for val in $DataList;
do
 echo $val
 ./bin/Release/ort_main -z "/home/a0496663/edgeai-tidl-apps/samples/${val}/" -v 1 -i "test_data/ADE_val_00001801.jpg" -l "test_data/labels.txt"  -a 1
done


# #running default models



#ss-tflite-default models
DataList="deeplabv3lite_mobilenetv2"
for val in $DataList;
do
 echo $val 
 echo 1
 ./bin/Release/ort_main -z "model-artifacts/ort/${val}/" -v 1 -i "test_data/ADE_val_00001801.jpg" -l "test_data/labels.txt" -a 1 
done

#cl-onnx-default models
DataList="resnet18-v1"
for val in $DataList;
do
 echo $val
 echo 2
 ./bin/Release/ort_main -z "model-artifacts/ort/${val}/" -v 1 -i "test_data/airshow.jpg" -l "test_data/labels.txt" -a 1 
done

#od-tflite-deafult models
DataList="ssd-lite_mobilenetv2_fpn"
for val in $DataList;
do
 echo $val
 echo 3
 ./bin/Release/ort_main -z "model-artifacts/ort/${val}/" -v 1 -i "test_data/ADE_val_00001801.jpg" -l "test_data/labels.txt" -a 1 
done
# cd -