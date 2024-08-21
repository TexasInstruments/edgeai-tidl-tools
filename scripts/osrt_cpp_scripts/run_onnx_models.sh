#!/bin/bash
cd ../../
Field_Separator=$IFS
IFS=,
arch=$(uname -m)
if [[ $arch == x86_64 ]]; then
loop_count=2
elif [[ $arch == aarch64 ]]; then
loop_count=20
else
echo 'Processor Architecture must be x86_64 or aarch64'
echo 'Processor Architecture "'$arch'" is Not Supported '
return
fi


# #od-onnx-modelzoo-recommended
# DataList="od-8000_onnxrt_mlperf_ssd_resnet34-ssd1200_onnx,od-8020_onnxrt_edgeai-mmdet_ssd_mobilenetv2_lite_512x512_20201214_model_onnx,od-8030_onnxrt_edgeai-mmdet_ssd_mobilenetv2_fpn_lite_512x512_20201110_model_onnx,od-8040_onnxrt_edgeai-mmdet_ssd_regnetx-200mf_fpn_bgr_lite_320x320_20201010_model_onnx,od-8050_onnxrt_edgeai-mmdet_ssd_regnetx-800mf_fpn_bgr_lite_512x512_20200919_model_onnx,od-8060_onnxrt_edgeai-mmdet_ssd_regnetx-1.6gf_fpn_bgr_lite_768x768_20200923_model_onnx,od-8080_onnxrt_edgeai-mmdet_yolov3_regnetx-1.6gf_bgr_lite_512x512_20210202_model_onnx,od-8090_onnxrt_edgeai-mmdet_retinanet_regnetx-800mf_fpn_bgr_lite_512x512_20200908_model_onnx,od-8100_onnxrt_weights_yolov5s6_640_ti_lite_37p4_56p0_onnx,od-8110_onnxrt_weights_yolov5s6_384_ti_lite_32p8_51p2_onnx,od-8120_onnxrt_weights_yolov5m6_640_ti_lite_44p1_62p9_onnx,od-8140_onnxrt_edgeai-yolox_yolox-s-ti-lite_39p1_57p9_onnx,od-8160_onnxrt_edgeai-tv_ssdlite_mobilenet_v2_fpn_lite_512x512_20211015_dummypp_onnx,od-8170_onnxrt_edgeai-tv_ssdlite_regnet_x_800mf_fpn_lite_20211030_dummypp_onnx"
# for val in $DataList;
# do
#  echo $val
#  ./bin/Release/ort_main -f  ../edgeai-modelzoo/modelartifacts/8bits/${val}/ -v 1 -i "test_data/ADE_val_00001801.jpg"  
# done

# #cl-onnx-modelzoo-recommended
# DataList="cl-6060_onnxrt_edgeai-tv_mobilenet_v1_20190906_onnx,cl-6070_onnxrt_edgeai-tv_mobilenet_v2_20191224_onnx,cl-6078_onnxrt_edgeai-tv_mobilenet_v2_qat-p2_20201213_onnx,cl-6080_onnxrt_torchvision_shufflenet_v2_x1.0_onnx,cl-6090_onnxrt_torchvision_mobilenet_v2_tv_onnx,cl-6098_onnxrt_torchvision_mobilenet_v2_tv_qat-p2_onnx,cl-6100_onnxrt_torchvision_resnet18_onnx,cl-6110_onnxrt_torchvision_resnet50_onnx,cl-6120_onnxrt_fbr-pycls_regnetx-400mf_onnx,cl-6130_onnxrt_fbr-pycls_regnetx-800mf_onnx,cl-6140_onnxrt_fbr-pycls_regnetx-1.6gf_onnx,cl-6150_onnxrt_edgeai-tv_mobilenet_v2_1p4_qat-p2_20210112_onnx,cl-6360_onnxrt_fbr-pycls_regnetx-200mf_onnx,cl-6440_onnxrt_pingolh-hardnet_hardnet68_onnx,cl-6450_onnxrt_pingolh-hardnet_hardnet85_onnx,cl-6460_onnxrt_pingolh-hardnet_hardnet68ds_onnx,cl-6470_onnxrt_pingolh-hardnet_hardnet39ds_onnx,cl-6480_onnxrt_edgeai-tv_mobilenet_v3_lite_small_20210429_onnx,cl-6488_onnxrt_edgeai-tv_mobilenet_v3_lite_small_qat-p2_20210429_onnx,cl-6490_onnxrt_edgeai-tv_mobilenet_v3_lite_large_20210507_onnx"
# for val in $DataList;
# do
#  echo $val
#  ./bin/Release/ort_main -f  ../edgeai-modelzoo/modelartifacts/8bits/${val}/ -v 1 -i "test_data/airshow.jpg"
# done

# ss-onnx-modelzoo-recommended
# DataList="ss-8610_onnxrt_edgeai-tv_deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4_onnx,ss-8630_onnxrt_edgeai-tv_unet_aspp_mobilenetv2_edgeailite_512x512_20210306_outby2_onnx,ss-8650_onnxrt_edgeai-tv_fpn_aspp_mobilenetv2_edgeailite_512x512_20210306_outby4_onnx,ss-8670_onnxrt_edgeai-tv_fpn_aspp_mobilenetv2_1p4_edgeailite_512x512_20210307_outby4_onnx,ss-8690_onnxrt_edgeai-tv_fpn_aspp_regnetx400mf_edgeailite_384x384_20210314_outby4_onnx,ss-8700_onnxrt_edgeai-tv_fpn_aspp_regnetx800mf_edgeailite_512x512_20210312_outby4_onnx,ss-8710_onnxrt_edgeai-tv_deeplabv3plus_mobilenetv2_edgeailite_512x512_20210405_onnx,ss-8720_onnxrt_edgeai-tv_fpn_aspp_regnetx800mf_edgeailite_512x512_20210405_onnx,ss-8730_onnxrt_edgeai-tv_deeplabv3_mobilenet_v3_large_lite_512x512_20210527_onnx"
# for val in $DataList;
# do
#  echo $val
#  ./bin/Release/ort_main -f  ../edgeai-modelzoo/modelartifacts/8bits/${val}/ -v 1 -i "test_data/ADE_val_00001801.jpg"
# done


# #running default models

#od,ss-onnx-deafult models
DataList="od-ort-ssd-lite_mobilenetv2_fpn"

for val in $DataList;
do
 echo $val
 ./bin/Release/ort_main -f "model-artifacts/${val}/" -v 1 -i "test_data/ADE_val_00001801.jpg"  -c ${loop_count}
done
cd -