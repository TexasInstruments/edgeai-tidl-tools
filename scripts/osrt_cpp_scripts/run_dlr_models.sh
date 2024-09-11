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
# DataList="od-5020_tvmdlr_gluoncv-mxnet_yolo3_mobilenet1.0_coco-symbol_json"
# for val in $DataList;
# do
#  echo $val
#   ./bin/Release/dlr_main -f  ../edgeai-modelzoo/modelartifacts/8bits/${val}/ -v 1 -i "test_data/ADE_val_00001801.jpg"   

# done

# #cl tvm dlr
# DataList="cl-3410_tvmdlr_gluoncv-mxnet_mobilenetv2_1.0-symbol_json,cl-3420_tvmdlr_gluoncv-mxnet_resnet50_v1d-symbol_json,cl-3430_tvmdlr_gluoncv-mxnet_xception-symbol_json,cl-3480_tvmdlr_gluoncv-mxnet_hrnet_w18_small_v2_c-symbol_json"
# for val in $DataList;
# do
#  echo $val
#   ./bin/Release/dlr_main -f  ../edgeai-modelzoo/modelartifacts/8bits/${val}/ -v 1 -i "test_data/airshow.jpg"  

# done

# #ss-dlr-modelzoo-recommended
# DataList="ss-5720_tvmdlr_edgeai-tv_fpn_aspp_regnetx800mf_edgeailite_512x512_20210405_onnx"
# for val in $DataList;
# do
#  echo $val
#   ./bin/Release/dlr_main -f  ../edgeai-modelzoo/modelartifacts/8bits/${val}/ -v 1 -i "test_data/ADE_val_00001801.jpg"
# done

#running defalt models

#cl-dlr-default models
if [[ $arch == x86_64 ]]; then
DataList="cl-dlr-onnx_mobilenetv2,cl-dlr-tflite_inceptionnetv3"
elif [[ $arch == aarch64 ]]; then
DataList="cl-dlr-onnx_mobilenetv2_device,cl-dlr-tflite_inceptionnetv3_device"
else
echo 'Processor Architecture must be x86_64 or aarch64'
echo 'Processor Architecture "'$arch'" is Not Supported '
return
fi

for val in $DataList;
do
 echo $val
 ./bin/Release/dlr_main -f "model-artifacts/${val}/artifacts/" -v 1 -i "test_data/airshow.jpg"  -c ${loop_count}
done

cd -