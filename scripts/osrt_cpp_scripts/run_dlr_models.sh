#!/bin/bash
cd ../../
#od-onnx-modelzoo-recommended
DataList="od-5020_tvmdlr_gluoncv-mxnet_yolo3_mobilenet1.0_coco-symbol_json"
Field_Separator=$IFS
IFS=,
for val in $DataList;
do
 echo $val
  ./bin/Release/dlr_main -z  ../edgeai-modelzoo/modelartifacts/8bits/${val}/ -v 1 -i "test_data/ADE_val_00001801.jpg"  -l "test_data/labels.txt"  -d 1  -y "cpu"

done

#cl tvm dlr
DataList="cl-3410_tvmdlr_gluoncv-mxnet_mobilenetv2_1.0-symbol_json,cl-3420_tvmdlr_gluoncv-mxnet_resnet50_v1d-symbol_json,cl-3430_tvmdlr_gluoncv-mxnet_xception-symbol_json,cl-3480_tvmdlr_gluoncv-mxnet_hrnet_w18_small_v2_c-symbol_json"
for val in $DataList;
do
 echo $val
  ./bin/Release/dlr_main -z  ../edgeai-modelzoo/modelartifacts/8bits/${val}/ -v 1 -i "test_data/airshow.jpg"  -l "test_data/labels.txt"  -d 1  -y "cpu"

done

#ss-dlr-modelzoo-recommended
DataList="ss-5720_tvmdlr_edgeai-tv_fpn_aspp_regnetx800mf_edgeailite_512x512_20210405_onnx"
for val in $DataList;
do
 echo $val
  ./bin/Release/dlr_main -z  ../edgeai-modelzoo/modelartifacts/8bits/${val}/ -v 1 -i "test_data/ADE_val_00001801.jpg"  -l "test_data/labels.txt"  -d 1  -y "cpu"
done

#running defalt models

#cl-dlr-default models
DataList="onnx_mobilenetv2,tflite_inceptionnetv3"
for val in $DataList;
do
 echo $val
 ./bin/Release/dlr_main -z "model-artifacts/dlr/${val}/" -v 1 -i "test_data/airshow.jpg" -l "test_data/labels.txt" -a 1 
done

cd -