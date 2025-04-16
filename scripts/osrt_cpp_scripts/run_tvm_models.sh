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

if [[ $arch == x86_64 ]]; then
    DataList="cl-tvm-ort-resnet18-v1,cl-tvm-ort-resnet18-v1_c7x"
elif [[ $arch == aarch64 ]]; then
    DataList="cl-tvm-ort-resnet18-v1_device,cl-tvm-ort-resnet18-v1_c7x_device"
else
    echo 'Processor Architecture must be x86_64 or aarch64'
    echo 'Processor Architecture "'$arch'" is Not Supported '
return
fi

for val in $DataList;
do
 echo $val
 ./bin/Release/tvm_main -f "model-artifacts/${val}/artifacts/" -v 1 -i "test_data/airshow.jpg"  -c ${loop_count}
done


if [[ $arch == x86_64 ]]; then
    DataList="od-tvm-ort-ssd-lite_mobilenetv2_fpn,ss-tvm-ort-deeplabv3lite_mobilenetv2"
elif [[ $arch == aarch64 ]]; then
    DataList="od-tvm-ort-ssd-lite_mobilenetv2_fpn_device,ss-tvm-ort-deeplabv3lite_mobilenetv2_device"
else
    echo 'Processor Architecture must be x86_64 or aarch64'
    echo 'Processor Architecture "'$arch'" is Not Supported '
return
fi

for val in $DataList;
do
 echo $val
 ./bin/Release/tvm_main -f "model-artifacts/${val}/artifacts/" -v 1 -i "test_data/ADE_val_00001801.jpg"  -c ${loop_count}
done

cd -
