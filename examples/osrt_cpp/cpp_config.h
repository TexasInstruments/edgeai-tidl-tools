/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define NUM_CONFIGS 4
namespace tidl
{
    namespace config
    {
        typedef enum Modeltype // TODO update to tidl modelzoo types
        {
            OD,
            SEG,
            CLF
        } Modeltype;

        typedef struct model_config
        {
            std::string artifact_path;
            std::string image_path;
            std::string model_path;
            std::string labels_path;
            Modeltype model_type;
            std::vector<float> mean;
            std::vector<float> std;
        } model_config;

        model_config model_configs[] =
            {
                {"model-artifacts/tfl/mobilenet_v1_1.0_224/",
                 "test_data/airshow.jpg",
                 "models/public/tflite/mobilenet_v1_1.0_224.tflite",
                 "test_data/labels.txt",
                 CLF,
                 {127.5, 127.5, 127.5},
                 {0.007843,0.007843,0.007843}},

                {"model-artifacts/tfl/ssd_mobilenet_v2_300_float/",
                 "test_data/ADE_val_00001801.jpg",
                 "models/public/tflite/ssd_mobilenet_v2_300_float.tflite",
                 "",
                 OD,
                 {127.5, 127.5, 127.5},
                 {0.007843,0.007843,0.007843}},
                {"model-artifacts/tfl/deeplabv3_mnv2_ade20k_float/",
                 "test_data/ADE_val_00001801.jpg",
                 "models/public/tflite/deeplabv3_mnv2_ade20k_float.tflite",
                 "",
                 SEG,
                 {127.5, 127.5, 127.5},
                {0.007843,0.007843,0.007843}},
                {"model-artifacts/ort/resnet18-v1/",
                 "test_data/airshow.jpg",
                 "models/public/onnx/resnet18_opset9.onnx",
                 "test_data/labels.txt",
                 CLF,
                 {123.675, 116.28, 103.53},
                 {0.017125, 0.017507, 0.017429}}

        };
    } //tidl::config
} //tidl