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

#define NUM_CONFIGS 3
namespace tflite
{
    namespace config
    {
        typedef enum TFliteModeltype
        {
            OD,
            SEG,
            CLF
        } TFliteModeltype;

        typedef struct tfl_config
        {
            std::string artifact_path;
            std::string image_path;
            std::string tflite_model_path;
            std::string tflite_labels_path;
            TFliteModeltype model_type;
            float mean;
            float std;
        } tfl_config;

        tfl_config model_configs[] =
            {
                {"model-artifacts/tfl/deeplabv3_mnv2_ade20k_float/",
                 "test_data/ADE_val_00001801.jpg",
                 "models/public/tflite/deeplabv3_mnv2_ade20k_float.tflite",
                 "",
                 OD, 127.5f, 127.5f},

                {"model-artifacts/tfl/ssd_mobilenet_v2_300_float/",
                 "test_data/ADE_val_00001801.jpg",
                 "models/public/tflite/ssd_mobilenet_v2_300_float.tflite",
                 "",
                 SEG, 127.5f, 127.5f},
                {"model-artifacts/tfl/deeplabv3_mnv2_ade20k_float/",
                 "test_data/airshow.jpg",
                 "models/public/tflite/deeplabv3_mnv2_ade20k_float.tflite"  ,
                 "",
                 CLF, 127.5f, 127.5f}

        };
    }
}