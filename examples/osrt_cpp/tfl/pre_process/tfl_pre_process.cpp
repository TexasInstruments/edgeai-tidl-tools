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

/* Module headers. */
#include "tfl_pre_process.h"

namespace tflite
{
    namespace preprocess
    {
        template <class T>
        cv::Mat preprocImage(const std::string &input_bmp_name,
        T *out, int wanted_height,int wanted_width,
        int wanted_channels, float mean, float scale)
        {
            int i;
            uint8_t *pSrc;
            cv::Mat image = cv::imread(input_bmp_name, cv::IMREAD_COLOR);
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            cv::resize(image, image, cv::Size(wanted_width, wanted_height), 0, 0, cv::INTER_AREA);
            if (image.channels() != wanted_channels)
            {
                // LOG(FATAL) << "Warning : Number of channels wanted differs from number of channels in the actual image \n";
                exit(-1);
            }
            pSrc = (uint8_t *)image.data;
            for (i = 0; i < wanted_height * wanted_width * wanted_channels; i++)
                out[i] = ((T)pSrc[i] - mean) / scale;
            return image;
        }

        int test(){
            return 200;
        }
    } // namespace tflite::preprocess
}
