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

#ifndef _PRE_PROCESS_H_
#define _PRE_PROCESS_H_

/* Third-party headers. */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>

#include <stdint.h>
#include <string>

/* module headers. */
#include "../utils/include/model_info.h"

namespace tidl
{
    namespace preprocess
    {
        template <class T>
        /**
  *  \brief Use OpenCV to open an image and resize according to requirment of model, 
  * scalar modification on based on mean and scale
  *
  * @param input_bmp_name
  * @returns original frame with some in-place post processing done
  */
        cv::Mat preprocImage(const std::string &input_bmp_name,
                             T *out,
                             tidl::modelInfo::PreprocessImageConfig preProcessImageConfig)
        {
            int i;
            uint8_t *pSrc;
            cv::Mat spl[3];
            int wanted_width = preProcessImageConfig.outDataWidth,
                wanted_height = preProcessImageConfig.outDataHeight,
                wanted_channels = preProcessImageConfig.numChans;
            std::vector<float> mean = preProcessImageConfig.mean;
            std::vector<float> scale = preProcessImageConfig.scale;
            cv::Mat image = cv::imread(input_bmp_name, cv::IMREAD_COLOR);
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            cv::resize(image, image, cv::Size(wanted_width, wanted_height), 0, 0, cv::INTER_AREA);
            if (image.channels() != wanted_channels)
            {
                printf("Warning : Number of channels wanted differs from number of channels in the actual image \n");
                exit(-1);
            }
            cv::split(image, spl);
            if (!strcmp(preProcessImageConfig.dataLayout.c_str(), "NHWC"))
            {
                std::cout << "template NHWC\n";
                for (i = 0; i < wanted_height * wanted_width; i++)
                {
                    for (int j = 0; j < wanted_channels; j++)
                    {
                        pSrc = (uint8_t *)spl[j].data;
                        out[i * wanted_channels + j] = ((T)pSrc[i] - mean[j]) * scale[j];
                    }
                }
            }
            else if (!strcmp(preProcessImageConfig.dataLayout.c_str(), "NCHW"))
            {
                std::cout << "template NCHW\n";
                for (int j = 0; j < wanted_channels; j++)
                {
                    pSrc = (uint8_t *)spl[j].data;
                    for (i = 0; i < wanted_height * wanted_width; i++)
                    {
                        out[j * (wanted_height * wanted_width) + i] = ((T)pSrc[i] - mean[j]) * scale[j];
                    }
                }
            }

            return image;
        }

        template cv::Mat preprocImage<uint8_t>(const std::string &input_bmp_name,
                                               uint8_t *out,
                                               tidl::modelInfo::PreprocessImageConfig preProcessImageConfig);

        template cv::Mat preprocImage<float>(const std::string &input_bmp_name,
                                             float *out,
                                             tidl::modelInfo::PreprocessImageConfig preProcessImageConfig);

    } // namespace tidl::preprocess
}
#endif /* _PRE_PROCESS_H_ */
