/*
Copyright (c) 2020 – 2021 Texas Instruments Incorporated

All rights reserved not granted herein.

Limited License.

Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
license under copyrights and patents it now or hereafter owns or controls to
make, have made, use, import, offer to sell and sell ("Utilize") this software
subject to the terms herein.  With respect to the foregoing patent license,
such license is granted  solely to the extent that any such patent is necessary
to Utilize the software alone.  The patent license shall not apply to any
combinations which include this software, other than combinations with devices
manufactured by or for TI (“TI Devices”).  No hardware patent is licensed
hereunder.

Redistributions must preserve existing copyright notices and reproduce this
license (including the above copyright notice and the disclaimer and
(if applicable) source code license limitations below) in the documentation
and/or other materials provided with the distribution

Redistribution and use in binary form, without modification, are permitted
provided that the following conditions are met:

*	No reverse engineering, decompilation, or disassembly of this software is
    permitted with respect to any software provided in binary form.

*	any redistribution and use are licensed by TI for use only with TI Devices.

*	Nothing shall obligate TI to provide you with source code for the software
    licensed and provided to you in object code.

If software source code is provided to you, modification and redistribution of
the source code are permitted provided that the following conditions are met:

*	any redistribution and use of the source code, including any resulting
    derivative works, are licensed by TI for use only with TI Devices.

*	any redistribution and use of any object code compiled from the source code
    and any resulting derivative works, are licensed by TI for use only with TI
    Devices.

Neither the name of Texas Instruments Incorporated nor the names of its
suppliers may be used to endorse or promote products derived from this software
without specific prior written permission.

DISCLAIMER.

THIS SOFTWARE IS PROVIDED BY TI AND TI’S LICENSORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL TI AND TI’S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

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
        /**
  *  \brief Use OpenCV to open an image and resize according to requirment 
  * of model, scalar modification on based on mean and scale
  *
  * @param input_bmp_name
  * @param out out data array
  * @param preProcessImageConfig prepprocess image config parsed from YAML
  * @returns original frame with some in-place post processing done
  */
        template <class T>
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
                LOG_ERROR("Warning : Number of channels wanted differs from number of channels in the actual image \n");
                exit(-1);
            }
            cv::split(image, spl);
            if (!strcmp(preProcessImageConfig.dataLayout.c_str(), "NHWC"))
            {
                LOG_INFO("template NHWC\n");
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
                LOG_INFO("template NCHW\n");
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
#endif // _PRE_PROCESS_H_
