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
                             tidl::modelInfo::PreprocessImageConfig preProcessImageConfig);

    } // namespace tidl::preprocess
}
#endif /* _PRE_PROCESS_H_ */
