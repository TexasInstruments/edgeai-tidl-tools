

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


namespace tflite
{
    namespace preprocess
    {
       template <class T>
        cv::Mat preprocImage(const std::string &input_bmp_name,
        T *out, int wanted_height,int wanted_width,
        int wanted_channels, float mean, float scale);
         
        int test();
    } // namespace tflite::preprocess
}
#endif /* _PRE_PROCESS_H_ */
