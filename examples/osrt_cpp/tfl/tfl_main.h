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

#ifndef TENSORFLOW_LITE_EXAMPLES_MAIN_H_
#define TENSORFLOW_LITE_EXAMPLES_MAIN_H_

#include <fcntl.h>
#include <sys/time.h>
#include <sys/types.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <yaml-cpp/yaml.h>

#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/profiling/profiler.h>
#include <tensorflow/lite/string_util.h>
#include <dlfcn.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <tensorflow/lite/tools/evaluation/utils.h>

#include "itidl_rt.h"
#include "../cpp_config.h"
#include "post_process/post_process.h"
#include "pre_process/pre_process.h"
#include "utils/include/arg_parsing.h"
#include "utils/include/utility_functs.h"
#include "utils/include/ti_logger.h"
#include "utils/include/model_info.h"

namespace tflite
{
    namespace main
    {

        /**
  *  \brief  Case Sensitive Implementation of endsWith for a string
  * It checks if the string 'mainStr' ends with given string 'toMatch'
  * @returns bool
  */
        bool endsWith(const std::string &mainStr, const std::string &toMatch)
        {
            if (mainStr.size() >= toMatch.size() &&
                mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0)
                return true;
            else
                return false;
        }
    } //namespace tflite::main
} //namespace tflite

#endif // TENSORFLOW_LITE_EXAMPLES_MAIN_H_