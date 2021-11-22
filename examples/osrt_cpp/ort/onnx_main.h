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

#ifndef ONNX_EXAMPLES_MAIN_H_
#define ONNX_EXAMPLES_MAIN_H_

#include <getopt.h>
#include <iostream>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <list>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/tidl/tidl_provider_factory.h>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>

#include "itidl_rt.h"
#include "post_process/post_process.h"
#include "pre_process/pre_process.h"
#include "utils/include/arg_parsing.h"
#include "utils/include/utility_functs.h"
#include "utils/include/ti_logger.h"
#include "utils/include/model_info.h"


namespace onnx
{
    namespace main
    {
        /*
        * Case Sensitive Implementation of endsWith()
        * It checks if the string 'mainStr' ends with given string 'toMatch'
        */
        bool endsWith(const std::string &mainStr, const std::string &toMatch)
        {
            if (mainStr.size() >= toMatch.size() &&
                mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0)
                return true;
            else
                return false;
        }
    } //namespace onnx::main
} //namespace onnx

#endif // ONNX_EXAMPLES_MAIN_H_