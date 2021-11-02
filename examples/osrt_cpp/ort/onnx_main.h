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

#include "../cpp_config.h"
#include "itidl_rt.h"
#include "../post_process/post_process.h"

#define LOG(x) std::cerr

namespace onnx
{
    namespace main
    {
        using namespace std;
        struct Settings
        {
            bool verbose = false;
            bool accel = false;
            bool device_mem = false;
            bool input_floating = false;
            bool profiling = false;
            bool allow_fp16 = false;
            bool gl_backend = false;
            bool hexagon_delegate = false;
            int loop_count = 1;
            std::vector<float> input_mean;
            std::vector<float> input_std;
            string artifact_path = "";
            string model_path = "";
            string input_bmp_name = "";
            string labels_path = "";
            string input_layer_type = "uint8_t";
            int number_of_threads = 4;
            int number_of_results = 5;
            int max_profiling_buffer_entries = 1024;
            int number_of_warmup_runs = 2;
            tidl::config::Modeltype model_type;
        };

        void display_usage()
        {
            LOG(INFO)
                << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
                << "--artifact_path, -f: [0|1], Path for Delegate artifacts folder \n"
                << "--count, -c: loop interpreter->Invoke() for certain times\n"
                << "--image, -i: image_name.bmp\n"
                << "--labels, -l: labels for the model\n"
                << "--onnx_model, -m: model_name.onnx\n"
                << "--threads, -t: number of threads\n"
                << "--verbose, -v: [0|1] print more information\n"
                << "--warmup_runs, -w: number of warmup runs\n"
                << "\n";
        }

        double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

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