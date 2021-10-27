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

#include "onnx_main.h"

namespace onnx
{
    namespace main
    {
        int RunInference(onnx::main::Settings *s)
        {
            std::string model_path = s->model_path;
            std::string image_path = s->input_bmp_name;
            std::string labels_path = s->labels_path;
            std::string artifacts_path = s->artifact_path;

            int tidl_flag = s->accel;
            
            std::cout << "Running model" << s->model_path << std::endl;
            std::cout << "Image" <<s->input_bmp_name << std::endl;
            std::cout << "Artifacts path" << s->artifact_path << std::endl;
            std::cout << "Labels path" << s->labels_path << std::endl;

            // Initialize  enviroment, maintains thread pools and state info
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

            // Initialize session options
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);

            c_api_tidl_options *options = (c_api_tidl_options *)malloc(sizeof(c_api_tidl_options));
            options->debug_level = 0;
            strcpy(options->artifacts_folder, artifacts_path.c_str());

            if (tidl_flag)
            {
                OrtSessionOptionsAppendExecutionProvider_Tidl(session_options, options);
            }
            else
            {
                OrtSessionOptionsAppendExecutionProvider_CPU(session_options, false);
            }

            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            // Do the thing
            Validator validator(env, model_path, image_path, labels_path, session_options);

            printf(" Done!\n");
            return 0;
        }

        int ONNX_Main(int argc, char **argv)
        {
            Settings s;
            int c;
            while (1)
            {
                static struct option long_options[] = {
                    {"accelerated", required_argument, nullptr, 'a'},
                    {"count", required_argument, nullptr, 'c'},
                    {"verbose", required_argument, nullptr, 'v'},
                    {"threads", required_argument, nullptr, 't'},
                    {"warmup_runs", required_argument, nullptr, 'w'},
                    {nullptr, 0, nullptr, 0}};

                /* getopt_long stores the option index here. */
                int option_index = 0;

                c = getopt_long(argc, argv,
                                "a:c:t:v:w:", long_options,
                                &option_index);

                /* Detect the end of the options. */
                if (c == -1)
                    break;

                switch (c)
                {
                case 'a':
                    s.accel = strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
                    break;
                case 'c':
                    s.loop_count =
                        strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
                    break;
                case 't':
                    s.number_of_threads = strtol( // NOLINT(runtime/deprecated_fn)
                        optarg, nullptr, 10);
                    break;
                case 'v':
                    s.verbose =
                        strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
                    break;
                case 'w':
                    s.number_of_warmup_runs =
                        strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
                    break;
                case 'h':
                case '?':
                    /* getopt_long already printed an error message. */
                    onnx::main::display_usage();
                    exit(-1);
                default:
                    exit(-1);
                }
            }
            for (int i = 0; i < NUM_CONFIGS; i++)
            {
                bool isOnnxModel = endsWith(tidl::config::model_configs[i].model_path, "onnx");
                if (isOnnxModel)
                {
                    s.artifact_path = tidl::config::model_configs[i].artifact_path;
                    s.model_path = tidl::config::model_configs[i].model_path;
                    s.labels_path = tidl::config::model_configs[i].labels_path;
                    s.input_bmp_name = tidl::config::model_configs[i].image_path;
                    s.input_mean = tidl::config::model_configs[i].mean;
                    s.input_std = tidl::config::model_configs[i].std;
                    s.model_type = tidl::config::model_configs[i].model_type;
                    RunInference(&s);
                }
            }

            return 0;
        }

    } //onnx::main
} //onnx::

int main(int argc, char *argv[])
{
    return onnx::main::ONNX_Main(argc, argv);
}
