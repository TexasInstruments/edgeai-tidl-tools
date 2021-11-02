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
#include "../pre_process/pre_process.h"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

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
            std::cout << "Image" << s->input_bmp_name << std::endl;
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
            // Validator validator(env, model_path, image_path, labels_path, session_options);
            Ort::AllocatorWithDefaultOptions allocator; // ORT Session
            Ort::Session session(env, model_path.c_str(), session_options);

            // Input information
            size_t num_input_nodes = session.GetInputCount();
            std::vector<const char *> input_node_names(num_input_nodes);
            Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_node_dims = tensor_info.GetShape();
            int wanted_channels = input_node_dims[1];
            int wanted_height = input_node_dims[2];
            int wanted_width = input_node_dims[3];

            size_t num_output_nodes = session.GetOutputCount();
            std::vector<const char *> output_node_names(num_output_nodes);
            output_node_names[0] = session.GetOutputName(0, allocator);
            char *output_name = session.GetOutputName(0, allocator);
            type_info = session.GetOutputTypeInfo(0);
            auto output_tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> output_node_dims = output_tensor_info.GetShape();
            size_t output_tensor_size = output_node_dims[1];

            int image_size;

            printf("Number of inputs = %zu\n", num_input_nodes);
            // iterate over all input nodes
            for (int i = 0; i < num_input_nodes; i++)
            {
                // print input node names
                char *input_name = session.GetInputName(i, allocator);
                printf("Input %d : name=%s\n", i, input_name);
                input_node_names[i] = input_name;

                // print input node types
                Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                ONNXTensorElementDataType type = tensor_info.GetElementType();
                printf("Input %d : type=%d\n", i, type);

                // print input shapes/dims
                input_node_dims = tensor_info.GetShape();
                printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
                for (int j = 0; j < input_node_dims.size(); j++)
                {
                    printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
                }
            }

            int num_iter = s->loop_count;
            std::vector<float> image_data(wanted_height * wanted_width * wanted_channels);
            cv::Mat img = tidl::preprocess::preprocImage<float>(image_path, image_data, wanted_height, wanted_width, wanted_channels, s->input_mean, s->input_std);
            size_t input_tensor_size = wanted_channels * wanted_height * wanted_width; // simplify ... using known dim values to calculate size

            // use OrtGetTensorShapeElementCount() to get official size!

            // std::vector<float> input_tensor_values(input_tensor_size);
            printf("Output name -- %s \n", *(output_node_names.data()));

            // create input tensor object from data values
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            void *inData = TIDLRT_allocSharedMem(16, input_tensor_size * sizeof(float));
            if (inData == NULL)
            {
                printf("Could not allocate memory for inData \n ");
                exit(0);
            }
            memcpy(inData, image_data.data(), input_tensor_size * sizeof(float));

            void *outData = TIDLRT_allocSharedMem(16, output_tensor_size * sizeof(float));
            if (outData == NULL)
            {
                printf("Could not allocate memory for outData \n ");
                exit(0);
            }

#if ORT_ZERO_COPY_API
            Ort::IoBinding binding(_session);
            const Ort::RunOptions &runOpts = Ort::RunOptions();
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *)inData, input_tensor_size, _input_node_dims.data(), 4);
            assert(input_tensor.IsTensor());
            std::vector<int64_t> _output_node_dims = {1, 1, 1, 1000};

            Ort::Value output_tensors = Ort::Value::CreateTensor<float>(memory_info, (float *)outData, output_tensor_size, _output_node_dims.data(), 4);
            assert(output_tensors.IsTensor());

            binding.BindInput(_input_node_names[0], input_tensor);
            binding.BindOutput(output_node_names[0], output_tensors);

            struct timeval start_time, stop_time;
            gettimeofday(&start_time, nullptr);
            for (int i = 0; i < num_iter; i++)
            {
                _session.Run(runOpts, binding);
            }
            gettimeofday(&stop_time, nullptr);
            float *floatarr = (float *)outData;

#else
            //Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, image_data.data(), input_tensor_size, _input_node_dims.data(), 4);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *)inData, input_tensor_size, input_node_dims.data(), 4);

            assert(input_tensor.IsTensor());

            // score model & input tensor, get back output tensor
            auto run_options = Ort::RunOptions();
            run_options.SetRunLogVerbosityLevel(2);

            auto output_tensors = session.Run(run_options, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);

            struct timeval start_time, stop_time;
            gettimeofday(&start_time, nullptr);
            for (int i = 0; i < num_iter; i++)
            {
                output_tensors = session.Run(run_options, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
            }
            gettimeofday(&stop_time, nullptr);

            assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
#endif
            std::cout << "invoked \n";
            std::cout << "average time: "
                      << (get_us(stop_time) - get_us(start_time)) / (num_iter * 1000)
                      << " ms \n";


            if(s->model_type == tidl::config::Modeltype::CLF){
                // Get pointer to output tensor float values
                std::vector<std::pair<float, int>> top_results;
                const float threshold = 0.001f;
                float *floatarr = output_tensors.front().GetTensorMutableData<float>();
                tidl::postprocess::get_top_n<float>(floatarr,
                                                1000, s->number_of_results, threshold,
                                                &top_results, false);
                // Determine most common index
                float max_val = 0.0;
                int max_index = 0;
                for (int i = 0; i < 1000; i++)
                {
                    if (floatarr[i] > max_val)
                    {
                        max_val = floatarr[i];
                        max_index = i;
                    }
                }
                std::cout << "MAX: class [" << max_index << "] = " << max_val << std::endl;
                std::vector<string> labels;
                size_t label_count;
                if (tidl::postprocess::ReadLabelsFile(s->labels_path, &labels, &label_count) != 0)
                    exit(-1);
                // std::cout << labels[max_index + 1] << std::endl;
                for (const auto &result : top_results)
                {
                    const float confidence = result.first;
                    const int index = result.second;
                    LOG(INFO) << confidence << ": " << index << " " << labels[index] << "\n";
                }
                int num_results = 5;
                img.data = tidl::postprocess::overlayTopNClasses(img.data, top_results, &labels, img.cols, img.rows, num_results);
      
            }
            else if(s->model_type == tidl::config::Modeltype::SEG){
                // Get pointer to output tensor float values
                int64_t* int64arr = output_tensors.front().GetTensorMutableData<int64_t>();
                int32_t* int32arr =(int32_t*) malloc(512*512*sizeof(int32_t));
                float alpha = 0.4f; 
                for (int i = 0; i < 512*512; i++)
                {
                    int32arr[i] = (int32_t) int64arr[i];
                }
                tidl::postprocess::blendSegMask(img.data, int32arr, img.cols, img.rows, wanted_width, wanted_height, alpha);
            }
            else if(s->model_type == tidl::config::Modeltype::OD){
                // Get pointer to output tensor float values
                float* floatarr = output_tensors.front().GetTensorMutableData<float>();
                std::list<float> detection_class_list,detectection_location_list,detectection_scores_list;
                
                //parsing 
                int num_detections = 0;                
                for (int i = 4; i < 1000; i = i+5)
                {
                    if(floatarr[i] > .3){
                        num_detections++;
                        detectection_scores_list.push_back(floatarr[i]);
                        detection_class_list.push_back((i-4)/5);
                        detectection_location_list.push_back(floatarr[i-1]/img.cols);
                        detectection_location_list.push_back(floatarr[i-2]/img.cols);
                        detectection_location_list.push_back(floatarr[i-3]/img.cols);
                        detectection_location_list.push_back(floatarr[i-4]/img.cols);
                    }
                }
                float detectection_scores[detectection_scores_list.size()];
                float detection_class[detection_class_list.size()];
                float detectection_location[detectection_location_list.size()];

                std::copy(detectection_scores_list.begin(), detectection_scores_list.end(), detectection_scores);
                std::copy(detection_class_list.begin(), detection_class_list.end(), detection_class);
                std::copy(detectection_location_list.begin(), detectection_location_list.end(), detectection_location);

     
                LOG(INFO) << "results " << num_detections << "\n";
                tidl::postprocess::overlayBoundingBox(img, num_detections, detectection_location);
                for (int i = 0; i < num_detections; i++)
                {
                LOG(INFO) << "class " << detection_class[i] << "\n";
                LOG(INFO) << "cordinates " << detectection_location[i * 4] << detectection_location[i * 4 + 1] << detectection_location[i * 4 + 2] << detectection_location[i * 4 + 3] << "\n";
                LOG(INFO) << "score " << detectection_scores[i] << "\n";
                }
            }  
                cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
                char filename[100];
                strcpy(filename, s->artifact_path.c_str());
                strcat(filename, "cpp_inference_out.jpg");
                bool check = cv::imwrite(filename, img);
                if (check == false)
                {
                    std::cout << "Saving the image, FAILED" << std::endl;
                }
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
