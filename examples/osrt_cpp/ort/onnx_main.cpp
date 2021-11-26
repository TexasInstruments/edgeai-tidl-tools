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
        int RunInference(tidl::modelInfo::ModelInfo *modelInfo, tidl::arg_parsing::Settings *s)
        {
            std::string model_path = modelInfo->m_infConfig.modelFile;
            std::string image_path = s->input_bmp_path;
            std::string labels_path = s->labels_file_path;
            std::string artifacts_path;
            /*check artifacts path need to be overwritten from cmd line args */
            if (strcmp(s->artifact_path.c_str(), ""))
                artifacts_path = s->artifact_path;
            else
                artifacts_path = modelInfo->m_infConfig.artifactsPath;
            cv::Mat img;
            void *inData;

            /* checking model path present or not*/
            if (!modelInfo->m_infConfig.modelFile.c_str())
            {
                printf("no model file name\n");
                exit(-1);
            }

            /* Initialize  enviroment, maintains thread pools and state info */
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
            /* Initialize session options */
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);

            c_api_tidl_options *options = (c_api_tidl_options *)malloc(sizeof(c_api_tidl_options));

            strcpy(options->artifacts_folder, artifacts_path.c_str());
            if (s->accel)
            {
                printf("accelerated mode\n");
                OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_Tidl(session_options, options);
            }
            else
            {
                OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_CPU(session_options, false);
            }

            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            Ort::AllocatorWithDefaultOptions allocator;
            /* ORT Session */
            Ort::Session session(env, model_path.c_str(), session_options);
            printf("Loaded model %s\n", modelInfo->m_infConfig.modelFile.c_str());
            /* Input information */
            size_t num_input_nodes = session.GetInputCount();
            std::vector<const char *> input_node_names(num_input_nodes);
            Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_node_dims = tensor_info.GetShape();
            ONNXTensorElementDataType input_tensor_type = tensor_info.GetElementType();
            int wanted_height = input_node_dims[2];
            int wanted_width = input_node_dims[3];
            int wanted_channels = input_node_dims[1];
            /* assuming NCHW*/
            if (wanted_channels != modelInfo->m_preProcCfg.numChans)
            {
                LOG(INFO) << "missmatch in YAML parsed wanted channels and model " << wanted_channels << " " << input_node_dims[1] << "\n";
                ;
            }
            if (wanted_height != modelInfo->m_preProcCfg.outDataHeight)
            {
                LOG(INFO) << "missmatch in YAML parsed wanted height and model " << wanted_height << " " << input_node_dims[2] << "\n";
            }
            if (wanted_width != modelInfo->m_preProcCfg.outDataWidth)
            {
                LOG(INFO) << "missmatch in YAML parsed wanted width and model " << wanted_width << " " << input_node_dims[3] << "\n";
                ;
            }
            /* output information */
            size_t num_output_nodes = session.GetOutputCount();
            std::vector<const char *> output_node_names(num_output_nodes);
            for (int i = 0; i < num_output_nodes; i++)
            {
              output_node_names[i] = session.GetOutputName(i, allocator);
            }
            type_info = session.GetOutputTypeInfo(0);
            auto output_tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> output_node_dims = output_tensor_info.GetShape();
            size_t output_tensor_size = output_node_dims[1];

            LOG(INFO) << "number of inputs: " << num_input_nodes << "\n";
            LOG(INFO) << "number of outputs: " << num_output_nodes << "\n";
            LOG(INFO) << "input(0) name: " << input_node_names[0] << "\n";
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
            if (num_input_nodes != 1)
            {
                LOG(INFO) << "supports only single input model \n";
                exit(1);
            }
            int num_iter = s->loop_count;
            size_t input_tensor_size = wanted_channels * wanted_height * wanted_width; // simplify ... using known dim values to calculate size
            if (input_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                inData = TIDLRT_allocSharedMem(32, input_tensor_size * sizeof(float));
                if (inData == NULL)
                {
                    printf("Could not allocate memory for inData \n ");
                    exit(0);
                }
                img = tidl::preprocess::preprocImage<float>(image_path, (float *)inData, modelInfo->m_preProcCfg);
            }
            else
            {
                printf("indata type not supported yet \n ");
                exit(0);
            }
            printf("Output name -- %s \n", *(output_node_names.data()));

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            printf("create cpu done\n");

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
            ssize_t input_tensor_size_bytes;
            if (input_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                input_tensor_size_bytes = input_tensor_size * sizeof(float);
            }
            /* add further input types here */
            else
            {
                printf("in data type not supported\n");
                exit(1);
            }
            Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, inData, input_tensor_size_bytes, input_node_dims.data(), 4, input_tensor_type);
            assert(input_tensor.IsTensor());
            /* score model & input tensor, get back output tensor */
            auto run_options = Ort::RunOptions();
            run_options.SetRunLogVerbosityLevel(2);
            std::vector<Ort::Value> output_tensors;
            if (s->loop_count >= 1)
            {
                printf("Session.Run() - Started for warmup runs\n");
                for (int i = 0; i < s->number_of_warmup_runs; i++)
                {
                    output_tensors = session.Run(run_options, input_node_names.data(), &input_tensor, num_input_nodes, output_node_names.data(),num_output_nodes);
                }
            }
            struct timeval start_time, stop_time;
            gettimeofday(&start_time, nullptr);
            for (int i = 0; i < num_iter; i++)
            {
                output_tensors = session.Run(run_options, input_node_names.data(), &input_tensor, num_input_nodes, output_node_names.data(), num_output_nodes);
            }
            gettimeofday(&stop_time, nullptr);
            assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
#endif

            printf("invoked\n");
            printf("average time: %lf ms \n",
                   (tidl::utility_functs::get_us(stop_time) - tidl::utility_functs::get_us(start_time)) / (num_iter * 1000));
            /* Get output tensor type */
            ONNXTensorElementDataType op_tensor_type = output_tensors.front()
                                                           .GetTypeInfo()
                                                           .GetTensorTypeAndShapeInfo()
                                                           .GetElementType();

            if (!strcmp(modelInfo->m_preProcCfg.taskType.c_str(), "classification"))
            {
                printf("preparing classification result \n");
                /* Get pointer to output tensor float values*/
                std::vector<std::pair<float, int>> top_results;
                const float threshold = 0.001f;
                /* assuming output tensor of size [1,1000] or [1, 1001]*/
                int output_size = output_node_dims.data()[output_node_dims.size() - 1];
                int outputoffset;
                if (output_size == 1001)
                    outputoffset = 0;
                else
                    outputoffset = 1;
                // Determine most common index
                float max_val = 0.0;
                int max_index = 0;
                if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
                {
                    int64_t *int64arr = output_tensors.front().GetTensorMutableData<int64_t>();
                    tidl::postprocess::get_top_n<int64_t>(int64arr,
                                                          output_size, s->number_of_results, threshold,
                                                          &top_results, true);
                    for (int i = 0; i < output_size; i++)
                    {
                        if (int64arr[i] > max_val)
                        {
                            max_val = int64arr[i];
                            max_index = i;
                        }
                    }
                }
                else if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                {
                    float *floatarr = output_tensors.front().GetTensorMutableData<float>();
                    tidl::postprocess::get_top_n<float>(floatarr,
                                                        output_size, s->number_of_results, threshold,
                                                        &top_results, true);
                    for (int i = 0; i < output_size; i++)
                    {
                        if (floatarr[i] > max_val)
                        {
                            max_val = floatarr[i];
                            max_index = i;
                        }
                    }
                }
                else
                {
                    printf("out data type not supported yet \n ");
                    exit(0);
                }

                std::vector<std::string> labels;
                size_t label_count;
                if (tidl::postprocess::ReadLabelsFile(s->labels_file_path, &labels, &label_count) != 0)
                    exit(-1);
                printf("MAX: class [%d] = class Name  %s Max val: %lf\n", max_index, labels[max_index + outputoffset].c_str(), max_val);
                for (const auto &result : top_results)
                {
                    const float confidence = result.first;
                    const int index = result.second;
                    std::cout << confidence << ": " << index << " " << labels[index + 1] << "\n";
                }
                int num_results = 5;
                img.data = tidl::postprocess::overlayTopNClasses(img.data, top_results, &labels, img.cols, img.rows, num_results);
            }
            else if (!strcmp(modelInfo->m_preProcCfg.taskType.c_str(), "segmentation"))
            {
                printf("preparing segmentation result \n");
                void *tensor_op_array;
                if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
                {
                    tensor_op_array = output_tensors.front().GetTensorMutableData<int64_t>();
                }
                else if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                {
                    tensor_op_array = output_tensors.front().GetTensorMutableData<float>();
                }
                else
                {
                    printf("out data type not supported\n");
                    exit(1);
                }
                float alpha = 0.4f;
                tidl::postprocess::blendSegMask(img.data, tensor_op_array, tidl::modelInfo::DlInferType_Int64, img.cols, img.rows, wanted_width, wanted_height, alpha);
            }
            else if (!strcmp(modelInfo->m_preProcCfg.taskType.c_str(), "detection"))
            {
                printf("preparing detection result \n");
                std::vector<int32_t> format = {0, 1, 2, 3, 4, 5};
                if (tidl::utility_functs::is_same_format(format, modelInfo->m_postProcCfg.formatter))
                {
                    printf("format found\n");
                    /* assuming three outputs: bboxes [1,nboxes,4] , labels [1,nboxes], score[1,nboxes] */
                    float *bboxes = output_tensors.at(0).GetTensorMutableData<float>();
                    std::cout << output_tensors.size()<< ":size\n";
                    int64_t *labels = output_tensors.at(1).GetTensorMutableData<int64_t>();
                    float *scores = output_tensors.at(2).GetTensorMutableData<float>();
                    int nboxes = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape()[1];
                    for (int i = 0; i < 200; i=i+4)
                    {
                        printf("%f %f %f %f\n",bboxes[i], bboxes[i+1], bboxes[i+2], bboxes[i+3]);
                    }
                    for (int i = 0; i < 200; i=i+4)
                    {
                        printf("%f\n",scores[i]);
                    }
                    for (int i = 0; i < 200; i=i+4)
                    {
                        printf("%ld\n",labels[i]);
                    }
                    
                    exit(1);
                }
                format = {0, 1, 2, 3, 5, 4};
                if (tidl::utility_functs::is_same_format(format, modelInfo->m_postProcCfg.formatter))
                {
                    std::list<float> detection_class_list, detectection_location_list, detectection_scores_list;
                    float detectection_scores[detectection_scores_list.size()];
                    float detection_class[detection_class_list.size()];
                    float detectection_location[detectection_location_list.size()];
                    int num_detections = 0;

                    if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
                    {
                        int64_t *int64arr = output_tensors.front().GetTensorMutableData<int64_t>();
                        //parsing
                        for (int i = 4; i < 1000; i = i + 5)
                        {
                            if (int64arr[i] > .3)
                            {
                                num_detections++;
                                detectection_scores_list.push_back(int64arr[i]);
                                detection_class_list.push_back((i - 4) / 5);
                                detectection_location_list.push_back(int64arr[i - 1] / img.cols);
                                detectection_location_list.push_back(int64arr[i - 2] / img.cols);
                                detectection_location_list.push_back(int64arr[i - 3] / img.cols);
                                detectection_location_list.push_back(int64arr[i - 4] / img.cols);
                            }
                        }
                    }
                    else if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                    {
                        float *floatarr = output_tensors.front().GetTensorMutableData<float>();
                        //parsing
                        for (int i = 4; i < 1000; i = i + 5)
                        {
                            if (floatarr[i] > .3)
                            {
                                num_detections++;
                                detectection_scores_list.push_back(floatarr[i]);
                                detection_class_list.push_back((i - 4) / 5);
                                detectection_location_list.push_back(floatarr[i - 1] / img.cols);
                                detectection_location_list.push_back(floatarr[i - 2] / img.cols);
                                detectection_location_list.push_back(floatarr[i - 3] / img.cols);
                                detectection_location_list.push_back(floatarr[i - 4] / img.cols);
                            }
                        }
                    }
                    else
                    {
                        printf("out data type not supported\n");
                        exit(1);
                    }

                    std::copy(detectection_scores_list.begin(), detectection_scores_list.end(), detectection_scores);
                    std::copy(detection_class_list.begin(), detection_class_list.end(), detection_class);
                    std::copy(detectection_location_list.begin(), detectection_location_list.end(), detectection_location);

                    printf("results %d\n", num_detections);
                    float threshold = 0.35f;
                    tidl::postprocess::overlayBoundingBox(img, num_detections, detectection_location, detectection_scores, threshold);
                    for (int i = 0; i < num_detections; i++)
                    {
                        printf("class %lf\n", detection_class[i]);
                        printf("cordinates %lf %lf %lf %lf\n", detectection_location[i * 4], detectection_location[i * 4 + 1], detectection_location[i * 4 + 2], detectection_location[i * 4 + 3]);
                        printf("score %lf\n", detectection_scores[i]);
                    }
                }
            }
            cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            char filename[100];
            strcpy(filename, "/home/a0496663/edgeai-tidl-tools/test_data/");
            strcat(filename, "cpp_inference_out.jpg");
            bool check = cv::imwrite(filename, img);
            if (check == false)
            {
                LOG(INFO) << "Saving the image, FAILED" << std::endl;
            }
            LOG(INFO) << "Done!\n";
            return 0;
        }

    }
}

int main(int argc, char *argv[])
{
    tidl::arg_parsing::Settings s;
    tidl::arg_parsing::parse_args(argc, argv, &s);
    tidl::arg_parsing::dump_args(&s);
    tidl::utils::logSetLevel((tidl::utils::LogLevel)s.log_level);
    // Parse the input configuration file
    tidl::modelInfo::ModelInfo model(s.model_zoo_path);
    int status = model.initialize();
    onnx::main::RunInference(&model, &s);
    return 0;
}
