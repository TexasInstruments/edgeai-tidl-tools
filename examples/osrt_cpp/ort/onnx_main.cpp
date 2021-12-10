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

#include "onnx_main.h"

namespace onnx
{
    namespace main
    {

        /**
         *  \brief  get the tensor type
         *  \param  index  index of tensor
         *  \param  output_tensors pointer of tflite
         * @returns int status
         */
        ONNXTensorElementDataType getTensorType(int index, std::vector<Ort::Value> *output_tensors)
        {
            /* Get tensor type */
            return (*output_tensors).at(index).GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType();
        }

        /**
         *  \brief  prepare the classification result inplace
         *  \param  img cv image to do inplace transform
         *  \param  output_tensors pointer of tflite
         *  \param  s settings struct pointer
         *  \param  output_node_dims
         * @returns int status
         */
        int prepClassificationResult(cv::Mat *img, std::vector<Ort::Value> *output_tensors, Settings *s,
                                     std::vector<int64_t> output_node_dims)
        {
            LOG_INFO("preparing classification result \n");
            ONNXTensorElementDataType op_tensor_type = getTensorType(0, output_tensors);
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
            /* Determine most common index */
            float max_val = 0.0;
            int max_index = 0;
            if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
            {
                int64_t *int64arr = (*output_tensors).front().GetTensorMutableData<int64_t>();
                getTopN<int64_t>(int64arr,
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
                float *floatarr = (*output_tensors).front().GetTensorMutableData<float>();
                getTopN<float>(floatarr,
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
                LOG_ERROR("out data type not supported yet \n ");
                return RETURN_FAIL;
            }

            std::vector<std::string> labels;
            size_t label_count;
            if (RETURN_FAIL == readLabelsFile(s->labels_file_path, &labels, &label_count) != 0)
            {
                LOG_ERROR("failed to read label file");
                return RETURN_FAIL;
            }
            LOG_INFO("MAX: class [%d] = class Name  %s Max val: %lf\n", max_index, labels[max_index + outputoffset].c_str(), max_val);
            for (const auto &result : top_results)
            {
                const float confidence = result.first;
                const int index = result.second;
                LOG_INFO("%f: %d %s\n", confidence, index, labels[index + outputoffset].c_str());
            }
            int num_results = 5;
            (*img).data = overlayTopNClasses((*img).data, top_results, &labels, (*img).cols, (*img).rows, num_results);
            return RETURN_SUCCESS;
        }

        /**
         *  \brief  prepare the segemntataion result inplace
         *  \param  img cv image to do inplace transform
         *  \param  output_tensors pointer of tflite
         *  \param  s settings struct pointer
         *  \param  alpha for img masking
         * @returns int status
         */
        int prepSegResult(cv::Mat *img, std::vector<Ort::Value> *output_tensors, Settings *s, float alpha)
        {
            LOG_INFO("preparing segmentation result \n");
            ONNXTensorElementDataType op_tensor_type = getTensorType(0, output_tensors);
            /* if indata and out data is diff resize the image check
            whether img need to be resized based on out data asssuming
            out put format [1,1,,width,height]*/
            int wanted_height = (*output_tensors).front().GetTensorTypeAndShapeInfo().GetShape()[2];
            int wanted_width = (*output_tensors).front().GetTensorTypeAndShapeInfo().GetShape()[3];
            cv::resize((*img), (*img), cv::Size(wanted_width, wanted_height), 0, 0, cv::INTER_AREA);
            if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
            {
                int64_t *tensor_op_array = (*output_tensors).front().GetTensorMutableData<int64_t>();
                (*img).data = blendSegMask<int64_t>((*img).data, tensor_op_array, (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
            }
            else if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                float *tensor_op_array = (*output_tensors).front().GetTensorMutableData<float>();
                int nclasses = (*output_tensors).at(0).GetTensorTypeAndShapeInfo().GetShape()[1];
                int nwidth = (*output_tensors).at(0).GetTensorTypeAndShapeInfo().GetShape()[2];
                int nheight = (*output_tensors).at(0).GetTensorTypeAndShapeInfo().GetShape()[3];
                LOG_INFO("nclasses :%d\n", nclasses);
                /* if op is of type [ 1, 1 , width, height ] ie, classwise
                array is merged by model */
                if (nclasses == 1)
                {
                    (*img).data = blendSegMask<float>((*img).data, tensor_op_array, (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
                }
                /* if op is of type [ 1, nclasses , width, height ] ie, classwise
                array is not  merged by model and need to be merged */
                else
                {
                    float max_val = tensor_op_array[512 * 512 * 1];
                    int max_class;
                    float arr[512 * 512] = {
                        0,
                    };
                    /* to avoid the background class start from class 1 of nclasses */
                    for (int i = 0; i < nwidth * nheight; i++)
                    {
                        float max_val = 0;
                        int max_class = 0;
                        for (int j = 0; j < nclasses; j++)
                        {
                            if (tensor_op_array[i + j * nwidth * nheight] >= max_val)
                            {
                                max_val = tensor_op_array[i + j * nwidth * nheight];
                                max_class = j;
                            }
                        }
                        arr[i] = max_class;
                    }
                    (*img).data = blendSegMask<float>((*img).data, arr, (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
                }
            }
            else
            {
                LOG_INFO("output data type not supported\n");
                return RETURN_FAIL;
            }
            return RETURN_SUCCESS;
        }

        /**
         *  \brief  prepare the od result inplace
         *  \param  img cv image to do inplace transform
         *  \param  output_tensors pointer of tflite
         *  \param  modelInfo
         * @returns int status
         */
        int prepDetectionResult(cv::Mat *img, std::vector<Ort::Value> *output_tensors, ModelInfo *modelInfo)
        {
            LOG_INFO("preparing detection result \n");
            ONNXTensorElementDataType op_tensor_type = getTensorType(0, output_tensors);
            std::vector<int32_t> format = {0, 1, 2, 3, 4, 5};
            float threshold = modelInfo->m_vizThreshold;
            /*verify this condition TODO */
            if ((*output_tensors).size() == 3 && isSameFormat(format, modelInfo->m_postProcCfg.formatter))
            {
                /* assuming three outputs:
                bboxes [1,nboxes,4] , labels [1,nboxes], score[1,nboxes] */
                /* How to confirm the type TODO auto var??
                output_tensors.at(0).GetTypeInfo().GetTensorTypeAndShapeInfo()
                .GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT */
                float *bboxes = (*output_tensors).at(0).GetTensorMutableData<float>();
                int64_t *labels = (*output_tensors).at(1).GetTensorMutableData<int64_t>();
                float *scores = (*output_tensors).at(2).GetTensorMutableData<float>();

                std::list<float> detection_class_list, detectection_location_list, detectection_scores_list;
                int num_detections = 0;
                int nboxes = (*output_tensors).front().GetTensorTypeAndShapeInfo().GetShape()[1];
                for (int i = 0; i < nboxes; i = i + 4)
                {
                    if (scores[i] >= threshold)
                    {
                        num_detections++;
                        detectection_scores_list.push_back(scores[i]);
                        detection_class_list.push_back(labels[i]);
                        detectection_location_list.push_back(bboxes[i + 3]);
                        detectection_location_list.push_back(bboxes[i + 2]);
                        detectection_location_list.push_back(bboxes[i + 1]);
                        detectection_location_list.push_back(bboxes[i]);
                    }
                }
                float detectection_scores[detectection_scores_list.size()];
                float detection_class[detection_class_list.size()];
                float detectection_location[detectection_location_list.size()];
                std::copy(detectection_scores_list.begin(), detectection_scores_list.end(), detectection_scores);
                std::copy(detection_class_list.begin(), detection_class_list.end(), detection_class);
                std::copy(detectection_location_list.begin(), detectection_location_list.end(), detectection_location);

                LOG_INFO("results %d\n", num_detections);
                overlayBoundingBox((*img), num_detections, detectection_location, detectection_scores, threshold);
                for (int i = 0; i < num_detections; i++)
                {
                    LOG_INFO("class %lf\n", detection_class[i]);
                    LOG_INFO("cordinates %lf %lf %lf %lf\n", detectection_location[i * 4], detectection_location[i * 4 + 1], detectection_location[i * 4 + 2], detectection_location[i * 4 + 3]);
                    LOG_INFO("score %lf\n", detectection_scores[i]);
                }
            }
            format = {0, 1, 2, 3, 5, 4};
            if ((*output_tensors).size() == 2 && isSameFormat(format, modelInfo->m_postProcCfg.formatter))
            {

                int num_detections = 0;
                float threshold = modelInfo->m_vizThreshold;
                std::list<float> detection_class_list, detectection_location_list, detectection_scores_list;
                if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
                {
                    int64_t *int64arr = (*output_tensors).front().GetTensorMutableData<int64_t>();
                    int64_t *labels = (*output_tensors).at(1).GetTensorMutableData<int64_t>();
                    int nboxes = (*output_tensors).at(0).GetTensorTypeAndShapeInfo().GetShape()[0];
                    int elem_per_box = (*output_tensors).at(0).GetTensorTypeAndShapeInfo().GetShape()[1];
                    /*parsing */
                    for (int i = elem_per_box - 1; i < nboxes * elem_per_box; i = i + elem_per_box)
                    {
                        if (int64arr[i] > threshold)
                        {
                            num_detections++;
                            detectection_scores_list.push_back(int64arr[i]);
                            detection_class_list.push_back(labels[i / elem_per_box]);
                            detectection_location_list.push_back(int64arr[i - 1] / (*img).cols);
                            detectection_location_list.push_back(int64arr[i - 2] / (*img).cols);
                            detectection_location_list.push_back(int64arr[i - 3] / (*img).cols);
                            detectection_location_list.push_back(int64arr[i - 4] / (*img).cols);
                        }
                    }
                }
                else if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                {
                    float *floatarr = (*output_tensors).at(0).GetTensorMutableData<float>();
                    int64_t *labels = (*output_tensors).at(1).GetTensorMutableData<int64_t>();
                    int nboxes = (*output_tensors).at(0).GetTensorTypeAndShapeInfo().GetShape()[0];
                    int elem_per_box = (*output_tensors).at(0).GetTensorTypeAndShapeInfo().GetShape()[1];
                    /*parsing */
                    for (int i = elem_per_box - 1; i < nboxes * elem_per_box; i = i + elem_per_box)
                    {
                        /*assuming last element is score */
                        if (floatarr[i] > threshold)
                        {
                            num_detections++;
                            detectection_scores_list.push_back(floatarr[i]);
                            detection_class_list.push_back(labels[i / elem_per_box]);
                            detectection_location_list.push_back(floatarr[i - 1] / (*img).cols);
                            detectection_location_list.push_back(floatarr[i - 2] / (*img).cols);
                            detectection_location_list.push_back(floatarr[i - 3] / (*img).cols);
                            detectection_location_list.push_back(floatarr[i - 4] / (*img).cols);
                        }
                    }
                }
                else
                {
                    LOG_INFO("out data type not supported\n");
                    return RETURN_FAIL;
                }
                float detectection_scores[detectection_scores_list.size()];
                float detection_class[detection_class_list.size()];
                float detectection_location[detectection_location_list.size()];
                std::copy(detectection_scores_list.begin(), detectection_scores_list.end(), detectection_scores);
                std::copy(detection_class_list.begin(), detection_class_list.end(), detection_class);
                std::copy(detectection_location_list.begin(), detectection_location_list.end(), detectection_location);

                LOG_INFO("results %d\n", num_detections);
                overlayBoundingBox((*img), num_detections, detectection_location, detectection_scores, threshold);
                for (int i = 0; i < num_detections; i++)
                {
                    LOG_INFO("class %lf\n", detection_class[i]);
                    LOG_INFO("cordinates %lf %lf %lf %lf\n", detectection_location[i * 4], detectection_location[i * 4 + 1], detectection_location[i * 4 + 2], detectection_location[i * 4 + 3]);
                    LOG_INFO("score %lf\n", detectection_scores[i]);
                }
            }
            return RETURN_SUCCESS;
        }

        /**
         *  \brief  print tensor info
         *  \param  session onnx session
         *  \param  input_node_names input array node names
         * @returns int status
         */
        int printTensorInfo(Ort::Session *session, std::vector<const char *> *input_node_names)
        {
            size_t num_input_nodes = (*session).GetInputCount();
            size_t num_output_nodes = (*session).GetOutputCount();
            Ort::TypeInfo type_info = (*session).GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_node_dims = tensor_info.GetShape();
            LOG_INFO("number of inputs:%d \n", num_input_nodes);
            LOG_INFO("number of outputs: ", num_output_nodes);
            LOG_INFO("input(0) name: ", (*input_node_names)[0]);
            /* iterate over all input nodes */
            for (int i = 0; i < num_input_nodes; i++)
            {
                /* print input node names */
                LOG_INFO("Input %d : name=%s\n", i, input_node_names[i]);

                /* print input node types */
                Ort::TypeInfo type_info = (*session).GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                ONNXTensorElementDataType type = tensor_info.GetElementType();
                LOG_INFO("Input %d : type=%d\n", i, type);
                /* print input shapes/dims */
                input_node_dims = tensor_info.GetShape();
                LOG_INFO("Input %d : num_dims=%zu\n", i, input_node_dims.size());
                for (int j = 0; j < input_node_dims.size(); j++)
                {
                    LOG_INFO("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
                }
            }
            if (num_input_nodes != 1)
            {
                LOG_INFO("supports only single input model \n");
                return RETURN_FAIL;
            }
            return RETURN_SUCCESS;
        }

        /**
         *  \brief  Actual infernce happening
         *  \param  ModelInfo YAML parsed model info
         *  \param  Settings user input options  and default values of setting if any
         * @returns int
         */
        int runInference(ModelInfo *modelInfo, Settings *s)
        {
            std::string model_path = modelInfo->m_infConfig.modelFile;
            std::string image_path = s->input_bmp_path;
            std::string labels_path = s->labels_file_path;
            std::string artifacts_path;

            /*check artifacts path need to be overwritten from cmd line args */
            if (s->artifact_path != "")
                artifacts_path = s->artifact_path;
            else
                artifacts_path = modelInfo->m_infConfig.artifactsPath;
            cv::Mat img;
            void *inData;

            /* checking model path present or not*/
            if (!modelInfo->m_infConfig.modelFile.c_str())
            {
                LOG_INFO("no model file name\n");
                return RETURN_FAIL;
            }

            /* Initialize  enviroment, maintains thread pools and state info */
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
            /* Initialize session options */
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            if (s->accel)
            {
                LOG_INFO("accelerated mode\n");
                c_api_tidl_options *options = (c_api_tidl_options *)malloc(sizeof(c_api_tidl_options));
                LOG_INFO("artifacts: %s\n", artifacts_path.c_str());
                strcpy(options->artifacts_folder, artifacts_path.c_str());
                options->debug_level = 0;
                if (options == NULL)
                {
                    LOG_ERROR("failed to allocate c_api_tidl_options \n");
                    return RETURN_FAIL;
                }
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
            LOG_INFO("Loaded model %s\n", modelInfo->m_infConfig.modelFile.c_str());

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
                LOG_INFO("missmatch in YAML parsed wanted channels:%d and model:%d\n", wanted_channels, input_node_dims[1]);
            }
            if (wanted_height != modelInfo->m_preProcCfg.outDataHeight)
            {
                LOG_INFO("missmatch in YAML parsed wanted height:%d and model:%d\n", wanted_height, input_node_dims[2]);
            }
            if (wanted_width != modelInfo->m_preProcCfg.outDataWidth)
            {
                LOG_INFO("missmatch in YAML parsed wanted width:%d and model:%d\n", wanted_width, input_node_dims[3]);
                ;
            }

            /* output information */
            size_t num_output_nodes = session.GetOutputCount();
            std::vector<const char *> output_node_names(num_output_nodes);
            for (int i = 0; i < num_output_nodes; i++)
            {
                output_node_names[i] = session.GetOutputName(i, allocator);
            }
            for (int i = 0; i < num_input_nodes; i++)
            {
                input_node_names[i] = session.GetInputName(i, allocator);
            }

            type_info = session.GetOutputTypeInfo(0);
            auto output_tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> output_node_dims = output_tensor_info.GetShape();
            size_t output_tensor_size = output_node_dims[1];

            if (RETURN_FAIL == printTensorInfo(&session, &input_node_names))
                return RETURN_FAIL;
            int num_iter = s->loop_count;

            /* simplify ... using known dim values to calculate size */
            size_t input_tensor_size = wanted_channels * wanted_height * wanted_width;
            if (input_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                inData = TIDLRT_allocSharedMem(32, input_tensor_size * sizeof(float));
                if (inData == NULL)
                {
                    LOG_INFO("Could not allocate memory for inData \n ");
                    return RETURN_FAIL;
                }
                img = preprocImage<float>(image_path, (float *)inData, modelInfo->m_preProcCfg);
            }
            else
            {
                LOG_INFO("indata type not supported yet \n ");
                return RETURN_FAIL;
            }
            LOG_INFO("Output name -- %s \n", *(output_node_names.data()));

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            LOG_INFO("create cpu done\n");

            void *outData = TIDLRT_allocSharedMem(16, output_tensor_size * sizeof(float));
            if (outData == NULL)
            {
                LOG_INFO("Could not allocate memory for outData \n ");
                return RETURN_FAIL;
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
                LOG_INFO("in data type not supported\n");
                return RETURN_FAIL;
            }
            Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, inData, input_tensor_size_bytes, input_node_dims.data(), 4, input_tensor_type);
            assert(input_tensor.IsTensor());
            /* score model & input tensor, get back output tensor */
            auto run_options = Ort::RunOptions();
            run_options.SetRunLogVerbosityLevel(2);
            std::vector<Ort::Value> output_tensors;
            if (s->loop_count >= 1)
            {
                LOG_INFO("Session.Run() - Started for warmup runs\n");
                for (int i = 0; i < s->number_of_warmup_runs; i++)
                {
                    output_tensors = session.Run(run_options, input_node_names.data(), &input_tensor, num_input_nodes, output_node_names.data(), num_output_nodes);
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

            LOG_INFO("invoked\n");
            LOG_INFO("average time: %lf ms \n",
                     (getUs(stop_time) - getUs(start_time)) / (num_iter * 1000));

            if (modelInfo->m_preProcCfg.taskType == "classification")
            {
                if (RETURN_FAIL == prepClassificationResult(&img, &output_tensors, s, output_node_dims))
                    return RETURN_FAIL;
            }
            else if (modelInfo->m_preProcCfg.taskType == "detection")
            {
                if (RETURN_FAIL == prepDetectionResult(&img, &output_tensors, modelInfo))
                    return RETURN_FAIL;
            }
            else if (modelInfo->m_preProcCfg.taskType == "segmentation")
            {
                if (RETURN_FAIL == prepSegResult(&img, &output_tensors, s, modelInfo->m_postProcCfg.alpha))
                    return RETURN_FAIL;
            }

            cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            char filename[500];
            strcpy(filename, "test_data/");
            strcat(filename, "cpp_inference_out");
            strcat(filename, modelInfo->m_preProcCfg.modelName.c_str());
            strcat(filename, ".jpg");
            bool check = cv::imwrite(filename, img);
            if (check == false)
            {
                LOG_ERROR("Saving the image, FAILED\n");
            }
            LOG_INFO("Done!\n");
            return RETURN_SUCCESS;
        }
    }

}

int main(int argc, char *argv[])
{
    Settings s;
    if (parseArgs(argc, argv, &s) == RETURN_FAIL)
    {
        LOG_ERROR("Failed to parse the args\n");
        return RETURN_FAIL;
    }
    dumpArgs(&s);
    logSetLevel((LogLevel)s.log_level);
    /* Parse the input configuration file */
    ModelInfo model(s.model_zoo_path);
    if (model.initialize() == RETURN_FAIL)
    {
        LOG_ERROR("Failed to initialize model\n");
        return RETURN_FAIL;
    }
    if (onnx::main::runInference(&model, &s) == RETURN_FAIL)
    {
        LOG_ERROR("Failed to run runInference\n");
        return RETURN_FAIL;
    }
    return RETURN_SUCCESS;
}
