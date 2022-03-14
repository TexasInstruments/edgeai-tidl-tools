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

#define NUM_PARLLEL_MODELS 2

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
        ONNXTensorElementDataType getTensorType(int index, vector<Ort::Value> *output_tensors)
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
        int prepClassificationResult(cv::Mat *img, vector<Ort::Value> *output_tensors, Settings *s,
                                     vector<int64_t> output_node_dims)
        {
            LOG_INFO("preparing classification result \n");
            cv::resize((*img), (*img), cv::Size(512, 512), 0, 0, cv::INTER_AREA);
            ONNXTensorElementDataType op_tensor_type = getTensorType(0, output_tensors);
            /* Get pointer to output tensor float values*/
            vector<pair<float, int>> top_results;
            const float threshold = 0.001f;
            /* assuming output tensor of size [1,1000] or [1, 1001]*/
            int output_size = output_node_dims.data()[output_node_dims.size() - 1];
            int outputoffset;
            if (output_size == 1001)
                outputoffset = 0;
            else
                outputoffset = 1;
            if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
            {
                int64_t *int64arr = (*output_tensors).front().GetTensorMutableData<int64_t>();
                getTopN<int64_t>(int64arr,
                                 output_size, s->number_of_results, threshold,
                                 &top_results, true);
            }
            else if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                float *floatarr = (*output_tensors).front().GetTensorMutableData<float>();
                getTopN<float>(floatarr,
                               output_size, s->number_of_results, threshold,
                               &top_results, true);
            }
            else
            {
                LOG_ERROR("out data type not supported yet \n ");
                return RETURN_FAIL;
            }

            vector<string> labels;
            size_t label_count;
            if (RETURN_FAIL == readLabelsFile(s->labels_file_path, &labels, &label_count) != 0)
            {
                LOG_ERROR("failed to read label file");
                return RETURN_FAIL;
            }
            for (const auto &result : top_results)
            {
                const float confidence = result.first;
                const int index = result.second;
                LOG_INFO("%f: %d %s\n", confidence, index, labels[index + outputoffset].c_str());
            }
            int num_results = 5;
            (*img).data = overlayTopNClasses((*img).data, top_results, &labels, (*img).cols, (*img).rows, num_results, outputoffset);
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
        int prepSegResult(cv::Mat *img, vector<Ort::Value> *output_tensors, Settings *s, float alpha)
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
            else if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
            {
                uint8_t *tensor_op_array = (*output_tensors).front().GetTensorMutableData<uint8_t>();
                (*img).data = blendSegMask<uint8_t>((*img).data, tensor_op_array, (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
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
                    float *arr = (float *)calloc((nwidth * nheight), sizeof(float));
                    /* get arr with argmax function calculated */
                    argMax<float>(arr, tensor_op_array, nwidth, nheight, nclasses);
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
         *  \brief  print tensor info
         *  \param  session onnx session
         *  \param  input_node_names input array node names
         * @returns int status
         */
        int printTensorInfo(Ort::Session *session, vector<const char *> *input_node_names)
        {
            size_t num_input_nodes = (*session).GetInputCount();
            size_t num_output_nodes = (*session).GetOutputCount();
            Ort::TypeInfo type_info = (*session).GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            vector<int64_t> input_node_dims = tensor_info.GetShape();
            LOG_INFO("number of inputs:%d \n", num_input_nodes);
            LOG_INFO("number of outputs: %d\n", num_output_nodes);
            LOG_INFO("input(0) name: %s\n", (*input_node_names)[0]);
            /* iterate over all input nodes */
            for (int i = 0; i < num_input_nodes; i++)
            {
                /* print input node names */
                LOG_INFO("Input %d : name=%s\n", i, (*input_node_names)[i]);

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

        void *allocTensorMem(int size, int accel)
        {
            void *ptr = NULL;
            if (accel)
            {
                ptr = TIDLRT_allocSharedMem(64, size);
            }
            else
            {
                ptr = malloc(size);
            }
            if (ptr == NULL)
            {
                LOG_ERROR("Could not allocate memory for a Tensor of size %d \n ", size);
                exit(0);
            }
            return ptr;
        }
        void freeTensorMem(void *ptr, int accel)
        {
            if (accel)
            {
                TIDLRT_freeSharedMem(ptr);
            }
            else
            {
                free(ptr);
            }
        }

        typedef struct
        {
            ModelInfo *modelInfo;
            Settings *s;
            int priority;
            int breath_time = 10;
            int model_id;
            Ort::Env *env;
            float actual_time;

        } ort_model_struct;

        /**
         *  \brief  Hepler func to validate dimension
         *  \param  input_node_dims
         *  \param  arg
         * @returns int status
         */
        int validateTesorDim(vector<int64_t> input_node_dims, ort_model_struct *arg)
        {
            /* assuming NCHW*/
            int wanted_height = input_node_dims[2];
            int wanted_width = input_node_dims[3];
            int wanted_channels = input_node_dims[1];
            if (wanted_channels != arg->modelInfo->m_preProcCfg.numChans)
            {
                LOG_ERROR("missmatch in YAML parsed wanted channels:%d and model:%d\n", wanted_channels, input_node_dims[1]);
                return RETURN_FAIL;
            }
            if (wanted_height != arg->modelInfo->m_preProcCfg.outDataHeight)
            {
                LOG_ERROR("missmatch in YAML parsed wanted height:%d and model:%d\n", wanted_height, input_node_dims[2]);
                return RETURN_FAIL;
            }
            if (wanted_width != arg->modelInfo->m_preProcCfg.outDataWidth)
            {
                LOG_ERROR("missmatch in YAML parsed wanted width:%d and model:%d\n", wanted_width, input_node_dims[3]);
                return RETURN_FAIL;
            }
            return RETURN_SUCCESS;
        }

        /**
         *  \brief  Hepler func to calculate the ouput tensor size
         *  \param  node_dims
         *  \param  tensor_type
         * @returns int status/ tensor_size
         */
        int calcTensorSize(std::vector<int64_t> node_dims, ONNXTensorElementDataType tensor_type)
        {
            size_t tensor_size = 1;
            for (int j = node_dims.size() - 1; j >= 0; j--)
            {
                tensor_size *= node_dims[j];
            }
            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                tensor_size *= sizeof(float);
            }
            else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
            {
                tensor_size *= sizeof(uint8_t);
            }
            else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
            {
                tensor_size *= sizeof(int64_t);
            }
            else
            {
                LOG_ERROR("Un Supported output tensor_type\n");
                return RETURN_FAIL;
            }
            return tensor_size;
        }

        /**
         *  \brief  Get the actual run time of model if ran individually
         *  \param  arg ort_model_struct containing model details to be ran
         * @returns int status
         */
        int getActualRunTime(ort_model_struct *arg)
        {
            LOG_INFO("Fetching actual inference time for model %s\n", arg->modelInfo->m_preProcCfg.modelName.c_str());
            Settings *s = arg->s;
            string artifacts_path = arg->modelInfo->m_infConfig.artifactsPath;
            string model_path = arg->modelInfo->m_infConfig.modelFile;
            cv::Mat img;
            void *inData;
            string input_img_path = s->input_img_paths[arg->model_id];
            int loop_count = s->loop_counts[arg->model_id];
            struct timeval start_time, stop_time;
            float avg_time;
            std::vector<Ort::Value> output_tensors;

            /* Initialize session options */
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.DisablePerSessionThreads();
            /* Initialize session options */
            if (s->accel)
            {
                c_api_tidl_options *options = (c_api_tidl_options *)malloc(sizeof(c_api_tidl_options));
                if (options == NULL)
                {
                    LOG_ERROR("failed to allocate c_api_tidl_options \n");
                    pthread_exit(NULL);
                }
                strcpy(options->artifacts_folder, artifacts_path.c_str());
                options->priority = 0;
                options->max_pre_empt_delay = FLT_MAX;
                options->debug_level = 0;
                OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_Tidl(session_options, options);
            }
            else
            {
                OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_CPU(session_options, false);
            }
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            Ort::AllocatorWithDefaultOptions allocator;
            Ort::Session session(*arg->env, model_path.c_str(), session_options);

            /* Input information */
            size_t num_input_nodes = session.GetInputCount();
            vector<const char *> input_node_names(num_input_nodes);
            Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            vector<int64_t> input_node_dims = tensor_info.GetShape();
            ONNXTensorElementDataType input_tensor_type = tensor_info.GetElementType();
            int wanted_height = input_node_dims[2];
            int wanted_width = input_node_dims[3];
            int wanted_channels = input_node_dims[1];

            if (validateTesorDim(input_node_dims, arg) != RETURN_SUCCESS)
            {
                LOG_ERROR("failed o validate input tensor dims\n");
                return RETURN_FAIL;
            }
            /* output information */
            size_t num_output_nodes = session.GetOutputCount();
            vector<const char *> output_node_names(num_output_nodes);
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
            vector<int64_t> output_node_dims = output_tensor_info.GetShape();
            size_t output_tensor_size = output_node_dims[1];

            if (RETURN_FAIL == printTensorInfo(&session, &input_node_names))
            {
                LOG_ERROR("failed to print tensor info\n");
                return RETURN_FAIL;
            }

            std::vector<Ort::Value> input_tensors;
            ssize_t input_tensor_size_bytes;
            /* simplify ... using known dim values to calculate size */
            size_t input_tensor_size = wanted_channels * wanted_height * wanted_width;
            if (input_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                inData = TIDLRT_allocSharedMem(32, input_tensor_size * sizeof(float));
                if (inData == NULL)
                {
                    LOG_ERROR("Could not allocate memory for inData \n ");
                    return RETURN_FAIL;
                }
                input_tensor_size_bytes = input_tensor_size * sizeof(float);
                inData = allocTensorMem(input_tensor_size_bytes, (arg->s->accel && arg->s->device_mem));
                img = preprocImage<float>(input_img_path, (float *)inData, arg->modelInfo->m_preProcCfg);
            }
            else if (input_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
            {
                inData = TIDLRT_allocSharedMem(32, input_tensor_size * sizeof(uint8_t));
                if (inData == NULL)
                {
                    LOG_ERROR("Could not allocate memory for inData \n ");
                    return RETURN_FAIL;
                }
                input_tensor_size_bytes = input_tensor_size * sizeof(uint8_t);
                inData = allocTensorMem(input_tensor_size_bytes, (arg->s->accel && arg->s->device_mem));
                img = preprocImage<uint8_t>(input_img_path, (uint8_t *)inData, arg->modelInfo->m_preProcCfg);
            }
            else
            {
                LOG_ERROR("indata type not supported yet \n ");
                return RETURN_FAIL;
            }

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, inData, input_tensor_size_bytes, input_node_dims.data(), 4, input_tensor_type);
            input_tensors.push_back(std::move(input_tensor));

            auto run_options = Ort::RunOptions();
            run_options.SetRunLogVerbosityLevel(0);
            /* running default 10 warmup runs*/
            std::vector<Ort::Value> output_tensors_warm_up;
            for (int i = 0; i < 10; i++)
            {
                output_tensors_warm_up = session.Run(run_options, input_node_names.data(), input_tensors.data(), 1, output_node_names.data(), num_output_nodes);
            }

            void *outData = TIDLRT_allocSharedMem(16, output_tensor_size * sizeof(float));
            if (outData == NULL)
            {
                LOG_ERROR("Could not allocate memory for outData \n ");
                return RETURN_FAIL;
            }
            Ort::IoBinding binding(session);
            binding.BindInput(input_node_names[0], input_tensors[0]);
            for (int idx = 0; idx < num_output_nodes; idx++)
            {
                auto node_dims = output_tensors_warm_up[idx].GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
                ONNXTensorElementDataType tensor_type = output_tensors_warm_up[idx].GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType();
                size_t tensor_size = calcTensorSize(node_dims, tensor_type);
                if (tensor_size == RETURN_FAIL)
                    return RETURN_FAIL;
                void *outData = allocTensorMem(tensor_size, (arg->s->accel && arg->s->device_mem));
                auto output_tensor = Ort::Value::CreateTensor(memory_info, (void *)outData, tensor_size, node_dims.data(), node_dims.size(), tensor_type);
                output_tensors.push_back(std::move(output_tensor));
                binding.BindOutput(output_node_names[idx], output_tensors[idx]);
            }
            gettimeofday(&start_time, nullptr);
            for (int i = 0; i < loop_count; i++)
            {
                session.Run(run_options, binding);
            }
            gettimeofday(&stop_time, nullptr);
            avg_time = ((getUs(stop_time) - getUs(start_time)) / (loop_count * 1000));
            arg->actual_time = avg_time;
            LOG_INFO("Done fetching actual inference time for model %s :avg_time %f\n", arg->modelInfo->m_preProcCfg.modelName.c_str(), avg_time);
            return RETURN_SUCCESS;
        }

        /**
         *  \brief  Compare the final tensor output and each iteration output
         *  \param  loop_count number of iteration run per model
         *  \param  output_tensor_length length of op tensor
         *  \param  output_tensors vector of out Ort:Value
         * @returns int status
         */
        int compTensorOut(int loop_count, int output_tensor_length,std::vector<Ort::Value> *output_tensors)
        {
            int is_ident_flag = 0;
            ONNXTensorElementDataType op_tensor_type = getTensorType(0, output_tensors);
            float *floatarr = (*output_tensors).at(loop_count -1).GetTensorMutableData<float>();
            if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
            {
                int64_t *arr_ref = (*output_tensors).front().GetTensorMutableData<int64_t>();
                for (size_t k = 0; k < loop_count; k++)
                {
                    int64_t *arr = (*output_tensors).at(k).GetTensorMutableData<int64_t>();
                    for (size_t j = 0; j < output_tensor_length; j++)
                    {
                        if (arr_ref[j] != arr[j])
                        {
                            is_ident_flag = 1;
                        }
                    }
                }
            }
            else if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                float *arr_ref = (*output_tensors).front().GetTensorMutableData<float>();
                for (size_t k = 0; k < loop_count; k++)
                {
                    float *arr = (*output_tensors).at(k).GetTensorMutableData<float>();
                    for (size_t j = 0; j < output_tensor_length; j++)
                    {
                        if (arr_ref[j] != arr[j])
                        {
                            is_ident_flag = 1;
                        }
                    }
                }
            }
            else if (op_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
            {
                uint8_t *arr_ref = (*output_tensors).front().GetTensorMutableData<uint8_t>();
                for (size_t k = 0; k < loop_count; k++)
                {
                    uint8_t *arr = (*output_tensors).at(k).GetTensorMutableData<uint8_t>();
                    for (size_t j = 0; j < output_tensor_length; j++)
                    {
                        if (arr_ref[j] != arr[j])
                        {
                            is_ident_flag = 1;
                        }
                    }
                }   
            }
            else
            {
                LOG_ERROR("out data type not supported yet \n ");
                return RETURN_FAIL;
            }
            return is_ident_flag;
        }

        void *infer(void *argument)
        {
            ort_model_struct *arg = (ort_model_struct *)argument;
            Settings *s = arg->s;
            string artifacts_path = arg->modelInfo->m_infConfig.artifactsPath;
            string model_path = arg->modelInfo->m_infConfig.modelFile;
            cv::Mat img;
            void *inData;
            string input_img_path = s->input_img_paths[arg->model_id];
            struct timeval start_time, stop_time, iter_start, iter_end;
            /*TODO decide the max number of buffer need to kept for op tensor val*/
            /*temperory keeping: 10*/
            int max_num_op_saved = 10;

            /* Initialize session options */
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.DisablePerSessionThreads();
            /* Initialize session options */
            if (s->accel)
            {
                LOG_INFO("accelerated mode\n");
                c_api_tidl_options *options = (c_api_tidl_options *)malloc(sizeof(c_api_tidl_options));
                strcpy(options->artifacts_folder, artifacts_path.c_str());
                char prior_char[2], pre_empt_char[64];
                std::sprintf(prior_char, "%d", arg->priority);
                options->priority = arg->priority;
                if (arg->s->max_pre_empts[arg->model_id] == -1)
                {
                    std::sprintf(pre_empt_char, "%d", FLT_MAX);
                }
                else
                {
                    std::sprintf(pre_empt_char, "%f", arg->s->max_pre_empts[arg->model_id]);
                }
                options->debug_level = 0;
                if (options == NULL)
                {
                    LOG_ERROR("failed to allocate c_api_tidl_options \n");
                    pthread_exit(NULL);
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
            pthread_mutex_lock(&ort_pr_lock);
            Ort::Session session(*arg->env, model_path.c_str(), session_options);
            pthread_mutex_unlock(&ort_pr_lock);
            LOG_INFO("Loaded model %s\n", arg->modelInfo->m_infConfig.modelFile.c_str());

            /* Input information */
            size_t num_input_nodes = session.GetInputCount();
            vector<const char *> input_node_names(num_input_nodes);
            Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            vector<int64_t> input_node_dims = tensor_info.GetShape();
            ONNXTensorElementDataType input_tensor_type = tensor_info.GetElementType();
            int wanted_height = input_node_dims[2];
            int wanted_width = input_node_dims[3];
            int wanted_channels = input_node_dims[1];

            if (validateTesorDim(input_node_dims, arg) != RETURN_SUCCESS)
            {
                LOG_ERROR("failed to validate input tensor dims\n");
                pthread_exit(NULL);
            }
            /* output information */
            size_t num_output_nodes = session.GetOutputCount();
            vector<const char *> output_node_names(num_output_nodes);
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
            vector<int64_t> output_node_dims = output_tensor_info.GetShape();
            size_t output_tensor_size = output_node_dims[1];

            if (RETURN_FAIL == printTensorInfo(&session, &input_node_names))
                pthread_exit(NULL);

            int num_iter = s->loop_counts[arg->model_id];

            std::vector<Ort::Value> input_tensors;
            ssize_t input_tensor_size_bytes;
            /* simplify ... using known dim values to calculate size */
            size_t input_tensor_size = wanted_channels * wanted_height * wanted_width;
            if (input_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                inData = TIDLRT_allocSharedMem(32, input_tensor_size * sizeof(float));
                if (inData == NULL)
                {
                    LOG_ERROR("Could not allocate memory for inData \n ");
                    pthread_exit(NULL);
                }
                input_tensor_size_bytes = input_tensor_size * sizeof(float);
                inData = allocTensorMem(input_tensor_size_bytes, (arg->s->accel && arg->s->device_mem));
                img = preprocImage<float>(input_img_path, (float *)inData, arg->modelInfo->m_preProcCfg);
            }
            else if (input_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
            {
                inData = TIDLRT_allocSharedMem(32, input_tensor_size * sizeof(uint8_t));
                if (inData == NULL)
                {
                    LOG_ERROR("Could not allocate memory for inData \n ");
                    pthread_exit(NULL);
                }
                input_tensor_size_bytes = input_tensor_size * sizeof(uint8_t);
                inData = allocTensorMem(input_tensor_size_bytes, (arg->s->accel && arg->s->device_mem));
                img = preprocImage<uint8_t>(input_img_path, (uint8_t *)inData, arg->modelInfo->m_preProcCfg);
            }
            else
            {
                LOG_ERROR("indata type not supported yet \n ");
                pthread_exit(NULL);
            }

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            LOG_INFO("create cpu done\n");

            Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, inData, input_tensor_size_bytes, input_node_dims.data(), 4, input_tensor_type);
            input_tensors.push_back(std::move(input_tensor));

            auto run_options = Ort::RunOptions();
            run_options.SetRunLogVerbosityLevel(2);
            auto output_tensors_warm_up = session.Run(run_options, input_node_names.data(), input_tensors.data(), 1, output_node_names.data(), num_output_nodes);

            void *outData = TIDLRT_allocSharedMem(16, output_tensor_size * sizeof(float));
            if (outData == NULL)
            {
                LOG_ERROR("Could not allocate memory for outData \n ");
                pthread_exit(NULL);
            }
            Ort::IoBinding binding(session);
            binding.BindInput(input_node_names[0], input_tensors[0]);
            std::vector<Ort::Value> output_tensors;
            /* assuming 1 op tensor in model */
            for (int idx = 0; idx < max_num_op_saved; idx++)
            {
                auto node_dims = output_tensors_warm_up[0].GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
                ONNXTensorElementDataType tensor_type = output_tensors_warm_up[0].GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType();
                size_t tensor_size = calcTensorSize(node_dims, tensor_type);
                if (tensor_size == RETURN_FAIL)
                    pthread_exit(NULL);
                void *outData = allocTensorMem(tensor_size, (arg->s->accel && arg->s->device_mem));
                auto output_tensor = Ort::Value::CreateTensor(memory_info, (void *)outData, tensor_size, node_dims.data(), node_dims.size(), tensor_type);
                output_tensors.push_back(std::move(output_tensor));
            }

            pthread_barrier_wait(&barrier);
            gettimeofday(&start_time, nullptr);
            // auto finish = system_clock::now() + minutes{1};
            auto finish = system_clock::now() + minutes{1};
            int k = 0, num_switches = 0;
            float time_spend;
            binding.BindOutput(output_node_names[0], output_tensors[0]);
            do
            {
                gettimeofday(&iter_start, nullptr);
                session.Run(run_options, binding);
                gettimeofday(&iter_end, nullptr);
                time_spend = (float)((getUs(iter_end) - getUs(iter_start)) / (1000));
                /* TODO decide the differnece : temperory using  1.1 ms */
                if (time_spend > arg->actual_time + 1.1 && num_switches < max_num_op_saved)
                {
                    /* allocating 0th tensor from out_pts array at invoke time
                    each time it exceeds its actual inference time */
                    binding.BindOutput(output_node_names[0], output_tensors[num_switches % max_num_op_saved]);
                    num_switches++;
                }
                k++;
            } while (system_clock::now() < finish);
            gettimeofday(&stop_time, nullptr);
            float avg_time = ((getUs(stop_time) - getUs(start_time)) / (k * 1000));
            LOG_INFO("Model %s start time %ld.%06ld\n", arg->modelInfo->m_preProcCfg.modelName.c_str(), start_time.tv_sec, start_time.tv_usec);
            LOG_INFO("Model %s stop time %ld.%06ld\n", arg->modelInfo->m_preProcCfg.modelName.c_str(), stop_time.tv_sec, stop_time.tv_usec);
            LOG_INFO("Total num context switches for model %s:%d \n", arg->modelInfo->m_preProcCfg.modelName.c_str(), num_switches);
            LOG_INFO("Total num iterations run for model %s:%d \n", arg->modelInfo->m_preProcCfg.modelName.c_str(), k);
            LOG_INFO("FPS for model %s :%f \n", arg->modelInfo->m_preProcCfg.modelName.c_str(), (float)((float)k / 60));

            if (arg->modelInfo->m_preProcCfg.taskType == "classification")
            {
                int count = (num_switches >= max_num_op_saved)?max_num_op_saved:num_switches;
                int output_tensor_length = output_node_dims.data()[output_node_dims.size() - 1];
                if (0 != compTensorOut(count, output_tensor_length, &output_tensors))
                {
                    LOG_ERROR("Compare op tensor iterations : failed\n");
                    pthread_exit(NULL);
                }else{
                    LOG_INFO("Compare op tensor iterations : success\n");
                }

                if (RETURN_FAIL == prepClassificationResult(&img, &output_tensors, s, output_node_dims))
                        pthread_exit(NULL);
                }
            else if (arg->modelInfo->m_preProcCfg.taskType == "detection")
            {

                /*store tensor_shape info of op tensors in arr
                to avoid recalculation*/
                vector<vector<int64_t>> tensor_shapes_vec;
                vector<vector<float>> f_tensor_unformatted;
                for (size_t i = 0; i < output_tensors.size(); i++)
                {
                    vector<int64_t> tensor_shape = output_tensors.at(i).GetTensorTypeAndShapeInfo().GetShape();
                    tensor_shapes_vec.push_back(tensor_shape);
                }

                /* num of detection in op tensor  assumes the lastbut one of
                1st op tensor*/
                vector<int64_t> tensor_shape = tensor_shapes_vec[0];
                int nboxes = tensor_shape[tensor_shape.size() - 2];

                for (size_t i = 0; i < output_tensors.size(); i++)
                {
                    /* temp vector to store converted ith tensor */
                    vector<float> f_tensor;
                    /* shape of the ith tensor*/
                    vector<int64_t> tensor_shape = tensor_shapes_vec[i];
                    /* type of the ith tensor*/
                    ONNXTensorElementDataType tensor_type = output_tensors.at(i).GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType();
                    /* num of values in ith tensor*/
                    int num_val_tensor;
                    /*Extract the last dimension from each of the output
                    tensors.last dimension will give the number of values present in
                    given tensor. Need to ignore all dimensions with value 1 since it
                    does not actually add a dimension */
                    auto temp = tensor_shape;
                    for (auto it = temp.begin(); it < temp.end(); it++)
                    {
                        if ((*it) == 1)
                        {
                            temp.erase(it);
                            it--;
                        }
                    }
                    if (temp.size() == 1)
                        num_val_tensor = 1;
                    else
                    {
                        num_val_tensor = temp[temp.size() - 1];
                    }

                    /*convert tensor to float vector*/
                    if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                    {
                        float *inDdata = output_tensors.at(i).GetTensorMutableData<float>();
                        createFloatVec<float>(inDdata, &f_tensor, tensor_shape);
                    }
                    else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
                    {
                        int64_t *inDdata = output_tensors.at(i).GetTensorMutableData<int64_t>();
                        createFloatVec<int64_t>(inDdata, &f_tensor, tensor_shape);
                    }
                    else
                    {
                        LOG_ERROR("out tensor data type not supported\n");
                        pthread_exit(NULL);
                    }
                    /*append all output tensors in to single vector<vector<float>*/
                    for (size_t j = 0; j < nboxes; j++)
                    {
                        vector<float> temp;
                        for (size_t k = 0; k < num_val_tensor; k++)
                        {
                            temp.push_back(f_tensor[j * num_val_tensor + k]);
                        }
                        f_tensor_unformatted.push_back(temp);
                    }
                }
                if (RETURN_FAIL == prepDetectionResult(&img, &f_tensor_unformatted, tensor_shapes_vec, arg->modelInfo, num_output_nodes, nboxes))
                    pthread_exit(NULL);
            }
            else if (arg->modelInfo->m_preProcCfg.taskType == "segmentation")
            {
                int count = (num_switches >= max_num_op_saved)?max_num_op_saved:num_switches;
                int wanted_height = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape()[2];
                int wanted_width = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape()[3];
                size_t output_tensor_length = wanted_height * wanted_width;
                if (0 != compTensorOut(count, output_tensor_length, &output_tensors))
                {
                    LOG_ERROR("Compare op tensor iterations : failed\n");
                    pthread_exit(NULL);
                }else{
                    LOG_INFO("Compare op tensor iterations : success\n");
                }
                if (RETURN_FAIL == prepSegResult(&img, &output_tensors, s, arg->modelInfo->m_postProcCfg.alpha))
                    pthread_exit(NULL);
            }
            /* freeing shared mem*/
            for (size_t i = 0; i < output_tensors.size(); i++)
            {
                void *ptr = output_tensors[i].GetTensorMutableData<void>();
                freeTensorMem(ptr, (s->accel && s->device_mem));
            }
            for (size_t i = 0; i < input_tensors.size(); i++)
            {
                void *ptr = input_tensors[i].GetTensorMutableData<void>();
                freeTensorMem(ptr, (s->accel && s->device_mem));
            }

            cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            string filename, foldername;
            foldername = foldername + "output_images/";
            struct stat buffer;
            if (stat(foldername.c_str(), &buffer) != 0)
            {
                if (mkdir(foldername.c_str(), 0777) == -1)
                {
                    LOG_ERROR("failed to create folder %s:%s\n", foldername, strerror(errno));
                    pthread_exit(NULL);
                }
            }
            if (stat(foldername.c_str(), &buffer) != 0)
            {
                if (mkdir(foldername.c_str(), 0777) == -1)
                {
                    LOG_ERROR("failed to create folder %s:%s\n", foldername, strerror(errno));
                    pthread_exit(NULL);
                }
            }
            filename = "cpp_out_";
            filename = filename + arg->modelInfo->m_preProcCfg.modelName.c_str();
            filename = filename + ".jpg";
            foldername = foldername + filename;
            if (false == cv::imwrite(foldername, img))
            {
                LOG_INFO("Saving the image, FAILED\n");
                pthread_exit(NULL);
            }

            LOG_INFO("\nCompleted_Model : 0, Name : %s, Total time : %f, Offload Time : 0 , DDR RW MBs : 0, Output File : %s \n \n",
                     arg->modelInfo->m_postProcCfg.modelName.c_str(), avg_time, filename.c_str());
            pthread_exit(NULL);
        }

        /**
         *  \brief  Actual infernce happening
         *  \param  ModelInfo YAML parsed model info
         *  \param  Settings user input options  and default values of setting if any
         * @returns int
         */
        int runInference(ModelInfo **modelInfos, Settings *s)
        {
            ort_model_struct args[NUM_PARLLEL_MODELS];
            OrtEnv *environment;
            int ret;
            OrtThreadingOptions *envOpts = nullptr;
            Ort::GetApi().CreateThreadingOptions(&envOpts);
            Ort::GetApi().SetGlobalInterOpNumThreads(envOpts, 0);
            Ort::GetApi().SetGlobalSpinControl(envOpts, false);
            Ort::GetApi().CreateEnvWithGlobalThreadPools(ORT_LOGGING_LEVEL_WARNING, "test", envOpts, &environment);
            Ort::Env env = Ort::Env(environment);
            struct Inference_info *inference_infos = (struct Inference_info *)malloc(sizeof(struct Inference_info) * s->number_of_threads * 4);
            for (size_t i = 0; i < NUM_PARLLEL_MODELS; i++)
            {
                LOG_INFO("Prep model %d\n", i);
                /* checking model path present or not*/
                if (!modelInfos[i]->m_infConfig.modelFile.c_str())
                {
                    LOG_ERROR("no model file name\n");
                    return RETURN_FAIL;
                }
                args[i].env = &env;
                args[i].modelInfo = modelInfos[i];
                args[i].s = s;
                args[i].model_id = i;
                args[i].priority = s->priors[i];
                if (RETURN_FAIL == getActualRunTime(&args[i]))
                    return RETURN_FAIL;
            }
            if (pthread_mutex_init(&ort_pr_lock, NULL) != 0)
            {
                LOG_ERROR("\n mutex init has failed\n");
                return RETURN_FAIL;
            }
            pthread_attr_t tattr;
            ret = pthread_attr_init(&tattr);
            pthread_barrierattr_t barr_attr;
            ret = pthread_barrier_init(&barrier, &barr_attr, (2 * s->number_of_threads));
            if (ret != 0)
            {
                LOG_ERROR("barrier creation failied exiting\n");
                return RETURN_FAIL;
            }

            pthread_t ptid[2 * NUM_PARLLEL_MODELS];

            for (size_t i = 0; i < s->number_of_threads; i++)
            {
                /* Creating a new thread*/
                pthread_create(&ptid[2 * i], &tattr, &infer, &args[0]);
                pthread_create(&ptid[2 * i + 1], &tattr, &infer, &args[1]);
            }
            for (size_t i = 0; i < s->number_of_threads; i++)
            {
                // Waiting for the created thread to terminate
                pthread_join(ptid[2 * i], NULL);
                pthread_join(ptid[2 * i + 1], NULL);
            }
            pthread_barrierattr_destroy(&barr_attr);
            pthread_mutex_destroy(&ort_pr_lock);
            return RETURN_SUCCESS;
        }
    } // namespace::onnx
} // namespace::onnx::main

int main(int argc, char *argv[])
{
    Settings s;
    if (parseArgs(argc, argv, &s) == RETURN_FAIL)
    {
        LOG_ERROR("Failed to parse the args\n");
        return RETURN_FAIL;
    }
    dumpArgs(&s);
    /* Parse the model YAML file */
    ModelInfo model1(s.model_paths[0]);
    if (model1.initialize() == RETURN_FAIL)
    {
        LOG_ERROR("Failed to initialize model\n");
        return RETURN_FAIL;
    }

    /* Parse the model YAML file */
    ModelInfo model2(s.model_paths[1]);
    if (model2.initialize() == RETURN_FAIL)
    {
        LOG_ERROR("Failed to initialize model\n");
        return RETURN_FAIL;
    }
    ModelInfo *model_infos[NUM_PARLLEL_MODELS];
    model_infos[0] = &model1;
    model_infos[1] = &model2;
    if (onnx::main::runInference(model_infos, &s) == RETURN_FAIL)
    {
        LOG_ERROR("Failed to run runInference\n");
        return RETURN_FAIL;
    }
    return RETURN_SUCCESS;
}
