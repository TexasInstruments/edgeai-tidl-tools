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

#include "tfl_priority_scheduling.h"

namespace tflite
{
    namespace main
    {

        typedef struct
        {
            ModelInfo *modelInfo;
            std::unique_ptr<tflite::FlatBufferModel> model;
            tflite::ops::builtin::BuiltinOpResolver resolver;
            Settings *s;
            int priority;
            int model_id;
            float actual_times[NUM_PARLLEL_MODELS];
            vector<double> xs, ys;

        } tfl_model_struct;

        /* global var to keep the inference count on each thread*/
        int infer_count = 0;

        /**
         *  \brief  prepare the segemntation result inplace
         *  \param  img cv image to do inplace transform
         *  \param  wanted_width
         *  \param  wanted_height
         *  \param  alpha
         *  \param  interpreter pointer of tflite
         *  \param  outputs pointer of output vector
         * @returns int status
         */
        int prepSegResult(cv::Mat *img, int wanted_width, int wanted_height, float alpha,
                          std::unique_ptr<tflite::Interpreter> *interpreter, const std::vector<int> *outputs)
        {
            LOG_INFO("preparing segmentation result \n");
            TfLiteType type = (*interpreter)->tensor((*outputs)[0])->type;
            if (type == TfLiteType::kTfLiteInt32)
            {
                int32_t *outputTensor = (*interpreter)->tensor((*outputs)[0])->data.i32;
                (*img).data = blendSegMask<int32_t>((*img).data, outputTensor, (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
            }
            else if (type == TfLiteType::kTfLiteInt64)
            {
                int64_t *outputTensor = (*interpreter)->tensor((*outputs)[0])->data.i64;
                (*img).data = blendSegMask<int64_t>((*img).data, outputTensor, (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
            }
            else if (type == TfLiteType::kTfLiteFloat32)
            {
                float *outputTensor = (*interpreter)->tensor((*outputs)[0])->data.f;
                (*img).data = blendSegMask<float>((*img).data, outputTensor, (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
            }
            else if (type == TfLiteType::kTfLiteUInt8)
            {
                uint8_t *outputTensor = (*interpreter)->tensor((*outputs)[0])->data.uint8;
                (*img).data = blendSegMask<uint8_t>((*img).data, outputTensor, (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
            }
            else
            {
                LOG_ERROR("op tensor type not supprted\n");
                return RETURN_FAIL;
            }
            return RETURN_SUCCESS;
        }

        /**
         *  \brief  prepare the classification result inplace
         *  \param  img cv image to do inplace transform
         *  \param  interpreter pointer of tflite
         *  \param  outputs pointer of output vector
         *  \param  s settings
         * @returns int status
         */
        int prepClassificationResult(cv::Mat *img, std::unique_ptr<tflite::Interpreter> *interpreter,
                                     const std::vector<int> *outputs, Settings *s)
        {
            LOG_INFO("preparing clasification result \n");
            const float threshold = 0.001f;
            std::vector<std::pair<float, int>> top_results;

            TfLiteIntArray *output_dims = (*interpreter)->tensor((*outputs)[0])->dims;
            /* assume output dims to be something like (1, 1, ... ,size) */
            auto output_size = output_dims->data[output_dims->size - 1];
            int outputoffset;
            if (output_size == 1001)
                outputoffset = 0;
            else
                outputoffset = 1;
            switch ((*interpreter)->tensor((*outputs)[0])->type)
            {
            case kTfLiteFloat32:
                getTopN<float>((*interpreter)->typed_output_tensor<float>(0), output_size,
                               s->number_of_results, threshold, &top_results, true);
                break;
            case kTfLiteUInt8:
                getTopN<uint8_t>((*interpreter)->typed_output_tensor<uint8_t>(0),
                                 output_size, s->number_of_results, threshold,
                                 &top_results, false);
                break;
            default:
                LOG_ERROR("cannot handle output type %d yet", (*interpreter)->tensor((*outputs)[0])->type);
                return RETURN_FAIL;
            }

            std::vector<string> labels;
            size_t label_count;

            if (readLabelsFile(s->labels_file_path, &labels, &label_count) != 0)
            {
                LOG_ERROR("label file not found!!! \n");
                return RETURN_FAIL;
            }

            for (const auto &result : top_results)
            {
                const float confidence = result.first;
                const int index = result.second;
                LOG_INFO("%f: %d :%s\n", confidence, index, labels[index + outputoffset].c_str());
            }
            int num_results = 5;
            (*img).data = overlayTopNClasses((*img).data, top_results, &labels, (*img).cols, (*img).rows, num_results, outputoffset);
            return RETURN_SUCCESS;
        }
        /**
         *  \brief  Compare the final tensor output and each iteration output
         *  \param  loop_count number of iteration run per model
         *  \param  output_tensor_length length of op tensor
         *  \param  interpreter tflite interpreter
         *  \param  out_ptrs each iteration output tensor start addresses
         * @returns int status
         */
        template <class T>
        int compTensorOut(int loop_count, int output_tensor_length, std::unique_ptr<tflite::Interpreter> *interpreter, void *out_ptrs[])
        {
            int is_ident_flag = 0;
            for (size_t k = 0; k < loop_count; k++)
            {
                T *val = (*interpreter)->typed_output_tensor<T>(0);
                for (size_t j = 0; j < output_tensor_length; j++)
                {
                    if (*((T *)((T *)out_ptrs[k] + j)) != *(val + j))
                    {
                        is_ident_flag = 1;
                    }
                }
            }
            return is_ident_flag;
        }

        template int compTensorOut<uint8_t>(int loop_count, int output_tensor_length, std::unique_ptr<tflite::Interpreter> *interpreter, void *out_ptrs[]);
        template int compTensorOut<float>(int loop_count, int output_tensor_length, std::unique_ptr<tflite::Interpreter> *interpreter, void *out_ptrs[]);
        template int compTensorOut<int32_t>(int loop_count, int output_tensor_length, std::unique_ptr<tflite::Interpreter> *interpreter, void *out_ptrs[]);

        /**
         *  \brief  Get the invoke run time of model for single run
         *  \param  arg tfl_model_struct containing model details to be ran
         * @returns int status
         */
        int getInvokeTime(tfl_model_struct *arg)
        {
            void *in_ptrs[16] = {NULL};
            void *out_ptrs[16] = {NULL};
            vector<int> inputs, outputs;
            int wanted_batch, wanted_height, wanted_width, wanted_channels;
            std::unique_ptr<tflite::Interpreter> interpreter;
            tflite::InterpreterBuilder(*arg->model, arg->resolver)(&interpreter);
            string input_img_path = arg->s->input_img_paths[arg->model_id];
            if (!interpreter)
            {
                LOG_ERROR("Failed to construct interpreter\n");
                return RETURN_FAIL;
            }
            inputs = interpreter->inputs();
            outputs = interpreter->outputs();
            if (inputs.size() != 1)
            {
                LOG_ERROR("Supports only single input models \n");
                return RETURN_FAIL;
            }
            if (arg->s->accel == 1)
            {
                /* This part creates the dlg_ptr */
                typedef TfLiteDelegate *(*tflite_plugin_create_delegate)(char **, char **, size_t, void (*report_error)(const char *));
                tflite_plugin_create_delegate tflite_plugin_dlg_create;
                char prior_char[2], pre_empt_char[64];
                std::sprintf(prior_char, "%d", 0);
                std::sprintf(pre_empt_char, "%f", FLT_MAX);
                char *keys[] = {(char *)"artifacts_folder", (char *)"num_tidl_subgraphs", (char *)"debug_level", (char *)"priority", (char *)"max_pre_empt_delay"};
                char *values[] = {(char *)arg->modelInfo->m_infConfig.artifactsPath.c_str(), (char *)"16", (char *)"0", (char *)prior_char, (char *)pre_empt_char};
                void *lib = dlopen("libtidl_tfl_delegate.so", RTLD_NOW);
                assert(lib);
                tflite_plugin_dlg_create = (tflite_plugin_create_delegate)dlsym(lib, "tflite_plugin_create_delegate");
                TfLiteDelegate *dlg_ptr = tflite_plugin_dlg_create(keys, values, 5, NULL);
                interpreter->ModifyGraphWithDelegate(dlg_ptr);
            }
            if (interpreter->AllocateTensors() != kTfLiteOk)
            {
                LOG_ERROR("Failed to allocate tensors!");
                return RETURN_FAIL;
            }
            if (arg->s->device_mem)
            {
                for (uint32_t i = 0; i < inputs.size(); i++)
                {
                    const TfLiteTensor *tensor = interpreter->input_tensor(i);
                    in_ptrs[i] = TIDLRT_allocSharedMem(tflite::kDefaultTensorAlignment, tensor->bytes);
                    if (in_ptrs[i] == NULL)
                    {
                        LOG_ERROR("Could not allocate Memory for input: %s\n", tensor->name);
                        return RETURN_FAIL;
                    }
                    interpreter->SetCustomAllocationForTensor(inputs[i], {in_ptrs[i], tensor->bytes});
                }
                for (uint32_t i = 0; i < outputs.size(); i++)
                {
                    const TfLiteTensor *tensor = interpreter->output_tensor(i);
                    out_ptrs[i] = TIDLRT_allocSharedMem(tflite::kDefaultTensorAlignment, tensor->bytes);
                    if (out_ptrs[i] == NULL)
                    {
                        LOG_INFO("Could not allocate Memory for ouput: %s\n", tensor->name);
                        return RETURN_FAIL;
                    }
                    interpreter->SetCustomAllocationForTensor(outputs[i], {out_ptrs[i], tensor->bytes});
                }
            }

            int input = inputs[0];
            /* get input dimension from the YAML parsed  and batch
            from input tensor assuming one tensor*/
            TfLiteIntArray *dims = interpreter->tensor(input)->dims;
            wanted_batch = dims->data[0];
            wanted_height = arg->modelInfo->m_preProcCfg.outDataHeight;
            wanted_width = arg->modelInfo->m_preProcCfg.outDataWidth;
            wanted_channels = arg->modelInfo->m_preProcCfg.numChans;
            /* assuming NHWC*/
            if (wanted_channels != dims->data[3])
            {
                LOG_ERROR("missmatch in YAML parsed wanted channels:%d and model channels:%d\n", wanted_channels, dims->data[3]);
            }
            if (wanted_height != dims->data[1])
            {
                LOG_ERROR("missmatch in YAML parsed wanted height:%d and model height:%d\n", wanted_height, dims->data[1]);
            }
            if (wanted_width != dims->data[2])
            {
                LOG_ERROR("missmatch in YAML parsed wanted width:%d and model width:%d\n", wanted_width, dims->data[2]);
            }
            cv::Mat img;
            switch (interpreter->tensor(input)->type)
            {
            case kTfLiteFloat32:
            {
                img = preprocImage<float>(input_img_path, &interpreter->typed_tensor<float>(input)[0], arg->modelInfo->m_preProcCfg);
                break;
            }
            case kTfLiteUInt8:
            {
                /* if model is already quantized update the scale and mean for
                preperocess computation */
                std::vector<float> temp_scale = arg->modelInfo->m_preProcCfg.scale,
                                   temp_mean = arg->modelInfo->m_preProcCfg.mean;
                arg->modelInfo->m_preProcCfg.scale = {1, 1, 1};
                arg->modelInfo->m_preProcCfg.mean = {0, 0, 0};
                img = preprocImage<uint8_t>(input_img_path, &interpreter->typed_tensor<uint8_t>(input)[0], arg->modelInfo->m_preProcCfg);
                /*restoring mean and scale to preserve the data */
                arg->modelInfo->m_preProcCfg.scale = temp_scale;
                arg->modelInfo->m_preProcCfg.mean = temp_mean;
                break;
            }
            default:
                LOG_ERROR("cannot handle input type %d yet\n", interpreter->tensor(input)->type);
                return RETURN_FAIL;
            }

            struct timeval start_time, stop_time;
            gettimeofday(&start_time, nullptr);
            if (interpreter->Invoke() != kTfLiteOk)
            {
                LOG_ERROR("Failed to invoke tflite!\n");
            }
            gettimeofday(&stop_time, nullptr);

            float avg_time = ((getUs(stop_time) - getUs(start_time)) / (1000));
            arg->actual_times[arg->model_id] = avg_time;
            return RETURN_SUCCESS;
        }


        /**
         *  \brief  Get the actual run time of model if ran individually
         *  \param  arg tfl_model_struct containing models details to be ran
         * @returns int status
         */
        int getActualRunTime(tfl_model_struct *arg0, tfl_model_struct *arg1){
            LOG_INFO("Fetching actual inference time for models\n");
            float total_m0,total_m1;
            int loop_count = 10;
            for (size_t i = 0; i < loop_count; i++)
            {
                getInvokeTime(arg0);
                total_m0 += arg0->actual_times[0];
                getInvokeTime(arg1);
                total_m1 += arg1->actual_times[1];
                LOG_INFO("Run times of model 0 and 1 are: %f %f\n",arg0->actual_times[0], arg1->actual_times[1]);
            }
            arg0->actual_times[0] = total_m0/loop_count;
            arg0->actual_times[1] = total_m1/loop_count;
            arg1->actual_times[0] = total_m0/loop_count;
            arg1->actual_times[1] = total_m1/loop_count;
            LOG_INFO("Done fetching actual inference time for model %s :avg_time %f\n", arg0->modelInfo->m_preProcCfg.modelName.c_str(), arg0->actual_times[0]);
            LOG_INFO("Done fetching actual inference time for model %s :avg_time %f\n", arg1->modelInfo->m_preProcCfg.modelName.c_str(), arg0->actual_times[1]);
             return RETURN_SUCCESS;   
        }
        
        /**
         *  \brief  Actual infernce happening
         *  \param  ModelInfo YAML parsed model info
         *  \param  Settings user input options  and default values of setting if any
         * @returns int
         */

        /**
         *  \brief  Thread function to carry out final model running
         *  \param  arguments struct for thread to run model
         * @returns exits as thread
         */
        void *infer(void *argument)
        {
            void *in_ptrs[16] = {NULL};
            void *out_ptrs[16] = {NULL};
            cv::Mat img;
            vector<int> inputs;
            vector<int> outputs;
            int wanted_batch, wanted_height, wanted_width, wanted_channels;
            struct timeval start_time, stop_time, iter_start, iter_end;
            /*TODO decide the max number of buffer need to kept for op tensor val*/
            /*temperory keeping: 10*/
            int max_num_op_saved = 10;

            tfl_model_struct *arg = (tfl_model_struct *)argument;
            pthread_mutex_lock(&tfl_pr_lock);
            std::unique_ptr<tflite::Interpreter> interpreter;
            tflite::InterpreterBuilder(*arg->model, arg->resolver)(&interpreter);
            pthread_mutex_unlock(&tfl_pr_lock);
            uint64_t thread_id = pthread_self();
            string input_img_path = arg->s->input_img_paths[arg->model_id];
            if (!interpreter)
            {
                LOG_ERROR("Failed to construct interpreter\n");
                pthread_exit(NULL);
            }
            inputs = interpreter->inputs();
            outputs = interpreter->outputs();
            LOG_INFO("tensors size: %d \n", interpreter->tensors_size());
            LOG_INFO("nodes size: %d\n", interpreter->nodes_size());
            LOG_INFO("number of inputs: %d\n", inputs.size());
            LOG_INFO("number of outputs: %d\n", outputs.size());
            LOG_INFO("input(0) name: %s\n", interpreter->GetInputName(0));

            if (inputs.size() != 1)
            {
                LOG_ERROR("Supports only single input models \n");
                pthread_exit(NULL);
            }
            if (arg->s->accel == 1)
            {
                /* This part creates the dlg_ptr */
                LOG_INFO("accelerated mode\n");
                typedef TfLiteDelegate *(*tflite_plugin_create_delegate)(char **, char **, size_t, void (*report_error)(const char *));
                tflite_plugin_create_delegate tflite_plugin_dlg_create;
                char prior_char[2], pre_empt_char[64];
                std::sprintf(prior_char, "%d", arg->priority);
                if (arg->s->max_pre_empts[arg->model_id] == -1)
                    arg->s->max_pre_empts[arg->model_id] = FLT_MAX;
                std::sprintf(pre_empt_char, "%f", arg->s->max_pre_empts[arg->model_id]);
                char *keys[] = {(char *)"artifacts_folder", (char *)"num_tidl_subgraphs", (char *)"debug_level", (char *)"priority", (char *)"max_pre_empt_delay"};
                char *values[] = {(char *)arg->modelInfo->m_infConfig.artifactsPath.c_str(), (char *)"16", (char *)"0", (char *)prior_char, (char *)pre_empt_char};
                void *lib = dlopen("libtidl_tfl_delegate.so", RTLD_NOW);
                assert(lib);
                tflite_plugin_dlg_create = (tflite_plugin_create_delegate)dlsym(lib, "tflite_plugin_create_delegate");
                TfLiteDelegate *dlg_ptr = tflite_plugin_dlg_create(keys, values, 5, NULL);
                interpreter->ModifyGraphWithDelegate(dlg_ptr);
                LOG_INFO("ModifyGraphWithDelegate - Done \n");
            }
            if (interpreter->AllocateTensors() != kTfLiteOk)
            {
                LOG_ERROR("Failed to allocate tensors!");
                pthread_exit(NULL);
            }
            if (arg->s->device_mem)
            {
                LOG_INFO("device mem enabled\n");
                for (uint32_t i = 0; i < inputs.size(); i++)
                {
                    const TfLiteTensor *tensor = interpreter->input_tensor(i);
                    in_ptrs[i] = TIDLRT_allocSharedMem(tflite::kDefaultTensorAlignment, tensor->bytes);
                    if (in_ptrs[i] == NULL)
                    {
                        LOG_INFO("Could not allocate Memory for input: %s\n", tensor->name);
                    }
                    interpreter->SetCustomAllocationForTensor(inputs[i], {in_ptrs[i], tensor->bytes});
                }
                for (uint32_t i = 0; i < max_num_op_saved; i++)
                {
                    const TfLiteTensor *tensor = interpreter->output_tensor(0);
                    out_ptrs[i] = TIDLRT_allocSharedMem(tflite::kDefaultTensorAlignment, tensor->bytes);
                    if (out_ptrs[i] == NULL)
                    {
                        LOG_INFO("Could not allocate Memory for ouput: %s\n", tensor->name);
                        pthread_exit(NULL);
                    }
                }
            }

            /* get input dimension from the YAML parsed  and batch
            from input tensor assuming one tensor*/
            TfLiteIntArray *dims = interpreter->tensor(inputs[0])->dims;
            wanted_batch = dims->data[0];
            wanted_height = arg->modelInfo->m_preProcCfg.outDataHeight;
            wanted_width = arg->modelInfo->m_preProcCfg.outDataWidth;
            wanted_channels = arg->modelInfo->m_preProcCfg.numChans;
            /* assuming NHWC*/
            if (wanted_channels != dims->data[3])
            {
                LOG_INFO("missmatch in YAML parsed wanted channels:%d and model channels:%d\n", wanted_channels, dims->data[3]);
            }
            if (wanted_height != dims->data[1])
            {
                LOG_INFO("missmatch in YAML parsed wanted height:%d and model height:%d\n", wanted_height, dims->data[1]);
            }
            if (wanted_width != dims->data[2])
            {
                LOG_INFO("missmatch in YAML parsed wanted width:%d and model width:%d\n", wanted_width, dims->data[2]);
            }
            switch (interpreter->tensor(inputs[0])->type)
            {
            case kTfLiteFloat32:
            {
                img = preprocImage<float>(input_img_path, &interpreter->typed_tensor<float>(inputs[0])[0], arg->modelInfo->m_preProcCfg);
                break;
            }
            case kTfLiteUInt8:
            {
                /* if model is already quantized update the scale and mean for
                preperocess computation */
                std::vector<float> temp_scale = arg->modelInfo->m_preProcCfg.scale,
                                   temp_mean = arg->modelInfo->m_preProcCfg.mean;
                arg->modelInfo->m_preProcCfg.scale = {1, 1, 1};
                arg->modelInfo->m_preProcCfg.mean = {0, 0, 0};
                img = preprocImage<uint8_t>(input_img_path, &interpreter->typed_tensor<uint8_t>(inputs[0])[0], arg->modelInfo->m_preProcCfg);
                /*restoring mean and scale to preserve the data */
                arg->modelInfo->m_preProcCfg.scale = temp_scale;
                arg->modelInfo->m_preProcCfg.mean = temp_mean;
                break;
            }
            default:
                LOG_ERROR("cannot handle input type %d yet\n", interpreter->tensor(inputs[0])->type);
            }
            pthread_barrier_wait(&barrier);
            gettimeofday(&start_time, nullptr);
            auto finish = system_clock::now() + minutes{1};
            int k = 0, num_switches = 0, exceeded_iter = 0;
            float time_spend, min_time_spend = FLT_MAX, max_time_spend = 0;
            float curr_actual_time = arg->actual_times[arg->model_id];
            float pll_actual_time = arg->actual_times[(arg->model_id+1) % NUM_PARLLEL_MODELS];
            float pll_max_pre_empt_dealy = arg->s->max_pre_empts[(arg->model_id+1) % NUM_PARLLEL_MODELS];
            const TfLiteTensor *tensor = interpreter->output_tensor(0);
            interpreter->SetCustomAllocationForTensor(outputs[0], {out_ptrs[0], tensor->bytes});
            /* plotting graph */
            RGBABitmapImageReference *imageReference = CreateRGBABitmapImageReference();
            StringReference *errorMessage = new StringReference();
            do
            {
                gettimeofday(&iter_start, nullptr);
                if (interpreter->Invoke() != kTfLiteOk)
                {
                    LOG_ERROR("Failed to invoke tflite!\n");
                    pthread_exit(NULL);
                }
                gettimeofday(&iter_end, nullptr);
                time_spend = (float)((getUs(iter_end) - getUs(iter_start)) / (1000));
                arg->ys.push_back(time_spend);
                arg->xs.push_back(k);
                if(time_spend >( curr_actual_time + pll_max_pre_empt_dealy ))
                   exceeded_iter++;
                if(time_spend < min_time_spend)
                   min_time_spend = time_spend; 
                if(time_spend > max_time_spend)
                   max_time_spend = time_spend; 

                if (time_spend > curr_actual_time + pll_actual_time)
                {
                    if(num_switches < max_num_op_saved){
                        /* allocating 0th tensor from out_pts array at invoke time
                        each time it exceeds its actual inference time */
                        const TfLiteTensor *tensor = interpreter->output_tensor(0);
                        interpreter->SetCustomAllocationForTensor(outputs[0], {out_ptrs[num_switches % max_num_op_saved], tensor->bytes});
                    }
                    num_switches++;
                }
                k++;
            } while (system_clock::now() < finish);
            gettimeofday(&stop_time, nullptr);
            /*plotting graph*/
            DrawScatterPlot(imageReference, 600, 400, &(arg->xs), &(arg->ys),errorMessage);
            vector<double> *pngdata = ConvertToPNG(imageReference->image);
            char graph_name[20];
            sprintf(graph_name,"model%d_graph.png",arg->model_id);
            WriteToFile(pngdata, graph_name);
            DeleteImage(imageReference->image);

            float avg_time = ((getUs(stop_time) - getUs(start_time)) / (k * 1000));
            LOG_INFO("Model %s start time %ld.%06ld\n", arg->modelInfo->m_preProcCfg.modelName.c_str(), start_time.tv_sec, start_time.tv_usec);
            LOG_INFO("Model %s stop time %ld.%06ld\n", arg->modelInfo->m_preProcCfg.modelName.c_str(), start_time.tv_sec, start_time.tv_usec);
            LOG_INFO("Total num context switches for model %s:%d \n", arg->modelInfo->m_preProcCfg.modelName.c_str(), num_switches);
            LOG_INFO("Total num iterations run for model %s:%d \n", arg->modelInfo->m_preProcCfg.modelName.c_str(), k);
            LOG_INFO("Total %d iterations for model %s exceeded the expected time. max:%f min:%fms avg:%fms  \n",exceeded_iter, arg->modelInfo->m_preProcCfg.modelName.c_str(), max_time_spend, min_time_spend, avg_time); 
            LOG_INFO("FPS for model %s :%f \n", arg->modelInfo->m_preProcCfg.modelName.c_str(), (float)((float)k / 60));

            if (arg->modelInfo->m_preProcCfg.taskType == "classification")
            {
                if (RETURN_FAIL == prepClassificationResult(&img, &interpreter, &outputs, arg->s))
                {
                    pthread_exit(NULL);
                }
                TfLiteIntArray *output_dims = interpreter->tensor(outputs[0])->dims;
                size_t output_tensor_length = output_dims->data[output_dims->size - 1];
                TfLiteType type = interpreter->tensor(outputs[0])->type;
                int count = (num_switches >= max_num_op_saved)?max_num_op_saved:num_switches;
                if (type == TfLiteType::kTfLiteInt32)
                {
                    if (0 != compTensorOut<int32_t>(count, output_tensor_length, &interpreter, out_ptrs))
                    {
                        LOG_ERROR("Compare op tensor iterations : failed\n");
                        pthread_exit(NULL);
                    }else{
                        LOG_INFO("Compare op tensor iterations : success\n");
                    }
                }
                else if (type == TfLiteType::kTfLiteFloat32)
                {
                    if (0 != compTensorOut<float>(count, output_tensor_length, &interpreter, out_ptrs))
                    {
                        LOG_ERROR("Compare op tensor iterations : failed\n");
                        pthread_exit(NULL);
                    }else{
                        LOG_INFO("Compare op tensor iterations : success\n");
                    }
                }
                else
                {
                    LOG_ERROR("op tensor type not supprted type: %d\n");
                    pthread_exit(NULL);
                }
            }
            else if (arg->modelInfo->m_preProcCfg.taskType == "segmentation")
            {
                float alpha = arg->modelInfo->m_postProcCfg.alpha;
                if (RETURN_FAIL == prepSegResult(&img, wanted_width, wanted_height, alpha, &interpreter, &outputs))
                {
                    LOG_ERROR("prepSegResult failed\n ");
                    pthread_exit(NULL);
                }
                size_t output_tensor_length = wanted_height * wanted_width;
                TfLiteType type = interpreter->tensor(outputs[0])->type;
                int count = (num_switches >= max_num_op_saved)?max_num_op_saved:num_switches;
                if (type == TfLiteType::kTfLiteUInt8)
                {
                    if (0 != compTensorOut<uint8_t>(count, output_tensor_length, &interpreter, out_ptrs))
                    {
                        LOG_ERROR("Compare op tensor iterations : failed\n");
                        pthread_exit(NULL);
                    }else{
                        LOG_INFO("Compare op tensor iterations : success\n");
                    }
                }
                else if (type == TfLiteType::kTfLiteFloat32)
                {
                    if (0 != compTensorOut<float>(count, output_tensor_length, &interpreter, out_ptrs))
                    {
                        LOG_ERROR("Compare op tensor iterations : failed\n");
                        pthread_exit(NULL);
                    }else{
                        LOG_INFO("Compare op tensor iterations : success\n");
                    }
                }
                else
                {
                    LOG_ERROR("op tensor type not supprted type: %d\n");
                    pthread_exit(NULL);
                }
            }
            else if (arg->modelInfo->m_preProcCfg.taskType == "detection")
            {
                /*store tensor_shape info of op tensors in arr
                        to avaoid recalculation*/
                int num_ops = outputs.size();
                vector<vector<float>> f_tensor_unformatted;
                /*num of detection in op tensor is assumed to be given by last tensor*/
                int nboxes;
                if (interpreter->tensor(outputs[num_ops - 1])->type == kTfLiteFloat32)
                    nboxes = (int)*interpreter->tensor(outputs[num_ops - 1])->data.f;
                else if (interpreter->tensor(outputs[num_ops - 1])->type == kTfLiteInt64)
                    nboxes = (int)*interpreter->tensor(outputs[num_ops - 1])->data.i64;
                else
                {
                    LOG_ERROR("unknown type for op tensor:%d\n", num_ops - 1);
                    pthread_exit(NULL);
                    ;
                }
                LOG_INFO("detected objects:%d \n", nboxes);
                /* TODO verify this holds true for every tfl model*/
                vector<vector<int64_t>> tensor_shapes_vec = {{nboxes, 4}, {nboxes, 1}, {nboxes, 1}, {nboxes, 1}};
                /* TODO Incase of only single tensor op od-2110 above tensor shape is
                invalid*/

                /* run through all tensors excpet last one which contain
                num_of detected boxes */
                for (size_t i = 0; i < num_ops - 1; i++)
                {
                    /* temp vector to store converted ith tensor */
                    vector<float> f_tensor;
                    /* shape of the ith tensor*/
                    vector<int64_t> tensor_shape = tensor_shapes_vec[i];

                    /* type of the ith tensor*/
                    TfLiteType tensor_type = interpreter->tensor(outputs[i])->type;
                    /* num of values in ith tensor is assumed to be the tensor's
                    shape in tflite*/
                    int num_val_tensor = tensor_shape[tensor_shape.size() - 1];
                    /*convert tensor to float vector*/
                    if (tensor_type == kTfLiteFloat32)
                    {
                        float *inDdata = interpreter->tensor(outputs[i])->data.f;
                        createFloatVec<float>(inDdata, &f_tensor, tensor_shape);
                    }
                    else if (tensor_type == kTfLiteInt64)
                    {
                        int64_t *inDdata = interpreter->tensor(outputs[i])->data.i64;
                        createFloatVec<int64_t>(inDdata, &f_tensor, tensor_shape);
                    }
                    else if (tensor_type == kTfLiteInt32)
                    {
                        int32_t *inDdata = (int32_t *)interpreter->tensor(outputs[i])->data.data;
                        createFloatVec<int32_t>(inDdata, &f_tensor, tensor_shape);
                    }
                    else
                    {
                        LOG_ERROR("out tensor data type not supported %d\n", tensor_type);
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
                if (RETURN_FAIL == prepDetectionResult(&img, &f_tensor_unformatted, tensor_shapes_vec, arg->modelInfo, num_ops - 1, nboxes))
                    pthread_exit(NULL);
            }

            LOG_INFO("saving image result file \n");
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
            foldername = foldername + "tfl-cpp/";
            if (stat(foldername.c_str(), &buffer) != 0)
            {
                if (mkdir(foldername.c_str(), 0777) == -1)
                {
                    LOG_ERROR("failed to create folder %s:%s\n", foldername, strerror(errno));
                    pthread_exit(NULL);
                }
            }
            filename = "post_proc_out_";
            filename = filename + arg->modelInfo->m_preProcCfg.modelName.c_str();
            filename = filename + ".jpg";
            foldername = foldername + filename;
            if (false == cv::imwrite(foldername, img))
            {
                LOG_INFO("Saving the image, FAILED\n");
                pthread_exit(NULL);
            }
            if (arg->s->device_mem)
            {
                for (uint32_t i = 0; i < inputs.size(); i++)
                {
                    if (in_ptrs[i])
                    {
                        TIDLRT_freeSharedMem(in_ptrs[i]);
                    }
                }
                for (uint32_t i = 0; i < outputs.size(); i++)
                {
                    if (out_ptrs[i])
                    {
                        TIDLRT_freeSharedMem(out_ptrs[i]);
                    }
                }
            }

            LOG_INFO("\nCompleted_Model : 0, Name : %s, Total time : %f, Offload Time : 0 , DDR RW MBs : 0, Output File : %s \n \n",
                     arg->modelInfo->m_postProcCfg.modelName.c_str(), avg_time, filename.c_str());

            /*exit the current thread*/
            pthread_exit(NULL);
        }

        int runInference(ModelInfo **modelInfos, Settings *s)
        {
            int ret;
            pthread_attr_t tattr;
            pthread_t tid;
            pthread_t ptid[2 * s->number_of_threads];
            /* preparing tflite model  from file*/
            std::unique_ptr<tflite::FlatBufferModel> model[NUM_PARLLEL_MODELS];
            tflite::ops::builtin::BuiltinOpResolver resolver[NUM_PARLLEL_MODELS];
            tfl_model_struct args[NUM_PARLLEL_MODELS];
            for (size_t i = 0; i < NUM_PARLLEL_MODELS; i++)
            {
                LOG_INFO("Prep model %d\n", i);
                /* checking model path present or not*/
                if (!modelInfos[i]->m_infConfig.modelFile.c_str())
                {
                    LOG_ERROR("no model file name\n");
                    return RETURN_FAIL;
                }

                model[i] = tflite::FlatBufferModel::BuildFromFile(modelInfos[i]->m_infConfig.modelFile.c_str());
                if (!model)
                {
                    LOG_ERROR("\nFailed to mmap model %s\n", modelInfos[i]->m_infConfig.modelFile);
                    return RETURN_FAIL;
                }
                LOG_INFO("Loaded model %s \n", modelInfos[i]->m_infConfig.modelFile.c_str());
                model[i]->error_reporter();
                LOG_INFO("resolved reporter\n");
                args[i].modelInfo = modelInfos[i];
                args[i].model = move(model[i]);
                args[i].resolver = resolver[i];
                args[i].s = s;
                args[i].model_id = i;
                args[i].priority = s->priors[i];
                // getActualRunTime(&args[i]);
            }
            getActualRunTime(&args[0],&args[1]);
            if (pthread_mutex_init(&tfl_pr_lock, NULL) != 0)
            {
                LOG_ERROR("\n mutex init has failed\n");
                return RETURN_FAIL;
            }

            /* initialized with default attributes */
            ret = pthread_attr_init(&tattr);
            /*iniitalizing barrier */
            pthread_barrierattr_t barr_attr;
            ret = pthread_barrier_init(&barrier, &barr_attr, (2 * s->number_of_threads));
            if (ret != 0)
            {
                LOG_ERROR("barrier creation failied exiting\n");
                return RETURN_FAIL;
            }
            for (size_t i = 0; i < s->number_of_threads; i++)
            {
                /* Creating a new thread*/
                pthread_create(&ptid[2 * i], &tattr, &infer, &args[0]);
                pthread_create(&ptid[2 * i + 1], &tattr, &infer, &args[1]);
            }
            for (size_t i = 0; i < s->number_of_threads; i++)
            {
                /*Waiting for the created thread to terminate*/
                pthread_join(ptid[i], NULL);
                pthread_join(ptid[2 * i + 1], NULL);
            }
            pthread_barrierattr_destroy(&barr_attr);
            pthread_mutex_destroy(&tfl_pr_lock);
            return RETURN_SUCCESS;
        }

    } // namespace main
} // namespace tflite

int main(int argc, char **argv)
{
    Settings s;
    if (parseArgs(argc, argv, &s) == RETURN_FAIL)
    {
        LOG_ERROR("Failed to parse the args\n");
        return RETURN_FAIL;
    }
    if (s.log_level <= 1)
        dumpArgs(&s);
    /* Parse the model YAML file */
    ModelInfo model1(s.model_paths[0]);
    if (model1.initialize(s.log_level) == RETURN_FAIL)
    {
        LOG_ERROR("Failed to initialize model\n");
        return RETURN_FAIL;
    }

    /* Parse the model YAML file */
    ModelInfo model2(s.model_paths[1]);
    if (model2.initialize(s.log_level) == RETURN_FAIL)
    {
        LOG_ERROR("Failed to initialize model\n");
        return RETURN_FAIL;
    }
    ModelInfo *model_infos[NUM_PARLLEL_MODELS];
    model_infos[0] = &model1;
    model_infos[1] = &model2;

    if (tflite::main::runInference(model_infos, &s) == RETURN_FAIL)
    {
        LOG_ERROR("Failed to run runInference\n");
        return RETURN_FAIL;
    }
    return RETURN_SUCCESS;
}
