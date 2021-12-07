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
#include "dlr_main.h"

namespace dlr
{
    namespace main
    {
        /**
  *  \brief  get the tensor type of input and output 
  *  \param  outputs 
  *  \param  inInput bool to determine input/output tensor to fetch teh details
  *          for
  *  \param  model DLR model
  * @returns const char* comataining the type name
  */
        template <class T>
        void fetchOutputTensors(std::vector<std::vector<T>> &outputs,
                                int num_outputs,
                                DLRModelHandle model)
        {

            for (int i = 0; i < num_outputs; i++)
            {
                int64_t cur_size = 0;
                int cur_dim = 0;
                GetDLROutputSizeDim(&model, i, &cur_size, &cur_dim);
                std::vector<T> output(cur_size, 0);
                outputs.push_back(output);
            }
            for (int i = 0; i < num_outputs; i++)
            {
                if (GetDLROutput(&model, i, outputs[i].data()) != 0)
                {
                    LOG_ERROR("Could not get output:%s", std::to_string(i));
                }
            }
        }

        template void fetchOutputTensors<float>(std::vector<std::vector<float>> &outputs,
                                                int num_outputs,
                                                DLRModelHandle model);
        template void fetchOutputTensors<int64_t>(std::vector<std::vector<int64_t>> &outputs,
                                                  int num_outputs,
                                                  DLRModelHandle model);

        /**
  *  \brief  get the tensor type of input and output 
  *  \param  index index of tensor to find the type
  *  \param  inInput bool to determine input/output tensor to fetch teh details
  *          for
  *  \param  model DLR model
  * @returns const char* comataining the type name
  */
        const char *getTensorType(int index, bool isInput, DLRModelHandle model)
        {
            const char *output_type_feild[] = {};
            const char **output_type = &output_type_feild[0];
            if (isInput)
            {
                /* determine the output type */
                GetDLRInputType(&model, index, output_type);
            }
            /* determine the output tensor type at index  */
            else
            {
                /* determine the output type */
                GetDLROutputType(&model, index, output_type);
            }
            return *output_type;
        }

        /**
  *  \brief  Actual infernce happening 
  *  \param  ModelInfo YAML parsed model info
  *  \param  Settings user input options  and default values of setting if any
  * @returns void
  */
        void RunInference(tidl::modelInfo::ModelInfo *modelInfo, tidl::arg_parsing::Settings *s)
        {
            int num_outputs, num_inputs;
            /*Initial inference time*/
            double fp_ms_avg = 0.0;
            DLRModelHandle model;
            int device_type;
            if (s->device_type == "cpu")
            {
                device_type = 1;
            }
            else if (s->device_type == "gpu")
            {
                device_type = 1;
            }
            else
            {
                LOG_ERROR("device type not supported: %s", s->device_type.c_str());
                exit(-1);
            }
            if (CreateDLRModel(&model, modelInfo->m_infConfig.artifactsPath.c_str(), device_type, 0) != 0)
            {
                LOG_ERROR("Could not load DLR Model\n");
                exit(-1);
            }

            /*input vector infering -assuming single input*/
            GetDLRNumInputs(&model, &num_inputs);
            if (num_inputs != 1)
            {
                LOG_ERROR("Model with more than one input not supported\n");
                exit(-1);
            }
            /* Query input name. */
            const char* input_name{nullptr};
            int status = GetDLRInputName(&model,
                                      0,
                                      &input_name);

            if (status < 0)
            {
                LOG_ERROR("GetDLRInputName(0) failed. Error [%s].\n",
                           DLRGetLastError());
                exit(-1);
            }
            LOG_INFO("%s :input name\n",input_name);
            int64_t input_size = 0;
            int input_dim = 0;
            GetDLRInputSizeDim(&model, 0, &input_size, &input_dim);
            int64_t input_shape[input_dim];
            GetDLRInputShape(&model, 0, input_shape);
            int wanted_batch_size = input_shape[0];
            int wanted_height = modelInfo->m_preProcCfg.outDataHeight;
            int wanted_width = modelInfo->m_preProcCfg.outDataWidth;
            int wanted_channels = modelInfo->m_preProcCfg.numChans;
            if (modelInfo->m_preProcCfg.dataLayout == "NHWC")
            {
                if (wanted_channels != input_shape[3])
                {
                    LOG_INFO("missmatch in YAML parsed wanted channels:%d and model channels:%d\n", wanted_channels, input_shape[3]);
                }
                if (wanted_height != input_shape[1])
                {
                    LOG_INFO("missmatch in YAML parsed wanted height:%d and model height:%d\n", wanted_height, input_shape[1]);
                }
                if (wanted_width != input_shape[2])
                {
                    LOG_INFO("missmatch in YAML parsed wanted width:%d and model width:%d\n", wanted_width, input_shape[2]);
                }
            }
            else if (modelInfo->m_preProcCfg.dataLayout == "NCHW")
            {
                if (wanted_channels != input_shape[1])
                {
                    LOG_INFO("missmatch in YAML parsed wanted channels:%d and model:%d\n", wanted_channels, input_shape[1]);
                }
                if (wanted_height != input_shape[2])
                {
                    LOG_INFO("missmatch in YAML parsed wanted height:%d and model:%d\n", wanted_height, input_shape[2]);
                }
                if (wanted_width != input_shape[3])
                {
                    LOG_INFO("missmatch in YAML parsed wanted width:%d and model:%d\n", wanted_width, input_shape[3]);
                    ;
                }
            }
            else
            {
                LOG_ERROR("data layout not supported\n");
                exit(-1);
            }

            LOG_INFO("Inference call started...\n");
            cv::Mat img;
            float image_data[wanted_height * wanted_width * wanted_channels];
            const char *input_type_feild[] = {};
            const char **input_type = &input_type_feild[0];

            GetDLRInputType(&model, 0, input_type);
            if (!strcmp(*input_type, "float32"))
            {
                img = tidl::preprocess::preprocImage<float>(s->input_bmp_path, image_data, modelInfo->m_preProcCfg);
            }
            else
            {
                LOG_ERROR("cannot handle input type %s yet", *input_type);
                exit(-1);
            }
            LOG_INFO("Classifying input:%s\n", s->input_bmp_path.c_str());

            /*Running inference */
            if (SetDLRInput(&model, input_name, input_shape, image_data, 4) != 0)
            {
                LOG_ERROR("Could not set input:%s\n", input_name);
                exit(-1);
            }
            if (RunDLRModel(&model) != 0)
            {
                LOG_ERROR("Could not run\n");
            }
            /*output vector infering*/
            GetDLRNumOutputs(&model, &num_outputs);
            if (modelInfo->m_preProcCfg.taskType == "classification")
            {
                LOG_INFO("preparing classification result \n");
                const float threshold = 0.001f;
                std::vector<std::pair<float, int>> top_results;
                /* get tensor type */
                const char *output_type = getTensorType(0, false, model);
                if (!strcmp(output_type, "float32"))
                {
                    std::vector<std::vector<float>> outputs;
                    fetchOutputTensors<float>(outputs, num_outputs, model);
                    float *tensor_op_array = outputs[0].data();
                    /*assuming 1 output vector */
                    tidl::postprocess::get_top_n<float>(outputs[0].data(),
                                                        1000, s->number_of_results, threshold,
                                                        &top_results, true);
                    std::vector<std::string> labels;
                    size_t label_count;

                    if (tidl::postprocess::ReadLabelsFile(s->labels_file_path, &labels, &label_count) != 0)
                    {
                        LOG_ERROR("Failed to load labels file\n");
                        exit(-1);
                    }

                    for (const auto &result : top_results)
                    {
                        const float confidence = result.first;
                        const int index = result.second;
                        int outputoffset;
                        int output_size = outputs[0].size();
                        if (output_size == 1001)
                            outputoffset = 0;
                        else
                            outputoffset = 1;

                        LOG(INFO) << confidence << ": " << index << " " << labels[index + outputoffset] << "\n";
                    }
                    int num_results = s->number_of_results;
                    img.data = tidl::postprocess::overlayTopNClasses(img.data, top_results, &labels, img.cols, img.rows, num_results);
                }
                else
                {
                    LOG_ERROR("output type not supported %s\n", *output_type);
                    exit(-1);
                }
            }
            else if (modelInfo->m_preProcCfg.taskType == "detection")
            {
                LOG_INFO("preparing detection result \n");
                float threshold = modelInfo->m_vizThreshold;
                std::vector<int32_t> format = {0, 1, 2, 3, 4, 5};
                if (num_outputs == 3 && tidl::utility_functs::is_same_format(format, modelInfo->m_postProcCfg.formatter))
                {
                    /* get tensor type */
                    const char *output0_type = getTensorType(0, false, model);
                    const char *output1_type = getTensorType(1, false, model);
                    const char *output2_type = getTensorType(2, false, model);
                    if (!strcmp(output0_type, "float32") && !strcmp(output1_type, "float32") && !strcmp(output2_type, "float32"))
                    {

                        /* case of 3 outputs
                        [1, nbox, 1]- for class ,
                        [1,nbox,1] - for scores ,
                        [1, nbox , 4] - for cordinates */
                        std::vector<std::vector<float>> outputs;
                        fetchOutputTensors<float>(outputs, num_outputs, model);
                        float *bboxes = outputs[2].data();
                        float *labels = outputs[0].data();
                        float *scores = outputs[1].data();

                        /* determine output dimesion of 0th output to determine nbox */
                        int64_t output_size = 0;
                        int output_dim = 0;
                        GetDLROutputSizeDim(&model, 0, &output_size, &output_dim);
                        int64_t output_shape[output_dim];
                        GetDLROutputShape(&model, 0, output_shape);
                        int nboxes = output_shape[1];
                        std::list<float> detection_class_list, detectection_location_list, detectection_scores_list;
                        int num_detections = 0;
                        for (int i = 0; i < nboxes; i = i + 3)
                        {
                            if (scores[i] >= threshold)
                            {
                                num_detections++;
                                detectection_scores_list.push_back(scores[i]);
                                detection_class_list.push_back(labels[i]);
                                detectection_location_list.push_back(bboxes[i + 3] / wanted_width);
                                detectection_location_list.push_back(bboxes[i + 2] / wanted_width);
                                detectection_location_list.push_back(bboxes[i + 1] / wanted_width);
                                detectection_location_list.push_back(bboxes[i] / wanted_width);
                            }
                        }
                        float detectection_scores[detectection_scores_list.size()];
                        float detection_class[detection_class_list.size()];
                        float detectection_location[detectection_location_list.size()];
                        std::copy(detectection_scores_list.begin(), detectection_scores_list.end(), detectection_scores);
                        std::copy(detection_class_list.begin(), detection_class_list.end(), detection_class);
                        std::copy(detectection_location_list.begin(), detectection_location_list.end(), detectection_location);

                        LOG_INFO("results %d\n", num_detections);
                        tidl::postprocess::overlayBoundingBox(img, num_detections, detectection_location, detectection_scores, threshold);
                        for (int i = 0; i < num_detections; i++)
                        {
                            LOG_INFO("class %lf\n", detection_class[i]);
                            LOG_INFO("cordinates %lf %lf %lf %lf\n", detectection_location[i * 4], detectection_location[i * 4 + 1], detectection_location[i * 4 + 2], detectection_location[i * 4 + 3]);
                            LOG_INFO("score %lf\n", detectection_scores[i]);
                        }
                    }
                    else
                    {
                        LOG_ERROR("out put format not yet supported\n");
                        exit(-1);
                    }
                }
            }
            else if (modelInfo->m_preProcCfg.taskType == "segmentation")
            {
                LOG_INFO("preparing segmentation result \n");
                float alpha = modelInfo->m_postProcCfg.alpha;
                /* determining the shape of output0
                assuming 1 output of shape [1 , 1 , width , height]*/
                int64_t output_size = 0;
                int output_dim = 0;
                GetDLROutputSizeDim(&model, 0, &output_size, &output_dim);
                int64_t output_shape[output_dim];
                GetDLROutputShape(&model, 0, output_shape);
                /* if indata and out data is diff resize the image
                check whether img need to be resized based on out data
                asssuming out put format [1,1,,width,height]*/
                if (wanted_height != output_shape[2] || wanted_width != output_shape[3])
                {
                    LOG_INFO("Resizing image to match output dimensions\n");
                    wanted_height = output_shape[2];
                    wanted_width = output_shape[3];
                    cv::resize(img, img, cv::Size(wanted_width, wanted_height), 0, 0, cv::INTER_AREA);
                }
                /* determine the output type */
                const char *output_type = getTensorType(0, false, model);
                if (!strcmp(output_type, "int64"))
                {
                    std::vector<std::vector<int64_t>> outputs;
                    fetchOutputTensors<int64_t>(outputs, num_outputs, model);
                    img.data = tidl::postprocess::blendSegMask<int64_t>(img.data, outputs[0].data(), img.cols, img.rows, wanted_width, wanted_height, alpha);
                }
                else if (!strcmp(output_type, "float32"))
                {
                    std::vector<std::vector<float>> outputs;
                    fetchOutputTensors<float>(outputs, num_outputs, model);
                    img.data = tidl::postprocess::blendSegMask<float>(img.data, outputs[0].data(), img.cols, img.rows, wanted_width, wanted_height, alpha);
                }
                else
                {
                    LOG_ERROR("output type not supported %s\n", *output_type);
                    exit(-1);
                }
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
            LOG_INFO("Done\n");
        }
    } // namespace main
} // namespace dlr

int main(int argc, char **argv)
{
    tidl::arg_parsing::Settings s;
    tidl::arg_parsing::parse_args(argc, argv, &s);
    tidl::arg_parsing::dump_args(&s);
    tidl::utils::logSetLevel((tidl::utils::LogLevel)s.log_level);
    /* Parse the input configuration file*/
    tidl::modelInfo::ModelInfo model(s.model_zoo_path);
    int status = model.initialize();
    dlr::main::RunInference(&model, &s);
    return 0;
}
