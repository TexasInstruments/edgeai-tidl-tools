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
#include <dlpack/dlpack.h>

namespace dlr
{
    namespace main
    {
        /**
         *  \brief  get the  output tensors
         *  \param  outputs
         *  \param  num_outputs num of outputs
         *  \param  model DLR model
         * @returns void
         */
        template <class T>
        int fetchOutputTensors(std::vector<std::vector<T>> &outputs,
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
                    return RETURN_FAIL;
                }
            }
            return RETURN_SUCCESS;
        }

        template int fetchOutputTensors<float>(std::vector<std::vector<float>> &outputs,
                                               int num_outputs,
                                               DLRModelHandle model);
        template int fetchOutputTensors<int64_t>(std::vector<std::vector<int64_t>> &outputs,
                                                 int num_outputs,
                                                 DLRModelHandle model);

        /**
         *  \brief  get the tensor type of input and output
         *  \param  index index of tensor to find the type
         *  \param  isInput bool to determine input/output tensor to fetch teh details
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
         *  \brief  prepare the classification result inplace
         *  \param  img cv image to do inplace transform
         *  \param  s settings struct pointer
         *  \param  model DLR model Handle
         *  \param  num_outputs
         * @returns int status
         */
        int prepClassificationResult(cv::Mat *img, Settings *s, DLRModelHandle model, int num_outputs, string output_binary)
        {
            LOG_INFO("preparing classification result \n");
            cv::resize((*img), (*img), cv::Size(512, 512), 0, 0, cv::INTER_AREA);
            const float threshold = 0.001f;
            std::vector<std::pair<float, int>> top_results;
            /* get tensor type */
            const char *output_type = getTensorType(0, false, model);
            if (!strcmp(output_type, "float32"))
            {
                std::vector<std::vector<float>> outputs;
                if (RETURN_FAIL == fetchOutputTensors<float>(outputs, num_outputs, model))
                    return RETURN_FAIL;
                float *tensor_op_array = outputs[0].data();
                /*assuming 1 output vector */
                getTopN<float>(outputs[0].data(),
                               1000, s->number_of_results, threshold,
                               &top_results, true);

                // Writing tensor data to binary file
                int output_size = outputs[0].size();
                ofstream fout(output_binary, ios::binary);
                fout.write(reinterpret_cast<char*>(tensor_op_array), output_size * sizeof(float));
                fout.close();

                std::vector<std::string> labels;
                size_t label_count;

                if (readLabelsFile(s->labels_file_path, &labels, &label_count) != 0)
                {
                    LOG_ERROR("Failed to load labels file\n");
                    return RETURN_FAIL;
                }

                int outputoffset;
                if (output_size == 1001)
                    outputoffset = 0;
                else
                    outputoffset = 1;
                for (const auto &result : top_results)
                {
                    const float confidence = result.first;
                    const int index = result.second;

                    LOG_INFO("%f: %d %s\n", confidence, index, labels[index + outputoffset].c_str());
                }
                int num_results = s->number_of_results;
                (*img).data = overlayTopNClasses((*img).data, top_results, &labels, (*img).cols, (*img).rows, num_results, outputoffset);
            }
            else
            {
                LOG_ERROR("output type not supported %s\n", *output_type);
                return RETURN_FAIL;
            }
            return RETURN_SUCCESS;
        }

        /**
         *  \brief  prepare the segmentation result inplace
         *  \param  img cv image to do inplace transform
         *  \param  model DLR model Handle
         *  \param  num_outputs
         *  \param  mdoelInfo pointer to modelInfo
         *  \param  wanted_width
         *  \param  wanted_height
         * @returns int status
         */
        int prepSegResult(cv::Mat *img, DLRModelHandle model, int num_outputs,
                          ModelInfo *modelInfo, int wanted_width, int wanted_height, string output_binary)
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
                cv::resize((*img), (*img), cv::Size(wanted_width, wanted_height), 0, 0, cv::INTER_AREA);
            }
            /* determine the output type */
            const char *output_type = getTensorType(0, false, model);
            if (!strcmp(output_type, "int64"))
            {
                std::vector<std::vector<int64_t>> outputs;
                fetchOutputTensors<int64_t>(outputs, num_outputs, model);
                (*img).data = blendSegMask<int64_t>((*img).data, outputs[0].data(), (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
                ofstream fout(output_binary, ios::binary);
                fout.write(reinterpret_cast<char*>(outputs[0].data()), outputs[0].size() * sizeof(int64_t));
                fout.close();
            }
            else if (!strcmp(output_type, "float32"))
            {
                std::vector<std::vector<float>> outputs;
                fetchOutputTensors<float>(outputs, num_outputs, model);
                (*img).data = blendSegMask<float>((*img).data, outputs[0].data(), (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
                ofstream fout(output_binary, ios::binary);
                fout.write(reinterpret_cast<char*>(outputs[0].data()), outputs[0].size() * sizeof(float));
                fout.close();
            }
            else if (!strcmp(output_type, "uint8"))
            {
                std::vector<std::vector<uint8_t>> outputs;
                fetchOutputTensors<uint8_t>(outputs, num_outputs, model);
                (*img).data = blendSegMask<uint8_t>((*img).data, outputs[0].data(), (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
                ofstream fout(output_binary, ios::binary);
                fout.write(reinterpret_cast<char*>(outputs[0].data()), outputs[0].size() * sizeof(uint8_t));
                fout.close();
            }
            else
            {
                LOG_ERROR("output type not supported %s\n", *output_type);
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
                device_type = 2;
            }
            else
            {
                LOG_ERROR("device type not supported: %s", s->device_type.c_str());
                return RETURN_FAIL;
            }
            if (CreateDLRModel(&model, modelInfo->m_infConfig.artifactsPath.c_str(), device_type, 0) != 0)
            {
                LOG_ERROR("Could not load DLR Model\n");
                return RETURN_FAIL;
            }

            /*input vector infering -assuming single input*/
            GetDLRNumInputs(&model, &num_inputs);
            if (num_inputs != 1)
            {
                LOG_ERROR("Model with more than one input not supported\n");
                return RETURN_FAIL;
            }
            /* Query input name. */
            const char *input_name{nullptr};
            int status = GetDLRInputName(&model,
                                         0,
                                         &input_name);

            if (status < 0)
            {
                LOG_ERROR("GetDLRInputName(0) failed. Error [%s].\n",
                          DLRGetLastError());
                return RETURN_FAIL;
            }
            LOG_INFO("%s :input name\n", input_name);
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
                return RETURN_FAIL;
            }

            LOG_INFO("Inference call started...\n");
            cv::Mat img;
            float *image_data;
            if(s->accel && s->device_mem){
                #ifdef DEVICE_AM62
                LOG_ERROR("TIDL Delgate mode is not allowed on AM62 devices...\n");
                return RETURN_FAIL;
                #else
                image_data = (float *)TIDLRT_allocSharedMem(128, sizeof(float) * wanted_height * wanted_width * wanted_channels);
                #endif
            }else{
                image_data = (float*)calloc(wanted_height * wanted_width * wanted_channels,sizeof(float) );
            }
            if (image_data == NULL)
            {
                LOG_ERROR("could not allocate space for image data \n");
                return RETURN_FAIL;
            }
            LOG_INFO("Input tetsor Pointer - %p \n", image_data);
            const char *input_type_feild[] = {};
            const char **input_type = &input_type_feild[0];

            GetDLRInputType(&model, 0, input_type);
            if (!strcmp(*input_type, "float32"))
            {
                img = preprocImage<float>(s->input_image_path, image_data, modelInfo->m_preProcCfg);
            }
            else if (!strcmp(*input_type, "uint8"))
            {
                img = preprocImage<uint8_t>(s->input_image_path, (uint8_t*)image_data, modelInfo->m_preProcCfg);
            }
            else
            {
                LOG_ERROR("cannot handle input type %s yet", *input_type);
                return RETURN_FAIL;
            }
            LOG_INFO("Classifying input:%s\n", s->input_image_path.c_str());

            /*Running inference */

            DLTensor dltensor;
            dltensor.device = {kDLCPU, 0};
            dltensor.ndim    = 4;
            dltensor.shape   = input_shape;
            dltensor.strides = nullptr;
            dltensor.byte_offset = 0;
            dltensor.dtype = {kDLUInt, static_cast<uint8_t>(8), 1};
            dltensor.data = image_data;

            if (SetDLRInputTensorZeroCopy(&model, input_name, &dltensor) != 0)
            //if (SetDLRInput(&model, input_name, input_shape, image_data, 4) != 0)
            {
                LOG_ERROR("Could not set input:%s\n", input_name);
                return RETURN_FAIL;
            }
            int num_iter = s->loop_count;
            if (s->loop_count >= 1)
            {
                LOG_INFO("Session.Run() - Started for warmup runs\n");
                for (int i = 0; i < s->number_of_warmup_runs; i++)
                {
                    if (RunDLRModel(&model) != 0)
                    {
                        LOG_ERROR("Could not run\n");
                    }
                }
            }
            struct timeval start_time, stop_time;
            gettimeofday(&start_time, nullptr);
            for (int i = 0; i < num_iter; i++)
            {
                if (RunDLRModel(&model) != 0)
                {
                    LOG_ERROR("Could not run\n");
                }
            }
            gettimeofday(&stop_time, nullptr);
            float avg_time = (getUs(stop_time) - getUs(start_time)) / (num_iter * 1000);
            LOG_INFO("average time: %lf ms \n", avg_time);

            /*output vector infering*/
            GetDLRNumOutputs(&model, &num_outputs);

            /* Create folder to dump tensors */
            string bin_filename, bin_foldername;
            bin_foldername = bin_foldername +  "output_binaries/";
            struct stat binary_folder_buffer;
            if (stat(bin_foldername.c_str(), &binary_folder_buffer) != 0)
            {
                if (mkdir(bin_foldername.c_str(), 0777) == -1)
                {
                    LOG_ERROR("failed to create folder %s:%s\n", bin_foldername, strerror(errno));
                    return RETURN_FAIL;
                }
            }
            if (stat(bin_foldername.c_str(), &binary_folder_buffer) != 0)
            {
                if (mkdir(bin_foldername.c_str(), 0777) == -1)
                {
                    LOG_ERROR("failed to create folder %s:%s\n", bin_foldername, strerror(errno));
                    return RETURN_FAIL;
                }
            }
            bin_filename = "cpp_out_";
            bin_filename = bin_filename + modelInfo->m_preProcCfg.modelName.c_str();
            bin_filename = bin_filename + ".bin";
            bin_foldername = bin_foldername + bin_filename;

            if (modelInfo->m_preProcCfg.taskType == "classification")
            {
                if (RETURN_FAIL == prepClassificationResult(&img, s, model, num_outputs, bin_foldername))
                    return RETURN_FAIL;
            }
            else if (modelInfo->m_preProcCfg.taskType == "detection")
            {

                /*store tensor_shape info of op tensors in arr
               to avaoid recalculation*/
                vector<vector<int64_t>> tensor_shapes_vec;
                vector<int64_t> tensor_size_vec;
                vector<vector<float>> f_tensor_unformatted;

                for (size_t i = 0; i < num_outputs; i++)
                {
                    int64_t output_size = 0;
                    int output_dim = 0;
                    int64_t output_shape[output_dim];
                    GetDLROutputSizeDim(&model, i, &output_size, &output_dim);
                    GetDLROutputShape(&model, i, output_shape);
                    vector<int64_t> tensor_shape;
                    tensor_size_vec.push_back(output_size);
                    for (size_t k = 0; k < output_dim; k++)
                    {
                        tensor_shape.push_back(output_shape[k]);
                    }

                    tensor_shapes_vec.push_back(tensor_shape);
                }
                /* num of detection in op tensor  assumes the size of
                1st tensor*/
                int64_t nboxes;
                int output_dim = 0;
                GetDLROutputSizeDim(&model, 0, &nboxes, &output_dim);

                for (size_t i = 0; i < num_outputs; i++)
                {
                    /* temp vector to store converted ith tensor */
                    vector<float> f_tensor;
                    /* shape of the ith tensor*/
                    vector<int64_t> tensor_shape = tensor_shapes_vec[i];
                    /* type of the ith tensor*/
                    const char *tensor_type = getTensorType(i, false, model);
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
                    if (!strcmp(tensor_type, "float32"))
                    {

                        std::vector<float> output(nboxes * num_val_tensor, 0);
                        if (GetDLROutput(&model, i, output.data()) != 0)
                        {
                            LOG_ERROR("Could not get output:%d", i);
                            return RETURN_FAIL;
                        }
                        /* already in float vec no need to convert */
                        std::copy(output.begin(), output.end(), back_inserter(f_tensor));
                    }
                    else if (tensor_type == "int64")
                    {
                        std::vector<int64_t> output(nboxes * num_val_tensor, 0);
                        if (GetDLROutput(&model, i, output.data()) != 0)
                        {
                            LOG_ERROR("Could not get output:%d", i);
                            return RETURN_FAIL;
                        }
                        std::copy(output.begin(), output.end(), back_inserter(f_tensor));
                    }
                    else
                    {
                        LOG_ERROR("out tensor data type not supported: %s\n", tensor_type);
                        return RETURN_FAIL;
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
                /* Updating the format coz format is NULL in param.yaml
                format [x1y1 x2y2 label score]*/
                modelInfo->m_postProcCfg.formatter = {2, 3, 4, 5, 0, 1};
                modelInfo->m_postProcCfg.formatterName = "DetectionBoxSL2BoxLS";

                ofstream fout(bin_foldername, ios::binary);
                for (int i = 0; i < f_tensor_unformatted.size(); i++)
                {
                    for (int j = 0; j < f_tensor_unformatted[i].size(); j++)
                    {
                       fout.write(reinterpret_cast<char*>(&f_tensor_unformatted[i][j]), sizeof(float));
                    }
                }
                fout.close();

                if (RETURN_FAIL == prepDetectionResult(&img, &f_tensor_unformatted, tensor_shapes_vec, modelInfo, num_outputs, nboxes))
                    return RETURN_FAIL;
            }
            else if (modelInfo->m_preProcCfg.taskType == "segmentation")
            {
                if (RETURN_FAIL == prepSegResult(&img, model, num_outputs, modelInfo, wanted_width, wanted_height, bin_foldername))
                    return RETURN_FAIL;
            }

            /* Writing post processed image */
            cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            string filename, foldername;
            foldername = foldername +  "output_images/";
            struct stat buffer;
            if (stat(foldername.c_str(), &buffer) != 0)
            {
                if (mkdir(foldername.c_str(), 0777) == -1)
                {
                    LOG_ERROR("failed to create folder %s:%s\n", foldername, strerror(errno));
                    return RETURN_FAIL;
                }
            }
            if (stat(foldername.c_str(), &buffer) != 0)
            {
                if (mkdir(foldername.c_str(), 0777) == -1)
                {
                    LOG_ERROR("failed to create folder %s:%s\n", foldername, strerror(errno));
                    return RETURN_FAIL;
                }
            }
            filename = "cpp_out_";
            filename = filename + modelInfo->m_preProcCfg.modelName.c_str();
            filename = filename + ".jpg";
            foldername = foldername + filename;
            if (false == cv::imwrite(foldername, img))
            {
                LOG_INFO("Saving the image, FAILED\n");
                return RETURN_FAIL;
            }
            if(s->accel && s->device_mem){
                #ifndef DEVICE_AM62
                TIDLRT_freeSharedMem(image_data);
                #endif
            }else{
                free(image_data);
            }
            
            LOG_INFO("\nCompleted_Model : 0, Name : %s, Total time : %f, Offload Time : 0 , DDR RW MBs : 0, Output File : %s \n \n",
                     modelInfo->m_postProcCfg.modelName.c_str(), avg_time, filename.c_str());
            return RETURN_SUCCESS;
        }
    } // namespace main
} // namespace dlr

int main(int argc, char **argv)
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
    ModelInfo model(s.artifact_path);
    if (model.initialize() == RETURN_FAIL)
    {
        LOG_ERROR("Failed to initialize model\n");
        return RETURN_FAIL;
    }
    if (dlr::main::runInference(&model, &s) == RETURN_FAIL)
    {
        LOG_ERROR("Failed to run runInference\n");
        return RETURN_FAIL;
    }
    return RETURN_SUCCESS;
}
