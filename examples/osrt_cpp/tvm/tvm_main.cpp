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

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include "tvm_main.h"

namespace tvm
{
    namespace main
    {
        /**
         *  \brief  get the  output tensor size.
         *  \param  PackedFunc the get_output_info function.
         *  \param  int index of the output that we want size for.
         * @returns int representing the size.
         */
        int getOutputSize(tvm::runtime::PackedFunc get_output_fn, int idx) {
            // get the output size
            tvm::runtime::NDArray output_array = get_output_fn(idx);
            const DLTensor *tensor = output_array.operator->();

            // shape would be something like this:
            // [1, 224, 224, 3]
            // [batch size, height, width, channels]
            // for an rgb image, channels is 3.
            int output_size = 1;
            for (int i = 0; i < tensor->ndim; i++) {
                if (tensor->shape[i] >= 0) {
                    output_size *= tensor->shape[i];
                }
            }

            return output_size;
        }

        /**
         *  \brief  get the  output tensor dimension.
         *  \param  PackedFunc the get_output_info function.
         *  \param  int index of the output that we want dimensions for.
         * @returns int representing the dimension.
         */
        int getOutputDimension(tvm::runtime::PackedFunc get_output_fn, int idx) {
            // get the output dimension
            tvm::runtime::NDArray output_array = get_output_fn(idx);
	        const DLTensor *tensor = output_array.operator->();

            return tensor->ndim;
        }

        /**
         *  \brief  get the  output tensors
         *  \param  outputs
         *  \param  num_outputs num of outputs
         *  \param  gmod TVM runtime module handle
         * @returns void
         */
        template <class T>
        int fetchOutputTensors(std::vector<std::vector<T>> &outputs,
                               int num_outputs,
                               tvm::runtime::Module &gmod)
        {
            auto get_output_info_fn = gmod.GetFunction("get_output_info");
            auto get_output_fn = gmod.GetFunction("get_output");

            for (int i = 0; i < num_outputs; i++)
            {
                int64_t cur_size = 0;
                int cur_dim = 0;

                cur_size = getOutputSize(get_output_fn, i);
                cur_dim = getOutputDimension(get_output_fn, i);

                std::vector<T> output(cur_size, 0);
                outputs.push_back(output);
            }

            for (int i = 0; i < num_outputs; i++)
            {
                try {
                    tvm::runtime::NDArray output_tensor = get_output_fn(i);
                    output_tensor.CopyToBytes(outputs[i].data(), outputs[i].size() * sizeof(T));
                }
                catch (const std::exception& e) {
                    LOG_ERROR("Could not get output:%s", std::to_string(i));
                    return RETURN_FAIL;
                }
            }
            return RETURN_SUCCESS;
        }

        template int fetchOutputTensors<float>(std::vector<std::vector<float>> &outputs,
                                               int num_outputs,
                                               tvm::runtime::Module &gmod);
        template int fetchOutputTensors<int64_t>(std::vector<std::vector<int64_t>> &outputs,
                                                 int num_outputs,
                                                 tvm::runtime::Module &gmod);

        /**
         *  \brief  get the tensor type of input and output
         *  \param  index index of tensor to find the type
         *  \param  isInput bool to determine input/output tensor to fetch the details
         *          for
         *  \param  gmod TVM runtime module handle
         * @returns const char* containing the type name
         */
        const char *getTensorType(int index, bool isInput, tvm::runtime::Module &gmod)
        {
            static std::string type_str;
            if (isInput)
            {
                auto get_input = gmod.GetFunction("get_input");
                tvm::runtime::NDArray input_tensor = get_input(index);

                if (input_tensor->dtype.code == kDLFloat && input_tensor->dtype.bits == 32)
                {
                    type_str = "float32";
                }
                else if (input_tensor->dtype.code == kDLInt && input_tensor->dtype.bits == 64)
                {
                    type_str = "int64";
                }
                else if (input_tensor->dtype.code == kDLUInt && input_tensor->dtype.bits == 8)
                {
                    type_str = "uint8";
                }
                else
                {
                    type_str = "unknown";
                }
            } else {
                /* determine the output type */
                auto get_output = gmod.GetFunction("get_output");
                tvm::runtime::NDArray output_tensor = get_output(index);

                if (output_tensor->dtype.code == kDLFloat && output_tensor->dtype.bits == 32)
                {
                    type_str = "float32";
                }
                else if (output_tensor->dtype.code == kDLInt && output_tensor->dtype.bits == 64)
                {
                    type_str = "int64";
                }
                else if (output_tensor->dtype.code == kDLUInt && output_tensor->dtype.bits == 8)
                {
                    type_str = "uint8";
                }
                else
                {
                    type_str = "unknown";
                }
            }

            return type_str.c_str();
        }

        /**
         *  \brief  prepare the classification result inplace
         *  \param  img cv image to do inplace transform
         *  \param  s settings struct pointer
         *  \param  gmod TVM runtime module handle
         *  \param  num_outputs
	 *  \param  output_binary file path into which the results are written
         * @returns int status
         */
        int prepClassificationResult(cv::Mat *img, Settings *s, tvm::runtime::Module gmod,
                                     tvm::runtime::PackedFunc get_output_fn,
                                     int num_outputs, string output_binary)
        {
            LOG_INFO("preparing classification result \n");
            cv::resize((*img), (*img), cv::Size(512, 512), 0, 0, cv::INTER_AREA);
            const float threshold = 0.001f;
            std::vector<std::pair<float, int>> top_results;

            // get output tensor
            DLTensor *output_tensor;

            tvm::runtime::NDArray output_array = get_output_fn(0);
            const DLTensor *dl_tensor = output_array.operator->();
            // check data type
            if (dl_tensor->dtype.code == kDLFloat && dl_tensor->dtype.bits == 32) {
                // float32
                size_t output_size = 1;
                for (size_t i = 0; i < dl_tensor->ndim; i++) {
                    output_size *= dl_tensor->shape[i];
                }

                // get top N classes
                getTopN<float>(static_cast<float*>(dl_tensor->data), 1000, s->number_of_results,
                               threshold, &top_results, true);

                // write tensor data to binary file
                ofstream fout(output_binary, ios::binary);
                fout.write(reinterpret_cast<char*>(dl_tensor->data), output_size * sizeof(float));
                fout.close();

                std::vector<std::string> labels;
                size_t label_count;

                if (readLabelsFile(s->labels_file_path, &labels, &label_count) != 0) {
                    LOG_ERROR("failed to load labels file\n");
                    return RETURN_FAIL;
                }

                int output_offset;
                if (output_size == 1001) {
                    output_offset = 0;
                } else {
                    output_offset = 1;
                }

                for (const auto &result : top_results) {
                    const float confidence = result.first;
                    const int index = result.second;

                    LOG_INFO("%f: %d %s\n", confidence, index, labels[index + output_offset].c_str());
                }

                int num_results = s->number_of_results;
                (*img).data = overlayTopNClasses((*img).data, top_results, &labels,
                                                 (*img).cols, (*img).rows, num_results, output_offset);
            } else {
                LOG_ERROR("output type not supported (only float32 supported)");
                return RETURN_FAIL;
            }

            return RETURN_SUCCESS;
        }


        /**
         *  \brief  prepare the segmentation result inplace
         *  \param  img cv image to do inplace transform
         *  \param  gmod TVM runtime module Handle
         *  \param  num_outputs
         *  \param  mdoelInfo pointer to modelInfo
         *  \param  wanted_width
         *  \param  wanted_height
	 *  \param  output_binary file path into which the results are written
         * @returns int status
         */
        int prepSegResult(cv::Mat *img, tvm::runtime::Module &gmod,
                          tvm::runtime::PackedFunc get_output_fn,
                          int num_outputs, ModelInfo *modelInfo,
                          int wanted_width, int wanted_height, string output_binary)
        {
            LOG_INFO("preparing segmentation result \n");
            float alpha = modelInfo->m_postProcCfg.alpha;

            // determining the shape of output0
            // assuming 1 output of shape [1 , 1 , width , height]
            int64_t output_size = 0;
            int output_dim = 0;
            output_size = getOutputSize(get_output_fn, 0);
            output_dim = getOutputDimension(get_output_fn, 0);
            tvm::runtime::NDArray output_tensor = get_output_fn(0);

            // if indata and out data is diff resize the image
            // check whether img need to be resized based on out data
            // asssuming out put format [1,1,,width,height]
            if (wanted_height != output_tensor->shape[2] || wanted_width != output_tensor->shape[3])
            {
                LOG_INFO("Resizing image to match output dimensions\n");
                wanted_height = output_tensor->shape[2];
                wanted_width = output_tensor->shape[3];
                cv::resize((*img), (*img), cv::Size(wanted_width, wanted_height), 0, 0, cv::INTER_AREA);
            }
            /* determine the output type */
            const char *output_type = getTensorType(0, false, gmod);
            if (!strcmp(output_type, "int64"))
            {
                std::vector<std::vector<int64_t>> outputs;
                fetchOutputTensors<int64_t>(outputs, num_outputs, gmod);
                (*img).data = blendSegMask<int64_t>((*img).data, outputs[0].data(), (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
                ofstream fout(output_binary, ios::binary);
                fout.write(reinterpret_cast<char*>(outputs[0].data()), outputs[0].size() * sizeof(int64_t));
                fout.close();
            }
            else if (!strcmp(output_type, "float32"))
            {
                std::vector<std::vector<float>> outputs;
                fetchOutputTensors<float>(outputs, num_outputs, gmod);
                (*img).data = blendSegMask<float>((*img).data, outputs[0].data(), (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
                ofstream fout(output_binary, ios::binary);
                fout.write(reinterpret_cast<char*>(outputs[0].data()), outputs[0].size() * sizeof(float));
                fout.close();
            }
            else if (!strcmp(output_type, "uint8"))
            {
                std::vector<std::vector<uint8_t>> outputs;
                fetchOutputTensors<uint8_t>(outputs, num_outputs, gmod);
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
	 *  \brief query the graph json file for the input tensor name
	 *  \param model deploy graph json as a string
	 * @returns std::vector of input names
	 */
        std::vector<std::string> getInputNames(const std::string& json_data) {
            std::vector<std::string> input_names;

	    // find "nodes" array in the input json string
	    size_t nodes_pos = json_data.find("\"nodes\":");
	    if (nodes_pos == std::string::npos) {
		LOG_ERROR("Could not find \"nodes\" in graph JSON\n");
		return input_names;
	    }

	    // inside the "nodes" array, look for `"op" : "null"` and
	    // "name" containing "input".
	    size_t pos = nodes_pos;
	    while (true) {
		pos = json_data.find("\"op\": \"null\"", pos);
		if (pos == std::string::npos) {
		    // try without space
		    pos = nodes_pos;
		    pos = json_data.find("\"op\":\"null\"", pos);
		    if (pos == std::string::npos) {
			break;
		    }
		}

		// Search backward for the "name" field that belongs to this node
		// Look for the opening brace of the current node
		size_t node_start = json_data.rfind("{", pos);
		if (node_start == std::string::npos) break;

		// Find "name" within the current node
		size_t name_pos = json_data.find("\"name\":", node_start, pos - node_start);
		if (name_pos == std::string::npos) {
		    // Try alternative format with space
		    name_pos = json_data.find("\"name\": ", node_start, pos - node_start);
		    if (name_pos == std::string::npos) {
			pos += 10; // Move past current "op":"null"
			continue;
		    }
		}

		// Extract the name value - find opening and closing quotes
		size_t name_start = json_data.find("\"", name_pos + 7) + 1; // +7 for "name":
		size_t name_end = json_data.find("\"", name_start);
		if (name_start == std::string::npos || name_end == std::string::npos) {
		    pos += 10; // Move past current "op":"null"
		    continue;
		}

		std::string node_name = json_data.substr(name_start, name_end - name_start);
		LOG_INFO("Found null op node with name: %s\n", node_name.c_str());

		// Check if this is an input node
		if (node_name.find("input") != std::string::npos ||
		    node_name.find("Input") != std::string::npos ||
		    node_name.find("data") == 0 ||
		    node_name.find("image") == 0) {
		    input_names.push_back(node_name);
		    LOG_INFO("Added as input node: %s\n", node_name.c_str());
		}

		// Move past this node for next iteration
		pos += 10; // Move past "op":"null"
	    }

	    if (input_names.empty()) {
		LOG_ERROR("No input nodes found in the graph JSON\n");
	    } else {
		LOG_INFO("Found %lu input nodes in the graph JSON\n", input_names.size());
	    }

	    return input_names;
        }

        /**
         *  \brief  Actual inference happening
         *  \param  ModelInfo YAML parsed model info
         *  \param  Settings user input options  and default values of setting if any
         * @returns int
         */
        int runInference(ModelInfo *modelInfo, Settings *s)
        {
            int num_outputs, num_inputs;
            /*Initial inference time*/
            double fp_ms_avg = 0.0;
            DLDevice dev;
            if (s->device_type == "cpu") {
                dev = { kDLCPU, 0};
            } else if (s->device_type == "gpu") {
                dev = {kDLCUDA, 0};
            } else {
                LOG_ERROR("device type not supported: %s", s->device_type.c_str());
                return RETURN_FAIL;
            }
            // load model into TVM
            std::string artifact_path = modelInfo->m_infConfig.artifactsPath;
            std::string model_so_path = artifact_path + "/deploy_lib.so";
            std::string model_json_path = artifact_path + "/deploy_graph.json";
            std::string model_params_path = artifact_path + "/deploy_param.params";

            // create tvm runtime module
            auto mod_factory = tvm::runtime::Module::LoadFromFile(model_so_path);

            // load json graph
            std::ifstream json_in(model_json_path);
            std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
            json_in.close();

	    // Extract input names from JSON before creating the graph executor
	    std::vector<std::string> input_names = getInputNames(json_data);
	    if (input_names.empty()) {
		LOG_ERROR("No input nodes found in the graph JSON\n");
		return RETURN_FAIL;
	    }

            // load parameters
            std::ifstream params_in(model_params_path, std::ios::binary);
            std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
            params_in.close();

            // create graph runtime module
            const tvm::runtime::PackedFunc *fn = tvm::runtime::Registry::Get("tvm.graph_executor.create");
            tvm::runtime::Module gmod = (*fn)(json_data, mod_factory, static_cast<int>(dev.device_type), dev.device_id);

            // setup  a few functions
            tvm::runtime::PackedFunc run_fn = gmod.GetFunction("run");
            tvm::runtime::PackedFunc load_params_fn = gmod.GetFunction("load_params");
            tvm::runtime::PackedFunc set_input_fn = gmod.GetFunction("set_input");
            tvm::runtime::PackedFunc get_output_fn = gmod.GetFunction("get_output");
            tvm::runtime::PackedFunc get_num_output_fn = gmod.GetFunction("get_num_outputs");
            tvm::runtime::PackedFunc get_input_info_fn = gmod.GetFunction("get_input_info");

            // load params
            load_params_fn(tvm::runtime::String(params_data));

            // initialize inputs
            int wanted_height = modelInfo->m_preProcCfg.outDataHeight;
            int wanted_width = modelInfo->m_preProcCfg.outDataWidth;
            int wanted_channels = modelInfo->m_preProcCfg.numChans;

            LOG_INFO("Inference call started...\n");
            cv::Mat img;
            float *image_data;

            // memory allocation
            if(s->accel && s->device_mem) {
                #ifdef DEVICE_AM62
                LOG_ERROR("TIDL Delgate mode is not allowed on AM62 devices...\n");
                return RETURN_FAIL;
                #else
                image_data = (float *)TIDLRT_allocSharedMem(128, sizeof(float) * wanted_height * wanted_width * wanted_channels);
                if (image_data == NULL)
                {
                    image_data = (float*)calloc(wanted_height * wanted_width * wanted_channels, sizeof(float));
                }
                #endif
            } else {
                image_data = (float*)calloc(wanted_height * wanted_width * wanted_channels,sizeof(float) );
            }

            if (image_data == NULL) {
                LOG_ERROR("could not allocate space for image data \n");
                return RETURN_FAIL;
            }

            LOG_INFO("Input tensor Pointer - %p \n", image_data);

            // preprocess image
            const char *input_type = getTensorType(0, true, gmod);

            if (!strcmp(input_type, "float32"))
            {
                img = preprocImage<float>(s->input_image_path, image_data, modelInfo->m_preProcCfg);
            }
            else if (!strcmp(input_type, "uint8"))
            {
                img = preprocImage<uint8_t>(s->input_image_path, (uint8_t*)image_data, modelInfo->m_preProcCfg);
            }
            else
            {
                LOG_ERROR("cannot handle input type %s yet", input_type);
                return RETURN_FAIL;
            }

            LOG_INFO("Classifying input:%s\n", s->input_image_path.c_str());

            /*Running inference */
            DLTensor input_tensor;
            input_tensor.device = dev;
            input_tensor.ndim = 4;

            // setup shape and initialize it in input_tensor
            int64_t shape[4]; // NCHW or NHWC

            if (modelInfo->m_preProcCfg.dataLayout == "NCHW") {
                shape[0] = 1; // batch size
                shape[1] = wanted_channels;
                shape[2] = wanted_height;
                shape[3] = wanted_width;
            } else {
                // NHWC
                shape[0] = 1;
                shape[1] = wanted_height;
                shape[2] = wanted_width;
                shape[3] = wanted_channels;
            }
            input_tensor.shape = shape;
            input_tensor.strides = nullptr;
            input_tensor.byte_offset = 0;
            input_tensor.data = image_data;
            input_tensor.dtype = {kDLUInt, static_cast<uint8_t>(8), 1};

	    LOG_INFO("Using input name: %s\n", input_names[0].c_str());
            set_input_fn(input_names[0], &input_tensor);

            // warmup runs
            int num_iter = s->loop_count;
            if (s->loop_count >= 1) {
                LOG_INFO("Session.Run() - Started for warmup runs\n");
                for (size_t i = 0; i < s->number_of_warmup_runs; i++) {
                    run_fn();
                }
            }

            // time the inference
            struct timeval start_time, stop_time;
            gettimeofday(&start_time, nullptr);

            for (int i = 0; i < num_iter; i++) {
                run_fn();
            }

            gettimeofday(&stop_time, nullptr);
            float avg_time = (getUs(stop_time) - getUs(start_time)) / (num_iter * 1000);
            LOG_INFO("average time: %lf ms \n", avg_time);

            // get number of outputs
            num_outputs = get_num_output_fn();

            // Create a directory for output
            string bin_filename, bin_foldername;
            bin_foldername = bin_foldername +  "output_binaries/";
            struct stat binary_folder_buffer;
            if (stat(bin_foldername.c_str(), &binary_folder_buffer) != 0) {
                if (mkdir(bin_foldername.c_str(), 0777) == -1) {
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
                if (RETURN_FAIL == prepClassificationResult(&img, s, gmod, get_output_fn, num_outputs, bin_foldername))
                    return RETURN_FAIL;
            }
            else if (modelInfo->m_preProcCfg.taskType == "detection")
            {
                /*store tensor_shape info of op tensors in arr
		  to avoid recalculation*/
                vector<vector<int64_t>> tensor_shapes_vec;
                vector<int64_t> tensor_size_vec;
                vector<vector<float>> f_tensor_unformatted;

                for (size_t i = 0; i < num_outputs; i++)
                {
                    // get the output size
                    int64_t output_size = getOutputSize(get_output_fn, i);

                    // get output dimension
                    int output_dim = getOutputDimension(get_output_fn, i);

                    // Get the actual tensor to extract shape information
                    tvm::runtime::NDArray output_tensor = get_output_fn(i);
		    const DLTensor *tensor = output_tensor.operator->();

                    // shape would be something like this:
                    // [1, 224, 224, 3]
                    // [batch size, height, width, channels]
                    // for an rgb image, channels is 3.
		    vector<int64_t> tensor_shape;
		    tensor_size_vec.push_back(output_size);
		    for (int k = 0; k < output_dim; k++) {
			tensor_shape.push_back(tensor->shape[k]);
		    }
		    tensor_shapes_vec.push_back(tensor_shape);

		    LOG_INFO("Output tensor %zu: shape = [", i);
		    for (size_t dim = 0; dim < tensor_shape.size(); dim++) {
			LOG_INFO("%ld%s", tensor_shape[dim], (dim < tensor_shape.size()-1) ? ", " : "");
		    }
		    LOG_INFO("], total_elements = %ld\n", output_size);
                }

		// Extract detection data in the EXACT same format as ONNX/TFLite
		// Structure: first all boxes from tensor0, then all boxes from tensor1

		// Calculate nboxes dynamically from first tensor's shape (similar to ONNX approach)
		vector<int64_t> tensor0_shape = tensor_shapes_vec[0];
		int nboxes = tensor0_shape[tensor0_shape.size() - 2];
		const int box_data_size = 5;     // x1, y1, x2, y2, score

		// Get raw tensor data
		std::vector<float> boxes_and_scores(tensor_size_vec[0], 0);
		tvm::runtime::NDArray boxes_tensor = get_output_fn(0);
		boxes_tensor.CopyToBytes(boxes_and_scores.data(), tensor_size_vec[0] * sizeof(float));

		std::vector<int64_t> class_ids(tensor_size_vec[1], 0);
		tvm::runtime::NDArray class_tensor = get_output_fn(1);
		class_tensor.CopyToBytes(class_ids.data(), tensor_size_vec[1] * sizeof(int64_t));

		// Structure data exactly like in tfl_main: loop through tensors first, then boxes
		// First process tensor 0 (boxes and scores)
		for (int j = 0; j < nboxes; j++)
		{
		    vector<float> temp;
		    int box_offset = j * box_data_size;
		    for (int k = 0; k < box_data_size; k++) {
		        temp.push_back(boxes_and_scores[box_offset + k]);
		    }
		    f_tensor_unformatted.push_back(temp);
		}

		// Then process tensor 1 (class IDs)
		for (int j = 0; j < nboxes; j++)
		{
		    vector<float> temp;
		    temp.push_back(class_ids[j]);
		    f_tensor_unformatted.push_back(temp);
		}

		LOG_INFO("detected objects:%d \n", nboxes);
		LOG_INFO("Total detection entries: %zu (should be %d)\n", f_tensor_unformatted.size(), nboxes * 2);
		LOG_INFO("First tensor0 box: x1=%.2f, y1=%.2f, x2=%.2f, y2=%.2f, score=%.3f\n",
			 f_tensor_unformatted[0][0], f_tensor_unformatted[0][1],
			 f_tensor_unformatted[0][2], f_tensor_unformatted[0][3],
			 f_tensor_unformatted[0][4]);
		LOG_INFO("First tensor1 class: %.0f\n", f_tensor_unformatted[nboxes][0]);

		// Set the formatter for the standard detection format
		modelInfo->m_postProcCfg.formatter = {0, 1, 2, 3, 5, 4}; // x1, y1, x2, y2, class, score
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
                if (RETURN_FAIL == prepSegResult(&img, gmod, get_output_fn, num_outputs, modelInfo, wanted_width, wanted_height, bin_foldername))
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

            if (image_data != NULL)
            {
                if(s->accel && s->device_mem){
                    #ifndef DEVICE_AM62
                    if (TIDLRT_isSharedMem(image_data))
                    {
                        TIDLRT_freeSharedMem(image_data);
                    }
                    else
                    {
                        free(image_data);
                    }
                    #endif
                }else{
                    free(image_data);
                }
            }

            LOG_INFO("\nCompleted_Model : 0, Name : %s, Total time : %f, Offload Time : 0 , DDR RW MBs : 0, Output Image File : %s, Output Bin File : %s\n \n",
                     modelInfo->m_postProcCfg.modelName.c_str(), avg_time, filename.c_str(), bin_filename.c_str());
            return RETURN_SUCCESS;
        }
    } // namespace main
} // namespace tvm

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
    if (tvm::main::runInference(&model, &s) == RETURN_FAIL)
    {
        LOG_ERROR("Failed to run runInference\n");
        return RETURN_FAIL;
    }
    return RETURN_SUCCESS;
}
