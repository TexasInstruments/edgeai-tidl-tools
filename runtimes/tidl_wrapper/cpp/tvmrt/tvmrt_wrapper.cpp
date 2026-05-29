/*
Copyright (c) 2026 Texas Instruments Incorporated

All rights reserved not granted herein.

Limited License.

Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
license under copyrights and patents it now or hereafter owns or controls to
make, have made, use, import, offer to sell and sell ("Utilize") this software
subject to the terms herein.  With respect to the foregoing patent license,
such license is granted  solely to the extent that any such patent is necessary
to Utilize the software alone.  The patent license shall not apply to any
combinations which include this software, other than combinations with devices
manufactured by or for TI ("TI Devices").  No hardware patent is licensed
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

THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "tvmrt_wrapper.h"
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <sys/utsname.h>

namespace fs = std::filesystem;

namespace tvmrt_wrapper
{
    /**
     * @brief Constructor for the TVMRT class
     *
     * Initializes the TVMRT object with the specified model path and TIDL offload setting.
     * Sets up the TVM device context (CPU by default).
     *
     * @param modelPath Path to the model file
     * @param tidlOffload Flag to enable TIDL hardware acceleration (unused for TVM)
     */
    TVMRT::TVMRT(std::string modelPath):
        m_modelPath(modelPath),
        m_numInputs(0),
        m_numOutputs(0)
    {
        // Set device to CPU
        m_device = {kDLCPU, 0};
    }

    /**
     * @brief Destructor for the TVMRT class
     *
     * Cleans up resources
     */
    TVMRT::~TVMRT()
    {
    }

    /**
     * @brief Creates and initializes the inference engine
     *
     * Sets up the TVM Runtime session with the specified options and prepares
     * the model for inference. This includes:
     * 1. Loading the TVM compiled artifacts (deploy_lib.so, deploy_graph.json, deploy_param.params)
     * 2. Creating the graph executor module
     * 3. Loading parameters
     * 4. Populating input and output tensor information
     *
     * @param options Map of configuration options for the inference engine
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t TVMRT::createInfer(std::map<std::string, std::string> &options)
    {
        int32_t status = 0;

        // Get artifacts folder
        if (options.find("artifacts_folder") != options.end())
        {
            m_artifactsPath = options["artifacts_folder"];
        }
        else
        {
            printf("[ERROR] 'artifacts_folder' is not provided for TVM runtime.\n");
            return -1;
        }

        // Determine suffix based on architecture, mirroring the Python wrapper
        const char *suffix = nullptr;
        struct utsname name;
        if (uname(&name) == 0)
        {
            if (strcmp(name.machine, "aarch64") == 0 || strcmp(name.machine, "arm64") == 0)
                suffix = ".evm";
            else if (strcmp(name.machine, "x86_64") == 0)
                suffix = ".pc";
        }
        if (suffix == nullptr)
        {
            printf("[ERROR] uname() failed or unsupported architecture\n");
            return -1;
        }

        // Construct paths to TVM artifacts using the suffixed filenames directly (no copy needed)
        std::string modelSoPath     = m_artifactsPath + "/deploy_lib.so"        + suffix;
        std::string modelJsonPath   = m_artifactsPath + "/deploy_graph.json"    + suffix;
        std::string modelParamsPath = m_artifactsPath + "/deploy_param.params"  + suffix;

        if (!fs::exists(modelSoPath) || !fs::exists(modelJsonPath) || !fs::exists(modelParamsPath))
        {
            printf("[ERROR] TVM model files not found in %s\n", m_artifactsPath.c_str());
            printf("[ERROR] Expected: deploy_lib.so%s, deploy_graph.json%s, deploy_param.params%s\n",
                   suffix, suffix, suffix);
            return -1;
        }

        try
        {
            // Load the compiled module
            tvm::runtime::Module modFactory = tvm::runtime::Module::LoadFromFile(modelSoPath, "so");

            // Load JSON graph
            std::ifstream jsonIn(modelJsonPath);
            std::string jsonData((std::istreambuf_iterator<char>(jsonIn)), std::istreambuf_iterator<char>());
            jsonIn.close();

            // Load parameters
            std::ifstream paramsIn(modelParamsPath, std::ios::binary);
            std::string paramsData((std::istreambuf_iterator<char>(paramsIn)), std::istreambuf_iterator<char>());
            paramsIn.close();

            // Create graph runtime module
            const tvm::runtime::PackedFunc *createGraphFn = tvm::runtime::Registry::Get("tvm.graph_executor.create");
            if (createGraphFn == nullptr)
            {
                printf("[ERROR] Could not find tvm.graph_executor.create function\n");
                return -1;
            }

            m_graphModule = (*createGraphFn)(jsonData, modFactory, static_cast<int>(m_device.device_type), m_device.device_id);

            // Load parameters
            tvm::runtime::PackedFunc loadParamsFn = m_graphModule.GetFunction("load_params");
            loadParamsFn(tvm::runtime::String(paramsData));

            // Extract input names from graph module
            m_inputNames = getInputNames();
            if (m_inputNames.empty())
            {
                printf("[ERROR] No input nodes found in the graph module\n");
                return -1;
            }

            // Populate input and output information
            status = populateInputInfo();
            if (status != 0)
            {
                printf("[ERROR] Failed to populate input information\n");
                return status;
            }

            status = populateOutputInfo();
            if (status != 0)
            {
                printf("[ERROR] Failed to populate output information\n");
                return status;
            }
        }
        catch (const std::exception& e)
        {
            printf("[ERROR] Exception during TVM model loading: %s\n", e.what());
            return -1;
        }

        return status;
    }

    /**
     * @brief Retrieves input names from the TVM graph module
     *
     * Uses TVM's get_input_info() API to extract input tensor names from the graph executor.
     * This approach is more robust than parsing JSON and doesn't require hardcoded name patterns.
     *
     * @return std::vector<std::string> Vector of input names
     */
    std::vector<std::string> TVMRT::getInputNames()
    {
        std::vector<std::string> inputNames;

        try
        {
            // Get the get_input_info function
            auto getInputInfoFn = m_graphModule.GetFunction("get_input_info");
            tvm::runtime::Map<tvm::runtime::String, tvm::runtime::ObjectRef> inputInfo = getInputInfoFn();

            // Extract shape information map (which is keyed by input names)
            auto shapeInfo = tvm::runtime::GetRef<tvm::runtime::Map<tvm::runtime::String, tvm::runtime::ObjectRef>>(
                inputInfo["shape"].as<tvm::runtime::MapNode>());

            // Iterate through the map to get input names
            for (const auto& kv : shapeInfo)
            {
                inputNames.push_back(std::string(kv.first));
            }
        }
        catch (const std::exception& e)
        {
            printf("[ERROR] Exception while getting input names: %s\n", e.what());
        }

        return inputNames;
    }

    /**
     * @brief Populates information about the model's input tensors
     *
     * Queries the TVM Runtime module for details about input tensors and populates
     * the corresponding member variables.
     *
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t TVMRT::populateInputInfo()
    {
        int32_t status = 0;

        m_numInputs = static_cast<int32_t>(m_inputNames.size());
        m_inputs.assign(m_numInputs, DlTensor());

        try
        {
            auto getInputFn = m_graphModule.GetFunction("get_input");

            for (int32_t i = 0; i < m_numInputs; i++)
            {
                DlTensor *info = &m_inputs[i];

                // Get input tensor
                tvm::runtime::NDArray inputTensor = getInputFn(m_inputNames[i]);
                const DLTensor *dlTensor = inputTensor.operator->();

                // Store name (use index as name since TVM doesn't always provide meaningful names)
                info->name = m_inputNames[i].c_str();

                // Get shape information
                info->numDim = dlTensor->ndim;
                info->shape.clear();
                info->numElem = 1;
                for (int32_t j = 0; j < info->numDim; j++)
                {
                    info->shape.push_back(dlTensor->shape[j]);
                    info->numElem *= dlTensor->shape[j];
                }

                // Get type information
                info->elemSize = DLType2TidlType(dlTensor->dtype, info->type, info->typeName);
                info->allocSize = info->numElem * info->elemSize;

                if (info->allocSize <= 0)
                {
                    printf("[ERROR] Invalid size(%ld) for input(%d).\n", info->allocSize, i);
                    status = -1;
                    break;
                }

                info->padT = 0;
                info->padB = 0;
                info->padL = 0;
                info->padR = 0;
                info->validSize = info->allocSize;
                info->data = nullptr;
            }
        }
        catch (const std::exception& e)
        {
            printf("[ERROR] Exception while populating input info: %s\n", e.what());
            status = -1;
        }

        return status;
    }

    /**
     * @brief Gets details about the model's input tensors
     *
     * @return const std::vector<DlTensor>* Pointer to vector of input tensor details
     */
    const std::vector<DlTensor>* TVMRT::getInputDetails()
    {
        return &m_inputs;
    }

    /**
     * @brief Populates information about the model's output tensors
     *
     * Queries the TVM Runtime module for details about output tensors and populates
     * the corresponding member variables.
     *
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t TVMRT::populateOutputInfo()
    {
        int32_t status = 0;

        try
        {
            auto getNumOutputsFn = m_graphModule.GetFunction("get_num_outputs");
            m_numOutputs = getNumOutputsFn();

            m_outputs.assign(m_numOutputs, DlTensor());
            m_outputNames.resize(m_numOutputs);

            auto getOutputFn = m_graphModule.GetFunction("get_output");

            for (int32_t i = 0; i < m_numOutputs; i++)
            {
                DlTensor *info = &m_outputs[i];

                // Get output tensor
                tvm::runtime::NDArray outputTensor = getOutputFn(i);
                const DLTensor *dlTensor = outputTensor.operator->();

                // Store name (use index as name)
                m_outputNames[i] = "output_" + std::to_string(i);
                info->name = m_outputNames[i].c_str();

                // Get shape information
                info->numDim = dlTensor->ndim;
                info->shape.clear();
                info->numElem = 1;
                for (int32_t j = 0; j < info->numDim; j++)
                {
                    info->shape.push_back(dlTensor->shape[j]);
                    info->numElem *= dlTensor->shape[j];
                }

                // Get type information
                info->elemSize = DLType2TidlType(dlTensor->dtype, info->type, info->typeName);
                info->allocSize = info->numElem * info->elemSize;

                if (info->allocSize <= 0)
                {
                    printf("[ERROR] Invalid size(%ld) for output(%d).\n", info->allocSize, i);
                    status = -1;
                    break;
                }

                info->padT = 0;
                info->padB = 0;
                info->padL = 0;
                info->padR = 0;
                info->validSize = info->allocSize;
                info->data = nullptr;
            }
        }
        catch (const std::exception& e)
        {
            printf("[ERROR] Exception while populating output info: %s\n", e.what());
            status = -1;
        }

        return status;
    }

    /**
     * @brief Gets details about the model's output tensors
     *
     * @return const std::vector<DlTensor>* Pointer to vector of output tensor details
     */
    const std::vector<DlTensor>* TVMRT::getOutputDetails()
    {
        return &m_outputs;
    }

    /**
     * @brief Prints detailed information about the model and its tensors
     *
     * Outputs model path, input/output tensor counts, and detailed information
     * about each tensor including name, type, shape, and size.
     */
    void TVMRT::dumpInfo()
    {
        printf("Model Path        = %s\n", m_modelPath.c_str());
        printf("Artifacts Path    = %s\n", m_artifactsPath.c_str());
        printf("Number of Inputs  = %d\n", m_numInputs);
        for (int32_t i = 0; i < m_numInputs; i++)
        {
            printf("INPUT [%d]: \n", i);
            m_inputs[i].dumpInfo();
        }
        printf("Number of Outputs  = %d\n", m_numOutputs);
        for (int32_t i = 0; i < m_numOutputs; i++)
        {
            printf("OUTPUT [%d]: \n", i);
            m_outputs[i].dumpInfo();
        }
    }

    /**
     * @brief Runs inference on the loaded model
     *
     * Executes the model with the provided input tensors and populates the output tensors
     * with the inference results.
     *
     * @param inputs Vector of input tensors containing the data for inference
     * @param outputs Vector of output tensors to store the inference results
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t TVMRT::runInfer(const std::vector<DlTensor *> &inputs, std::vector<DlTensor *> &outputs)
    {
        int32_t status = 0;

        try
        {
            auto setInputFn = m_graphModule.GetFunction("set_input");
            auto getOutputFn = m_graphModule.GetFunction("get_output");
            auto runFn = m_graphModule.GetFunction("run");

            // Set input tensors
            for (int32_t i = 0; i < m_numInputs; i++)
            {
                const DlTensor *inputInfo = inputs[i];

                // Create DLTensor wrapper for the input data
                DLTensor dlTensor;
                dlTensor.data = inputInfo->data;
                dlTensor.device = m_device;
                dlTensor.ndim = inputInfo->numDim;
                dlTensor.dtype = m_inputs[i].type == TIDL_UnsignedChar ? DLDataType{kDLUInt, 8, 1} :
                                 m_inputs[i].type == TIDL_SignedChar ? DLDataType{kDLInt, 8, 1} :
                                 m_inputs[i].type == TIDL_SinglePrecFloat ? DLDataType{kDLFloat, 32, 1} :
                                 DLDataType{kDLFloat, 32, 1};
                dlTensor.shape = const_cast<int64_t*>(inputInfo->shape.data());
                dlTensor.strides = nullptr;
                dlTensor.byte_offset = 0;

                setInputFn(m_inputNames[i], &dlTensor);
            }

            // Run inference
            runFn();

            // Get output tensors
            for (int32_t i = 0; i < m_numOutputs; i++)
            {
                DlTensor *outputInfo = outputs[i];
                tvm::runtime::NDArray outputTensor = getOutputFn(i);

                // Copy output data
                outputTensor.CopyToBytes(outputInfo->data, outputInfo->allocSize);
            }
        }
        catch (const std::exception& e)
        {
            printf("[ERROR] Exception during inference: %s\n", e.what());
            status = -1;
        }

        return status;
    }

    /**
     * @brief Gets the performance data
     *
     * @return  const std::map<std::string, std::pair<float, std::string>>
     *          Map containing performance data where key is performance
     *          metric, and values are (data, unit)
     *
     *          'total_time': Total time taken for run (ms)
     *          'core_time': Total time taken barring the io copy time (ms)
     *          'subgraph_time': Total TIDL Subgraphs processing time (ms)
     *          'read_total': Total DDR Read bytes [X for x86 runs]
     *          'write_total': Total DDR Write bytes [X for x86 runs]
     */
    const std::map<std::string, std::pair<float, std::string>> TVMRT::getPerformance()
    {
        std::map<std::string, std::pair<float, std::string>> data;

        try
        {
            // Get the benchmark data function from TVM graph module
            auto getBenchmarkDataFn = m_graphModule.GetFunction("get_benchmark_data");
            if (getBenchmarkDataFn == nullptr)
            {
                printf("[WARN] get_benchmark_data function not available in TVM module\n");
                return data;
            }

            // Call get_benchmark_data to get the Map<String, String>
            tvm::runtime::Map<tvm::runtime::String, tvm::runtime::String> benchmarkData = getBenchmarkDataFn();

            // Extract timing data
            uint64_t runStart = 0, runEnd = 0;
            uint64_t ddrReadStart = 0, ddrReadEnd = 0;
            uint64_t ddrWriteStart = 0, ddrWriteEnd = 0;

            if (benchmarkData.count("ts:run_start"))
                runStart = std::stoull(std::string(benchmarkData["ts:run_start"]));
            if (benchmarkData.count("ts:run_end"))
                runEnd = std::stoull(std::string(benchmarkData["ts:run_end"]));
            if (benchmarkData.count("ddr:read_start"))
                ddrReadStart = std::stoull(std::string(benchmarkData["ddr:read_start"]));
            if (benchmarkData.count("ddr:read_end"))
                ddrReadEnd = std::stoull(std::string(benchmarkData["ddr:read_end"]));
            if (benchmarkData.count("ddr:write_start"))
                ddrWriteStart = std::stoull(std::string(benchmarkData["ddr:write_start"]));
            if (benchmarkData.count("ddr:write_end"))
                ddrWriteEnd = std::stoull(std::string(benchmarkData["ddr:write_end"]));

            uint64_t totalTime = runEnd - runStart;
            uint64_t readTotal = ddrReadEnd - ddrReadStart;
            uint64_t writeTotal = ddrWriteEnd - ddrWriteStart;

            // Extract subgraph timing data
            uint64_t subgraphProcTime = 0;
            uint64_t subgraphCopyInTime = 0;
            uint64_t subgraphCopyOutTime = 0;
            int32_t numSubgraphs = 0;

            // Iterate through subgraphs
            for (int32_t i = 0; ; i++)
            {
                std::string procStartKey = "ts:subgraph_" + std::to_string(i) + "_proc_start";
                std::string procEndKey = "ts:subgraph_" + std::to_string(i) + "_proc_end";
                std::string copyInStartKey = "ts:subgraph_" + std::to_string(i) + "_copy_in_start";
                std::string copyInEndKey = "ts:subgraph_" + std::to_string(i) + "_copy_in_end";
                std::string copyOutStartKey = "ts:subgraph_" + std::to_string(i) + "_copy_out_start";
                std::string copyOutEndKey = "ts:subgraph_" + std::to_string(i) + "_copy_out_end";

                if (!benchmarkData.count(procStartKey))
                {
                    break;
                }

                numSubgraphs++;

                uint64_t procStart = std::stoull(std::string(benchmarkData[procStartKey]));
                uint64_t procEnd = std::stoull(std::string(benchmarkData[procEndKey]));
                uint64_t copyInStart = std::stoull(std::string(benchmarkData[copyInStartKey]));
                uint64_t copyInEnd = std::stoull(std::string(benchmarkData[copyInEndKey]));
                uint64_t copyOutStart = std::stoull(std::string(benchmarkData[copyOutStartKey]));
                uint64_t copyOutEnd = std::stoull(std::string(benchmarkData[copyOutEndKey]));

                subgraphProcTime += (procEnd - procStart);
                subgraphCopyInTime += (copyInEnd - copyInStart);
                subgraphCopyOutTime += (copyOutEnd - copyOutStart);
            }

            uint64_t totalCopyTime = subgraphCopyInTime + subgraphCopyOutTime;

            // Convert to milliseconds
            float totalTimeMs = (float)(totalTime) / 1000000.0f;
            float totalCopyTimeMs = 0.0f;
            if (numSubgraphs == 1)
            {
                totalCopyTimeMs = (float)(totalCopyTime) / 1000000.0f;
            }
            float totalCoreTimeMs = totalTimeMs - totalCopyTimeMs;
            float subgraphProcTimeMs = (float)(subgraphProcTime) / 1000000.0f;

            // Populate the data map
            data["total_time"] = {totalTimeMs, "ms"};
            data["core_time"] = {totalCoreTimeMs, "ms"};
            data["subgraph_time"] = {subgraphProcTimeMs, "ms"};
            data["read_total"] = {(float)(readTotal), "bytes"};
            data["write_total"] = {(float)(writeTotal), "bytes"};
        }
        catch (const std::exception& e)
        {
            printf("[WARN] Exception while getting performance data: %s\n", e.what());
        }

        return data;
    }

    /**
     * @brief Converts DLDataType to TIDL type
     *
     * Maps DLDataType to their corresponding TIDL type identifiers and
     * returns the size in bytes of the data type.
     *
     * @param dlType The DLDataType to convert
     * @param tidlType Reference to store the corresponding TIDL type
     * @param typeName Reference to store the corresponding type name
     * @return int32_t Size in bytes of the data type
     */
    int32_t TVMRT::DLType2TidlType(const DLDataType &dlType, int32_t &tidlType, std::string &typeName)
    {
        int32_t size = 0;

        if (dlType.code == kDLInt)
        {
            if (dlType.bits == 8)
            {
                tidlType = TIDL_SignedChar;
                size = sizeof(int8_t);
                typeName = "int8_t";
            }
            else if (dlType.bits == 16)
            {
                tidlType = TIDL_SignedShort;
                size = sizeof(int16_t);
                typeName = "int16_t";
            }
            else if (dlType.bits == 32)
            {
                tidlType = TIDL_SignedWord;
                size = sizeof(int32_t);
                typeName = "int32_t";
            }
            else if (dlType.bits == 64)
            {
                tidlType = TIDL_SignedDoubleWord;
                size = sizeof(int64_t);
                typeName = "int64_t";
            }
        }
        else if (dlType.code == kDLUInt)
        {
            if (dlType.bits == 8)
            {
                tidlType = TIDL_UnsignedChar;
                size = sizeof(uint8_t);
                typeName = "uint8_t";
            }
            else if (dlType.bits == 16)
            {
                tidlType = TIDL_UnsignedShort;
                size = sizeof(uint16_t);
                typeName = "uint16_t";
            }
            else if (dlType.bits == 32)
            {
                tidlType = TIDL_UnsignedWord;
                size = sizeof(uint32_t);
                typeName = "uint32_t";
            }
            else if (dlType.bits == 64)
            {
                tidlType = TIDL_UnsignedDoubleWord;
                size = sizeof(uint64_t);
                typeName = "uint64_t";
            }
        }
        else if (dlType.code == kDLFloat)
        {
            if (dlType.bits == 32)
            {
                tidlType = TIDL_SinglePrecFloat;
                size = sizeof(float);
                typeName = "float";
            }
        }

        if (size == 0)
        {
            tidlType = -1;
            typeName = "invalid";
        }

        return size;
    }

} // namespace tvmrt_wrapper
