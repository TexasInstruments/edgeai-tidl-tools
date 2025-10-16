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

#include "onnxrt_wrapper.h"
#include <iomanip>
#include <sstream>

namespace onnxrt_wrapper
{
    /**
     * @brief Constructor for the ONNXRT class
     * 
     * Initializes the ONNXRT object with the specified model path and TIDL offload setting.
     * Sets up the ONNX Runtime environment with warning-level logging and configures
     * CPU memory allocation for tensors.
     * 
     * @param modelPath Path to the ONNX model file
     * @param tidlOffload Flag to enable TIDL hardware acceleration
     */
    ONNXRT::ONNXRT(std::string modelPath, bool tidlOffload):
        m_modelPath(modelPath),
        m_tidlOffload(tidlOffload),
        m_env(ORT_LOGGING_LEVEL_WARNING, __FUNCTION__),
        m_memInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
    }

    /**
     * @brief Destructor for the ONNXRT class
     * 
     * Cleans up resources by deleting the ONNX Runtime session object
     */
    ONNXRT::~ONNXRT()
    {
        delete m_session;
    }

    /**
     * @brief Creates and initializes the inference engine
     * 
     * Sets up the ONNX Runtime session with the specified options and prepares
     * the model for inference. This includes:
     * 1. Configuring session options with optimization level and logging
     * 2. Setting up TIDL acceleration if enabled
     * 3. Creating the ONNX Runtime session with the model
     * 4. Populating input and output tensor information
     * 
     * @param options Map of configuration options for the inference engine
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t ONNXRT::createInfer(std::map<std::string, std::string> &options)
    {
        OrtStatus              *ortStatus;
        Ort::SessionOptions     sessionOpts;
        c_api_tidl_options      tidlOpts{};
        int32_t                 status;

        // Set graph optimization level and logging severity
        sessionOpts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        sessionOpts.SetLogSeverityLevel(3);

        if (m_tidlOffload)
        {
            OrtStatus *def_status = OrtSessionsOptionsSetDefault_Tidl(&tidlOpts);

            // Parse and set from options map
            if (options.find("artifacts_folder") != options.end())
            {
                strcpy(tidlOpts.artifacts_folder, options["artifacts_folder"].c_str());
            }
            else
            {
                status = -1;
                throw std::runtime_error("'artifacts_folder' is not provided.");
            }

            if (options.find("debug_level") != options.end())
            {
                try
                {
                    tidlOpts.debug_level = std::stoi(options["debug_level"]);
                }
                catch (const std::invalid_argument& e)
                {
                    throw std::runtime_error("Could not parse debug_level");
                }
            }
            if (options.find("priority") != options.end())
            {
                try
                {
                    tidlOpts.priority = std::stoi(options["priority"]);
                }
                catch (const std::invalid_argument& e)
                {
                    throw std::runtime_error("Could not parse priority");
                }
            }
            if (options.find("max_pre_empt_delay") != options.end())
            {
                try
                {
                    tidlOpts.max_pre_empt_delay = std::stof(options["max_pre_empt_delay"]);
                }
                catch (const std::invalid_argument& e)
                {
                    throw std::runtime_error("Could not parse max_pre_empt_delay");
                }
            }
            if (options.find("core_number") != options.end())
            {
                try
                {
                    tidlOpts.core_number = std::stoi(options["core_number"]);
                }
                catch (const std::invalid_argument& e)
                {
                    throw std::runtime_error("Could not parse core_number");
                }
            }


            ortStatus = OrtSessionOptionsAppendExecutionProvider_Tidl(sessionOpts, &tidlOpts);
        }
        else
        {
            ortStatus = OrtSessionOptionsAppendExecutionProvider_CPU(sessionOpts, false);
        }


        if (ortStatus == NULL)
        {
            m_session = new Ort::Session(m_env, m_modelPath.c_str(), sessionOpts);

            // Query the input information
            status = populateInputInfo();
        }
        else
        {
            status = -1;
            throw std::runtime_error("ONNXRT setting session options failed");
        }


        // Query the output information
        if (status == 0)
        {
            /* This needs a warmup run, hence dummy input needs to be provided */
            for (int32_t i = 0; i < m_numInputs; i++)
            {
                DlTensor *info = &m_inputs[i];
                info->data = (void *)malloc(info->allocSize);
            }

            status = populateOutputInfo();

            for (int32_t i = 0; i < m_numInputs; i++)
            {
                DlTensor *info = &m_inputs[i];

                if ((void *)info->data != nullptr)
                {
                    free((void *)info->data);
                }
            }
        }

        if (status < 0)
        {
            throw std::runtime_error("ORTInferer object creation failed.");
        }

        return status;
    }

    /**
     * @brief Populates information about the model's input tensors
     * 
     * Queries the ONNX Runtime session for details about input tensors and populates
     * the corresponding member variables. For each input tensor, it retrieves:
     * - Name
     * - Shape (dimensions)
     * - Data type
     * - Element count and size
     * 
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t ONNXRT::populateInputInfo()
    {
        int32_t status = 0;

        /* Query the number of inputs. */
        m_numInputs = m_session->GetInputCount();

        /* Initialize member variables */
        m_inputs.assign(m_numInputs, DlTensor());
        m_inputNames.assign(m_numInputs, nullptr);
        m_inputTypes.assign(m_numInputs, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

        for (int32_t i = 0; i < m_numInputs; i++)
        {
            DlTensor *info = &m_inputs[i];

            /* Query input properties. */
            auto inputName = m_session->GetInputNameAllocated(i, m_allocator);
            info->name = inputName.get();
            if (info->name == nullptr)
            {
                printf("[ERROR] GetInputNameAllocated failed for input(%d).\n", i);
                status = -1;
                break;
            }
            m_inputNames[i] = info->name;
            m_inputNamesPtr.push_back(std::move(inputName));

            auto typeInfo = m_session->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

            info->shape = tensorInfo.GetShape();
            info->numDim = tensorInfo.GetDimensionsCount();
            info->numElem = tensorInfo.GetElementCount();

            m_inputTypes[i] = tensorInfo.GetElementType();
            info->elemSize = Onnx2TidlType(m_inputTypes[i], info->type, info->typeName);

            info->allocSize = info->numElem * info->elemSize;

            if (info->allocSize <= 0)
            {
                printf("Invalid size(%ld) for input(%d).\n",info->allocSize, i);
                status = -1;
                break;
            }
            
            info->padT = 0;
            info->padB = 0;
            info->padL = 0;
            info->padR = 0;
            info->validSize = info->allocSize;
        }

        return status;
    }

    /**
     * @brief Gets details about the model's input tensors
     * 
     * @return const std::vector<DlTensor>* Pointer to vector of input tensor details
     */
    const std::vector<DlTensor>* ONNXRT::getInputDetails()
    {
        return &m_inputs;
    }

    /**
     * @brief Populates information about the model's output tensors
     * 
     * Queries the ONNX Runtime session for details about output tensors and populates
     * the corresponding member variables. This method:
     * 1. Gets output names from the session
     * 2. Performs a warmup inference run to determine output tensor properties
     * 3. Extracts shape, dimensions, data type, and size information for each output
     * 
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t ONNXRT::populateOutputInfo()
    {
        int32_t                 status = 0;

        /* Query the number of outputs. */
        m_numOutputs = m_session->GetOutputCount();

        /* Initialize member variables */
        m_outputs.assign(m_numOutputs, DlTensor());
        m_outputNames.assign(m_numOutputs, nullptr);
        m_outputTypes.assign(m_numOutputs, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

        /* Get output names for warmup run */
        for (int32_t i = 0; i < m_numOutputs; i++)
        {
            DlTensor *info = &m_outputs[i];

            auto outputName = m_session->GetOutputNameAllocated(i, m_allocator);
            info->name = outputName.get();
            if (info->name == nullptr)
            {
                printf("[ERROR] GetOutputNameAllocated failed for input(%d).\n", i);
                status = -1;
                break;
            }
            m_outputNames[i] = info->name;
            m_outputNamesPtr.push_back(std::move(outputName));
        }

        /* Warmup Run to get output tensor properties */
        std::vector<Ort::Value> inputValues;
        std::vector<Ort::Value> outputValues;
        auto runOpts = Ort::RunOptions();
 
        for (uint32_t i = 0; i < m_numInputs; i++)
        {
            const DlTensor *info = &m_inputs[i];
            Ort::Value v = Ort::Value::CreateTensor(m_memInfo,
                                                    (void *)info->data,
                                                    (size_t)info->allocSize,
                                                    info->shape.data(),
                                                    info->shape.size(),
                                                    m_inputTypes[i]);
            inputValues.push_back(std::move(v));
        }

        for (uint32_t i = 0; i < m_numOutputs; i++)
        {
            outputValues.emplace_back(nullptr);
        }

        outputValues = m_session->Run(runOpts,
                                      m_inputNames.data(),
                                      inputValues.data(),
                                      m_numInputs,
                                      m_outputNames.data(),
                                      m_numOutputs);

        /* Get output properties after warmup run */
        for (int32_t i = 0; i < m_numOutputs; i++)
        {
            DlTensor *info = &m_outputs[i];

            auto &tensor = outputValues[i];
            const auto &tensorInfo = tensor.GetTensorTypeAndShapeInfo();
            info->shape = tensorInfo.GetShape();
            info->numDim = tensorInfo.GetDimensionsCount();
            info->numElem = tensorInfo.GetElementCount();

            m_outputTypes[i] = tensorInfo.GetElementType();
            info->elemSize = Onnx2TidlType(m_outputTypes[i], info->type, info->typeName);

            info->allocSize = info->numElem * info->elemSize;
            if (info->allocSize <= 0)
            {
                printf("Invalid size(%ld) for output(%d).\n",info->allocSize, i);
                status = -1;
                break;
            }

            info->padT = 0;
            info->padB = 0;
            info->padL = 0;
            info->padR = 0;
            info->validSize = info->allocSize;
        }

        return status;
    }

    /**
     * @brief Gets details about the model's output tensors
     * 
     * @return const std::vector<DlTensor>* Pointer to vector of output tensor details
     */
    const std::vector<DlTensor>* ONNXRT::getOutputDetails()
    {
        return &m_outputs;
    }

    /**
     * @brief Prints detailed information about the model and its tensors
     * 
     * Outputs model path, input/output tensor counts, and detailed information
     * about each tensor including name, type, shape, and size.
     */
    void ONNXRT::dumpInfo()
    {
        printf("Model Path        = %s\n", m_modelPath.c_str());
        printf("Number of Inputs  = %d\n", m_numInputs);
        for (uint32_t i = 0; i < m_numInputs; i++)
        {
            printf("INPUT [%d]: \n", i);
            m_inputs[i].dumpInfo();
        }
        printf("Number of Outputs  = %d\n", m_numOutputs);
        for (uint32_t i = 0; i < m_numOutputs; i++)
        {
            printf("OUTPUT [%d]: \n", i);
            m_outputs[i].dumpInfo();
        }
    }

    /**
     * @brief Runs inference on the loaded model
     * 
     * Executes the model with the provided input tensors and populates the output tensors
     * with the inference results. This method:
     * 1. Sets up run options with appropriate logging levels
     * 2. Creates ONNX Runtime tensor values from the input DlTensors
     * 3. Binds input and output tensors to the session
     * 4. Executes the model inference
     * 
     * @param inputs Vector of input tensors containing the data for inference
     * @param outputs Vector of output tensors to store the inference results
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t ONNXRT::runInfer(const std::vector<DlTensor *> &inputs, std::vector<DlTensor *> &outputs)
    {
        int32_t status = 0;

        Ort::IoBinding binding(*m_session);
        auto runOpts = Ort::RunOptions();

        runOpts.SetRunLogVerbosityLevel(2);
        runOpts.SetRunLogSeverityLevel(3);

        std::vector<Ort::Value> inputValues;
        std::vector<Ort::Value> outputValues;
 
        for (uint32_t i = 0; i < m_numInputs; i++)
        {
            const DlTensor *info = inputs[i];

            Ort::Value v = Ort::Value::CreateTensor(m_memInfo,
                                                    (void *)info->data,
                                                    (size_t)info->allocSize,
                                                    info->shape.data(),
                                                    info->shape.size(),
                                                    m_inputTypes[i]);
            inputValues.push_back(std::move(v));
            binding.BindInput(info->name, inputValues[i]);
        }

        for (uint32_t i = 0; i < m_numOutputs; i++)
        {
            const DlTensor *info = outputs[i];

            Ort::Value v = Ort::Value::CreateTensor(m_memInfo,
                                                    (void *)info->data,
                                                    (size_t)info->allocSize,
                                                    info->shape.data(),
                                                    info->shape.size(),
                                                    m_outputTypes[i]);
            outputValues.push_back(std::move(v));
            binding.BindOutput(info->name, outputValues[i]);
        }

        m_session->Run(runOpts, binding);

        return status;
    }

    /**
     * @brief Gets the performance data
     *
     * @return  std::map<std::string, std::pair<float, std::string>>
     *          Map containing performance data where key is performance
     *          metric, and values are (data, unit) 
     * 
     *          'total_time': Total time taken for run (ms)
     *          'core_time': Total time taken barring the io copy time (ms)
     *          'subgraph_time': Total TIDL Subgraphs processing time (ms)
     *          'read_total': Total DDR Read bytes [X for x86 runs]
     *          'write_total': Total DDR Write bytes [X for x86 runs]
     */
    const std::map<std::string, std::pair<float, std::string>> ONNXRT::getPerformance()
    {
        std::map<std::string, std::pair<float, std::string>> data;
        c_api_tidl_benchmark_data benchmarkData;

        OrtSession* session = static_cast<OrtSession*>(*m_session);
        OrtSessionGetTIBenchmarkData_Tidl(session, &benchmarkData);

        uint64_t totalTime = (benchmarkData.run_end - benchmarkData.run_start);
        uint64_t readTotal = (benchmarkData.ddr_read_end - benchmarkData.ddr_read_start);
        uint64_t writeTotal = (benchmarkData.ddr_write_end - benchmarkData.ddr_write_start);
        
        uint64_t subgraphProcTime = 0;
        uint64_t subgraphCopyInTime = 0;
        uint64_t subgraphCopyOutTime = 0;
        for (uint32_t i = 0; i < benchmarkData.num_subgraph_data; i++)
        {
            subgraphProcTime += (benchmarkData.proc_end[i] - benchmarkData.proc_start[i]);
            subgraphCopyInTime += (benchmarkData.copy_in_end[i] - benchmarkData.copy_in_start[i]);
            subgraphCopyOutTime += (benchmarkData.copy_out_end[i] - benchmarkData.copy_out_start[i]);
        }

        uint64_t totalCopyTime = subgraphCopyInTime + subgraphCopyOutTime;

        // Converting to milliseconds
        float totalTimeMs = (float)(totalTime) / 1000000;   
        float totalCopyTimeMs = 0;
        if (benchmarkData.num_subgraph_data == 1)
        {
            totalCopyTimeMs = (float)(totalCopyTime) / 1000000;
        }
        float totalCoreTimeMs = (totalTimeMs - totalCopyTimeMs);
        float subgraphProcTimeMs = (float)(subgraphProcTime) / 1000000;

        data["total_time"] = {totalTimeMs,"ms"};
        data["core_time"] = {totalCoreTimeMs,"ms"};
        data["subgraph_time"] = {subgraphProcTimeMs,"ms"};
        data["read_total"] = {(float)(readTotal),"bytes"};
        data["write_total"] = {(float)(writeTotal),"bytes"};

        return data;

    }

    /**
     * @brief Converts ONNX tensor element data type to TIDL type
     * 
     * Maps ONNX data types to their corresponding TIDL type identifiers and
     * returns the size in bytes of the data type.
     * 
     * @param onnxType The ONNX tensor element data type to convert
     * @param tidlType Reference to store the corresponding TIDL type
     * @param typeName Reference to store the corresponding type name
     * @return int32_t Size in bytes of the data type
     * 
     */
    int32_t ONNXRT::Onnx2TidlType(const ONNXTensorElementDataType &onnxType, int32_t &tidlType, std::string &typeName)
    {
        int32_t size;

        switch (onnxType)
        {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                tidlType = TIDL_SignedChar;
                size = sizeof(int8_t);
                typeName = "int8_t";
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                tidlType = TIDL_UnsignedChar;
                size = sizeof(uint8_t);
                typeName = "uint8_t";
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                tidlType = TIDL_SignedShort;
                size = sizeof(int16_t);
                typeName = "int16_t";
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                tidlType = TIDL_UnsignedShort;
                size = sizeof(uint16_t);
                typeName = "uint16_t";
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                tidlType = TIDL_SignedWord;
                size = sizeof(int32_t);
                typeName = "int32_t";
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                tidlType = TIDL_UnsignedWord;
                size = sizeof(uint32_t);
                typeName = "uint32_t";
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                tidlType = TIDL_SignedDoubleWord;
                size = sizeof(int64_t);
                typeName = "int64_t";
                break;
            
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
                tidlType = TIDL_UnsignedDoubleWord;
                size = sizeof(uint64_t);
                typeName = "uint64_t";
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                tidlType = TIDL_SinglePrecFloat;
                size = sizeof(float);
                typeName = "float";
                break;

            default:
                tidlType = -1;
                size = 0;
                typeName = "invalid";
        }

        return size;
    }

    /**
     * @brief Converts TIDL type to ONNX tensor element data type
     * 
     * Maps TIDL type identifiers to their corresponding ONNX data types and
     * returns the size in bytes of the data type.
     * 
     * @param tidlType The TIDL type to convert
     * @param onnxType Reference to store the corresponding ONNX tensor element data type
     * @return int32_t Size in bytes of the data type
     */
    int32_t ONNXRT::Tidl2OnnxType(const int32_t &tidlType, ONNXTensorElementDataType &onnxType)
    {
        int32_t size;

        switch (tidlType)
        {
            case TIDL_SignedChar:
                onnxType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
                size = sizeof(int8_t);
                break;

            case TIDL_UnsignedChar:
                onnxType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
                size = sizeof(uint8_t);
                break;

            case TIDL_SignedShort:
                onnxType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
                size = sizeof(int16_t);
                break;

            case TIDL_UnsignedShort:
                onnxType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
                size = sizeof(uint16_t);
                break;

            case TIDL_SignedWord:
                onnxType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
                size = sizeof(int32_t);
                break;

            case TIDL_UnsignedWord:
                onnxType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
                size = sizeof(uint32_t);
                break;

            case TIDL_SignedDoubleWord:
                onnxType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
                size = sizeof(int64_t);
                break;
            
            case TIDL_UnsignedDoubleWord:
                onnxType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
                size = sizeof(uint64_t);
                break;

            case TIDL_SinglePrecFloat:
                onnxType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
                size = sizeof(float);
                break;

            default:
                onnxType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
                size = 0;
        }

        return size;
    }

} // namespace onnxrt_wrapper
