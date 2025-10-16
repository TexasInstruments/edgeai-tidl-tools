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

#include "tidlrt_wrapper.h"
#include <dirent.h>
#include <sys/stat.h>
#include <stdexcept>
#include <cstring>
#include <time.h>

namespace tidlrt_wrapper
{

    /**
     * @brief Constructor for the TIDLRT class
     * 
     */
    TIDLRT::TIDLRT()
    {
        m_params.stats = NULL;
        m_params.netPtr = NULL;
        m_params.ioBufDescPtr = NULL;
        m_handle = NULL;
    }

    /**
     * @brief Destructor for the TIDLRT class
     * 
     */
    TIDLRT::~TIDLRT()
    {
        if (m_params.stats != NULL)
        {
            free(m_params.stats);
        }
        if (m_params.netPtr != NULL)
        {
            free(m_params.netPtr);
        }
        if (m_params.ioBufDescPtr != NULL)
        {
            free(m_params.ioBufDescPtr);
        }
        if (m_handle != NULL)
        {
            (void)TIDLRT_deactivate(m_handle);
            (void)TIDLRT_delete(m_handle);
        }
    }

    /**
     * @brief Creates and initializes the inference engine
     * 
     * Sets up the TIDL Runtime session with the specified options and prepares
     * the model for inference. This includes:
     * 1. Validating the artifacts folder for required binary files
     * 2. Checking for multiple network binary files (_net.bin) and raising an error if found
     * 3. Loading network and IO binary files
     * 4. Configuring TIDL runtime parameters
     * 5. Populating input and output tensor information
     * 
     * @param options Map of configuration options for the inference engine, including:
     *                - artifacts_folder: Path to the folder containing model artifacts
     *                - debug_level: (optional) Debug level for tracing and logging
     *                - priority: (optional) Target priority for execution
     *                - max_pre_empt_delay: (optional) Maximum pre-emption delay
     *                - core_number: (optional) Number of cores to use
     *                - core_start_idx: (optional) Starting core index
     * @return int32_t Status code (0 for success, negative for failure)
     * @throws std::runtime_error If required files are missing or multiple network binary files are found
     */
    int32_t TIDLRT::createInfer(std::map<std::string, std::string> &options)
    {
        int32_t status = 0;

        if (options.find("artifacts_folder") == options.end())
        {
            throw std::runtime_error("'artifacts_folder' is not provided.");
        }

        // Get the artifacts folder path
        std::string artifacts_folder = options["artifacts_folder"];
        
        // Get net and io binary file from artifacts_folder
        std::vector<std::string> net_files;
        std::vector<std::string> io_files;
        try
        {
            DIR* dir = opendir(artifacts_folder.c_str());
            if (dir == nullptr)
            {
                throw std::runtime_error("Could not open artifacts folder: " + artifacts_folder);
            }
            
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr)
            {
                std::string filename = entry->d_name;
                if (filename == "." || filename == "..")
                {
                    continue;
                }
                
                std::string full_path = artifacts_folder + "/" + filename;
                struct stat file_stat;
                if (stat(full_path.c_str(), &file_stat) == 0 && S_ISREG(file_stat.st_mode))
                {
                    // Check if file ends with .bin and has _io_ in the name
                    if (endsWith(filename, "_1.bin") && filename.find("_io_") != std::string::npos)
                    {
                        io_files.push_back(full_path);
                    }

                    else if (endsWith(filename, "_net.bin"))
                    {
                        net_files.push_back(full_path);
                    }
                }
            }

            closedir(dir);
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error("Error accessing artifacts folder: " + std::string(e.what()));
        }

        if (net_files.size() == 0)
        {
            throw std::runtime_error("No network binary files found in artifacts folder");
        }
        if (io_files.size() == 0)
        {
            throw std::runtime_error("No io binary files found in artifacts folder");
        }

        if (net_files.size() > 1)
        {
            throw std::runtime_error("Multiple network binary files found in artifacts folder");
        }
        if (io_files.size() > 1)
        {
            throw std::runtime_error("Multiple io binary files found in artifacts folder");
        }

        m_netBinPath = net_files[0];
        m_ioBinPath = io_files[0];

        // Set sTIDLRT_Params_t
        status = TIDLRT_setParamsDefault(&m_params);
        if (status != 0)
        {
            throw std::runtime_error("TIDLRT_setParamsDefault failed.");
        }

        if (options.find("debug_level") != options.end())
        {
            try
            {
                int32_t debugLevel = std::stoi(options["debug_level"]);
                if (debugLevel <= 2)
                {
                    m_params.traceLogLevel = debugLevel;
                    m_params.traceWriteLevel = 0;
                }
                else if (debugLevel == 3)
                {
                    m_params.traceLogLevel = 1;
                    m_params.traceWriteLevel = 1;
                }
                else if (debugLevel == 4)
                {
                    m_params.traceLogLevel = 1;
                    m_params.traceWriteLevel = 3;
                }
                else if (debugLevel == 5)
                {
                    m_params.traceLogLevel = debugLevel;
                    m_params.traceWriteLevel = 3;
                }
                else
                {
                    m_params.traceLogLevel = debugLevel;
                    m_params.traceWriteLevel = 0;
                }  
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
                m_params.targetPriority = std::stoi(options["priority"]);
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
                m_params.maxPreEmptDelay = std::stof(options["max_pre_empt_delay"]);
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
                m_params.coreNum = std::stoi(options["core_number"]);
            }
            catch (const std::invalid_argument& e)
            {
                throw std::runtime_error("Could not parse core_number");
            }
        }
        if (options.find("core_start_idx") != options.end())
        {
            try
            {
                m_params.coreStartIdx = std::stoi(options["core_start_idx"]);
            }
            catch (const std::invalid_argument& e)
            {
                throw std::runtime_error("Could not parse core_start_idx");
            }
        }

        m_params.stats = (sTIDLRT_PerfStats_t*)malloc(sizeof(sTIDLRT_PerfStats_t));

        FILE *fNetwork = fopen(m_netBinPath.c_str(), "rb");
        if (fNetwork == NULL)
        {
            throw std::runtime_error("Unable to open network file");
        }
        fseek(fNetwork, 0, SEEK_END);
        m_params.net_capacity = ftell(fNetwork);
        fseek(fNetwork, 0, SEEK_SET);
        m_params.netPtr = malloc(m_params.net_capacity);
        (void)fread(m_params.netPtr, m_params.net_capacity, 1, fNetwork);
        fclose(fNetwork);

        FILE *fIOBuf = fopen(m_ioBinPath.c_str(), "rb");
        if (fIOBuf == NULL)
        {
            throw std::runtime_error("Unable to open network file");
        }
        fseek(fIOBuf, 0, SEEK_END);
        m_params.io_capacity = ftell(fIOBuf);
        fseek(fIOBuf, 0, SEEK_SET);
        m_params.ioBufDescPtr = malloc(m_params.io_capacity);
        (void)fread(m_params.ioBufDescPtr, m_params.io_capacity, 1, fIOBuf);
        fclose(fIOBuf);
        status = populateInputInfo();
        if (status != 0)
        {
            throw std::runtime_error("TIDLRT_create parsing input info failed");
        }

        status = populateOutputInfo();
        if (status != 0)
        {
            throw std::runtime_error("TIDLRT_create parsing output info failed");
        }

        status = TIDLRT_create(&m_params, &m_handle);
        if (status != 0)
        {
            throw std::runtime_error("TIDLRT_create failed");
        }
        
        return status;
    }

    /**
     * @brief Populates information about the model's input tensors
     * 
     * Queries the TIDL IO buffer descriptor for details about input tensors and populates
     * the corresponding member variables. For each input tensor, it retrieves:
     * - Name
     * - Shape (dimensions)
     * - Data type
     * - Element count and size
     * - Padding information
     * 
     * Also initializes the TIDLRT tensor structures required for inference.
     * 
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t TIDLRT::populateInputInfo()
    {
        int32_t status = 0;
        sTIDL_IOBufDesc_t *ioBuf = (sTIDL_IOBufDesc_t *)m_params.ioBufDescPtr;

        m_numInputs = ioBuf->numInputBuf;

        /* Initialize member variables */
        m_inputs.assign(m_numInputs, DlTensor());
        m_inTensor.assign(m_numInputs, sTIDLRT_Tensor_t());

        for (int32_t i = 0; i < m_numInputs; i++)
        {
            DlTensor *info = &m_inputs[i];
            info->name = (char *)ioBuf->inDataName[i];
            
            info->numDim = 6;
            info->shape.assign(info->numDim, 0);
            info->shape[0] = ioBuf->inNumBatches[i];
            info->shape[1] = ioBuf->inDIM1[i];
            info->shape[2] = ioBuf->inDIM2[i];
            info->shape[3] = ioBuf->inNumChannels[i];
            info->shape[4] = ioBuf->inHeight[i];
            info->shape[5] = ioBuf->inWidth[i];

            info->numElem = 1;
            for (int32_t j = 0; j < info->numDim; j++)
            {
                info->numElem *= info->shape[j];
            }

            info->padT = ioBuf->inPadT[i];
            info->padB = ioBuf->inPadB[i];
            info->padL = ioBuf->inPadL[i];
            info->padR = ioBuf->inPadR[i];

            info->elemSize = Tidl2TidlType(ioBuf->inElementType[i], info->type, info->typeName);

            info->allocSize = ioBuf->inBufSize[i] * info->elemSize;
            if (info->allocSize <= 0)
            {
                printf("Invalid size(%ld) for input(%d).\n",info->allocSize, i);
                status = -1;
                break;
            }

            /**
             * For input tensors, TIDL requires additional memory allocation beyond what's needed for the actual data.
             * This extra memory is used for internal processing and is equal to the size of one padded 2D plane
             * (width + padding) * (height + padding).
             * 
             * allocSize: Total memory size including this extra allocation (as reported by inBufSize)
             * validSize: The actual usable size for tensor data, excluding the extra allocation
             * 
             * The calculation below subtracts the size of one padded 2D plane from allocSize to get validSize.
             * This ensures that data loading operations don't overwrite the reserved memory needed by TIDL.
             */
            info->validSize = info->allocSize - (1 * (info->shape[5] + info->padL + info->padR) * (info->shape[4] + info->padT + info->padB) * info->elemSize);

            m_inTensorPtr[i] = &m_inTensor[i];

            TIDLRT_setTensorDefault(m_inTensorPtr[i]);
            m_inTensorPtr[i]->bufferSize  = ioBuf->inBufSize[i];
            m_inTensorPtr[i]->elementType = ioBuf->inElementType[i];
            m_inTensorPtr[i]->scale = ioBuf->inTensorScale[i];
            m_inTensorPtr[i]->zeroPoint = ioBuf->inZeroPoint[i];
            m_inTensorPtr[i]->layout = ioBuf->inLayout[i];

            m_inTensorPtr[i]->dimValues[0] = ioBuf->inNumBatches[i];
            m_inTensorPtr[i]->dimValues[1] = ioBuf->inDIM1[i];
            m_inTensorPtr[i]->dimValues[2] = ioBuf->inDIM2[i];
            m_inTensorPtr[i]->dimValues[3] = ioBuf->inNumChannels[i];
            m_inTensorPtr[i]->dimValues[4] = ioBuf->inHeight[i];
            m_inTensorPtr[i]->dimValues[5] = ioBuf->inWidth[i];
        
            m_inTensorPtr[i]->padValues[0] = ioBuf->inPadL[i];
            m_inTensorPtr[i]->padValues[1] = ioBuf->inPadR[i];
            m_inTensorPtr[i]->padValues[2] = ioBuf->inPadT[i];
            m_inTensorPtr[i]->padValues[3] = ioBuf->inPadB[i];

            m_inTensorPtr[i]->pitch[4] = (ioBuf->inPadL[i] + ioBuf->inWidth[i] + ioBuf->inPadR[i]);
            m_inTensorPtr[i]->pitch[3] = ioBuf->inChannelPitch[i];
            m_inTensorPtr[i]->pitch[2] = m_inTensorPtr[i]->pitch[3] * ioBuf->inNumChannels[i];;
            m_inTensorPtr[i]->pitch[1] = m_inTensorPtr[i]->pitch[2] * ioBuf->inDIM2[i];
            m_inTensorPtr[i]->pitch[0] = m_inTensorPtr[i]->pitch[1] * ioBuf->inNumBatches[i];
    
            strcpy((char*)m_inTensorPtr[i]->name,(char*)ioBuf->inDataName[i]);

        }

        return status;
    }

    /**
     * @brief Gets details about the model's input tensors
     * 
     * @return const std::vector<DlTensor>* Pointer to vector of input tensor details
     */
    const std::vector<DlTensor>* TIDLRT::getInputDetails()
    {
        return &m_inputs;
    }

    /**
     * @brief Populates information about the model's output tensors
     * 
     * Queries the TIDL IO buffer descriptor for details about output tensors and populates
     * the corresponding member variables. For each output tensor, it retrieves:
     * - Name
     * - Shape (dimensions)
     * - Data type
     * - Element count and size
     * - Padding information
     * 
     * Also initializes the TIDLRT tensor structures required for inference.
     * 
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t TIDLRT::populateOutputInfo()
    {
        int32_t status = 0;
        sTIDL_IOBufDesc_t *ioBuf = (sTIDL_IOBufDesc_t *)m_params.ioBufDescPtr;

        m_numOutputs = ioBuf->numOutputBuf;

        /* Initialize member variables */
        m_outputs.assign(m_numOutputs, DlTensor());
        m_outTensor.assign(m_numOutputs, sTIDLRT_Tensor_t());
        
        for (int32_t i = 0; i < m_numOutputs; i++)
        {
            DlTensor *info = &m_outputs[i];
            info->name = (char *)ioBuf->outDataName[i];
            
            info->numDim = 6;
            info->shape.assign(info->numDim, 0);
            info->shape[0] = ioBuf->outNumBatches[i];
            info->shape[1] = ioBuf->outDIM1[i];
            info->shape[2] = ioBuf->outDIM2[i];
            info->shape[3] = ioBuf->outNumChannels[i];
            info->shape[4] = ioBuf->outHeight[i];
            info->shape[5] = ioBuf->outWidth[i];

            info->numElem = 1;
            for (int32_t j = 0; j < info->numDim; j++)
            {
                info->numElem *= info->shape[j];
            }

            info->padT = ioBuf->outPadT[i];
            info->padB = ioBuf->outPadB[i];
            info->padL = ioBuf->outPadL[i];
            info->padR = ioBuf->outPadR[i];

            info->elemSize = Tidl2TidlType(ioBuf->outElementType[i], info->type, info->typeName);

            info->allocSize = ioBuf->outBufSize[i] * info->elemSize;
            if (info->allocSize <= 0)
            {
                printf("Invalid size(%ld) for input(%d).\n",info->allocSize, i);
                status = -1;
                break;
            }
            /**
             * For output tensors, the entire allocated buffer is usable.
             * Unlike input tensors, there's no need for extra memory allocation for internal processing.
             * Therefore, validSize equals allocSize for output tensors.
             */
            info->validSize = info->allocSize;

            m_outTensorPtr[i] = &m_outTensor[i];

            TIDLRT_setTensorDefault(m_outTensorPtr[i]);
            m_outTensorPtr[i]->bufferSize  = ioBuf->outBufSize[i];
            m_outTensorPtr[i]->elementType = ioBuf->outElementType[i];
            m_outTensorPtr[i]->scale = ioBuf->outTensorScale[i];
            m_outTensorPtr[i]->zeroPoint = ioBuf->outZeroPoint[i];
            m_outTensorPtr[i]->layout = ioBuf->outLayout[i];

            m_outTensorPtr[i]->dimValues[0] = ioBuf->outNumBatches[i];
            m_outTensorPtr[i]->dimValues[1] = ioBuf->outDIM1[i];
            m_outTensorPtr[i]->dimValues[2] = ioBuf->outDIM2[i];
            m_outTensorPtr[i]->dimValues[3] = ioBuf->outNumChannels[i];
            m_outTensorPtr[i]->dimValues[4] = ioBuf->outHeight[i];
            m_outTensorPtr[i]->dimValues[5] = ioBuf->outWidth[i];
        
            m_outTensorPtr[i]->padValues[0] = ioBuf->outPadL[i];
            m_outTensorPtr[i]->padValues[1] = ioBuf->outPadR[i];
            m_outTensorPtr[i]->padValues[2] = ioBuf->outPadT[i];
            m_outTensorPtr[i]->padValues[3] = ioBuf->outPadB[i];

            m_outTensorPtr[i]->pitch[4] = (ioBuf->outPadL[i] + ioBuf->outWidth[i] + ioBuf->outPadR[i]);
            m_outTensorPtr[i]->pitch[3] = ioBuf->outChannelPitch[i];
            m_outTensorPtr[i]->pitch[2] = m_outTensorPtr[i]->pitch[3] * ioBuf->outNumChannels[i];;
            m_outTensorPtr[i]->pitch[1] = m_outTensorPtr[i]->pitch[2] * ioBuf->outDIM2[i];
            m_outTensorPtr[i]->pitch[0] = m_outTensorPtr[i]->pitch[1] * ioBuf->outNumBatches[i];
    
            strcpy((char*)m_outTensorPtr[i]->name,(char*)ioBuf->outDataName[i]);
        }

        return status;
    }

    /**
     * @brief Gets details about the model's output tensors
     * 
     * @return const std::vector<DlTensor>* Pointer to vector of output tensor details
     */
    const std::vector<DlTensor>* TIDLRT::getOutputDetails()
    {
        return &m_outputs;
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
     *          'graph_time': Total TIDL graph processing time (ms)
     *          'read_total': Total DDR Read bytes [X for x86 runs]
     *          'write_total': Total DDR Write bytes [X for x86 runs]
     */
    const std::map<std::string, std::pair<float, std::string>> TIDLRT::getPerformance()
    {
        std::map<std::string, std::pair<float, std::string>> data;

        if (m_handle == NULL || m_params.stats == NULL)
        {
            return data;
        }

        // Get performance stats from TIDL runtime
        sTIDLRT_PerfStats_t *stats = m_params.stats;

        uint64_t copyInTime = (stats->cpIn_time_end - stats->cpIn_time_start);
        uint64_t copyOutTime = (stats->cpOut_time_end - stats->cpOut_time_start);
        uint64_t procTime = (stats->proc_time_end - stats->proc_time_start);
        uint64_t readTotal = (m_ddrReadEnd - m_ddrReadStart);
        uint64_t writeTotal = (m_ddrWriteEnd - m_ddrWriteStart);

        float totalTimeMs = (float)(m_invokeEnd - m_invokeStart) / 1000000;
        float copyTimeMs = (float)(copyInTime + copyOutTime) / 1000000;
        float coreTimeMs = (float)(totalTimeMs - copyTimeMs) / 1000000;
        float procTimeMs = (float)(procTime) / 1000000;

        data["total_time"] = {totalTimeMs, "ms"};
        data["core_time"] = {coreTimeMs, "ms"};
        data["graph_time"] = {procTimeMs, "ms"};
        data["read_total"] = {(float)readTotal, "bytes"};
        data["write_total"] = {(float)(writeTotal), "bytes"};

        return data;
    }

    /**
     * @brief Prints detailed information about the model and its tensors
     * 
     * Outputs model path, input/output tensor counts, and detailed information
     * about each tensor including name, type, shape, and size.
     */
    void TIDLRT::dumpInfo()
    {
        printf("Net Path = %s\n", m_netBinPath.c_str());
        printf("IO Path = %s\n", m_ioBinPath.c_str());
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
     * 1. Verifies that the TIDL Runtime handle has been created
     * 2. Sets up input and output tensor pointers with the provided data buffers
     * 3. Detects and configures shared memory if used
     * 4. Invokes the TIDL Runtime for model execution
     * 
     * @param inputs Vector of input tensors containing the data for inference
     * @param outputs Vector of output tensors to store the inference results
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t TIDLRT::runInfer(const std::vector<DlTensor *> &inputs, std::vector<DlTensor *> &outputs)
    {
        int32_t status = 0;
        struct timespec timestamp;

        if (m_handle == NULL)
        {
            status = -1;
            printf("[ERROR] TIDLRT Handle is not created");
            return status;
        }

        for (uint32_t i = 0; i < m_numInputs; i++)
        {
            m_inTensorPtr[i]->ptr = inputs[i]->data;
            if (TIDLRT_isSharedMem(m_inTensorPtr[i]->ptr))
            {
                m_inTensorPtr[i]->memType = TIDLRT_MEM_SHARED;
            }
        }

        for (uint32_t i = 0; i < m_numOutputs; i++)
        {
            m_outTensorPtr[i]->ptr = outputs[i]->data;
            if (TIDLRT_isSharedMem(m_outTensorPtr[i]->ptr))
            {
                m_outTensorPtr[i]->memType = TIDLRT_MEM_SHARED;
            }
        }

        TIDLRT_getDdrStats(&m_ddrReadStart, &m_ddrWriteStart);
        (void)clock_gettime(CLOCK_MONOTONIC, &timestamp);
        m_invokeStart = ((uint64_t)timestamp.tv_sec * (uint64_t)1000000000ULL) + (uint64_t)timestamp.tv_nsec;

        status = TIDLRT_invoke(m_handle, m_inTensorPtr, m_outTensorPtr);

        (void)clock_gettime(CLOCK_MONOTONIC, &timestamp);
        m_invokeEnd = ((uint64_t)timestamp.tv_sec * (uint64_t)1000000000ULL) + (uint64_t)timestamp.tv_nsec;
        TIDLRT_getDdrStats(&m_ddrReadEnd, &m_ddrWriteEnd);

        return status;
    }

    /**
     * @brief Converts TIDL tensor element data type to standard C++ type
     * 
     * Maps TIDL data types to their corresponding C++ type identifiers and
     * returns the size in bytes of the data type.
     * 
     * @param type The TIDL tensor element data type to convert
     * @param tidlType Reference to store the corresponding TIDL type (may be modified for certain types)
     * @param typeName Reference to store the corresponding human-readable type name
     * @return int32_t Size in bytes of the data type
     * 
     */
    int32_t TIDLRT::Tidl2TidlType(const int32_t &type, int32_t &tidlType, std::string &typeName)
    {
        int32_t size;
        tidlType = type;

        switch (type)
        {
            case TIDL_SignedChar:
                size = sizeof(int8_t);
                typeName = "int8_t";
                break;

            case TIDL_UnsignedChar:
                size = sizeof(uint8_t);
                typeName = "uint8_t";
                break;

            case TIDL_SignedShort:
                size = sizeof(int16_t);
                typeName = "int16_t";
                break;

            case TIDL_UnsignedShort:
                size = sizeof(uint16_t);
                typeName = "uint16_t";
                break;

            case TIDL_SignedWord:
                tidlType = 5;
                size = sizeof(int32_t);
                typeName = "int32_t";
                break;

            case TIDL_UnsignedWord:
                size = sizeof(uint32_t);
                typeName = "uint32_t";
                break;

            case TIDL_SignedDoubleWord:
                tidlType = 8;
                size = sizeof(int64_t);
                typeName = "int64_t";
                break;
            
            case TIDL_UnsignedDoubleWord:
                size = sizeof(uint64_t);
                typeName = "uint64_t";
                break;

            case TIDL_SinglePrecFloat:
                size = sizeof(float);
                typeName = "float";
                break;

            default:
                size = 0;
                typeName = "invalid";
        }

        return size;
    }


    /**
     * @brief Checks if a string ends with a specific suffix
     * 
     * Utility method to determine if a given string ends with a specified suffix.
     * Used in createInfer to identify network and IO binary files by their filename patterns.
     * 
     * @param str The string to check
     * @param end The suffix to look for
     * @return bool True if the string ends with the specified suffix, false otherwise
     */
    bool TIDLRT::endsWith(const std::string& str, const std::string& end)
    {
        if (end.size() > str.size())
        {
            return false;
        }
        return str.substr(str.size() - end.size()) == end;
    }


} // namespace tidlrt_wrapper
