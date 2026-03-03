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

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <yaml-cpp/yaml.h>
#include <iomanip>
#include <sstream>

#include "preemption_example.h"
#include "argsparser.h"
#include "utils.h"
#include "datasetloader/dataset_loader.h"

#define DEFAULT_MEM_ALIGNMENT (64)

namespace fs = std::filesystem;

pthread_mutex_t gPriorityLock;
pthread_barrier_t gBarrier;

struct ModelInfo
{
    std::string                         model;

    int                                 priority;

    float                               maxPreemptDelay;

    std::string                         runtime;
};

/* Specify tests to be run -- Vector of tests, each test has N threads, with each thread running M models */
static std::vector<std::vector<ModelInfo>> gModelsMap = 
{
    /* Test 1 */
    {
        /* Threads*/
        {
            "ss-ort-deeplabv3lite_mobilenetv2", 0, FLT_MAX, "tidlrt" // {Model, Priority, Max Preempt Delay, runtime}
        },
        {
            "cl-ort-resnet18-v1", 0, FLT_MAX, "tidlrt"
        }
    },
    /* Test 2 */
    {
        /* Threads*/
        {
            "ss-ort-deeplabv3lite_mobilenetv2", 1, FLT_MAX, "tidlrt"
        },
        {
            "cl-ort-resnet18-v1", 0, FLT_MAX, "tidlrt"
        }
    },
    /* Test 3 */
    {
        /* Threads*/
        {
            "ss-ort-deeplabv3lite_mobilenetv2", 1, 7, "tidlrt"
        },
        {
            "cl-ort-resnet18-v1", 0, FLT_MAX, "tidlrt"
        }
    },
    /* Test 4 */
    {
        /* Threads*/
        {
            "ss-ort-deeplabv3lite_mobilenetv2", 1, 3, "tidlrt"
        },
        {
            "cl-ort-resnet18-v1", 0, FLT_MAX, "tidlrt"
        }
    },
    /* Test 5 */
    {
        /* Threads*/
        {
            "ss-ort-deeplabv3lite_mobilenetv2", 1, 0, "tidlrt"
        },
        {
            "cl-ort-resnet18-v1", 0, FLT_MAX, "tidlrt"
        }
    }
};

struct RunConfig
{
    ModelInfo                           modelInfo;

    bool                                isPreemptionEnabled;

    std::string                         outputDir;

    int32_t                             duration;

    int32_t                             numItr;

    float                               avgInvokeTime;

    std::map<std::string, std::string>  inferOptions;
};

/**
 * @class PreemptionExample
 * @brief A class that demonstrates running inference with different runtime types (ONNX, TFLite, TVM, TIDL)
 *
 * This class provides functionality to run machine learning models using different runtime backends.
 * It supports ONNX Runtime, TensorFlow Lite Runtime, TVM Runtime and TIDL Runtime. The class handles:
 * - Parsing and validating configuration from YAML files
 * - Setting up runtime sessions based on model type
 * - Managing memory allocation for input and output tensors
 * - Loading input data from files (.bin or .npz)
 * - Running inference on the loaded models
 * - Saving output results to files
 *
 */
class PreemptionExample
{
public:
    ModelInfo                           m_modelInfo;
    bool                                m_verbose;
    std::string                         m_artifactsBaseDir;
    std::string                         m_outputsBaseDir;
    std::string                         m_artifactsDir;

public:

    /**
     * @brief Constructor for the PreemptionExample class
     * 
     * Initializes the PreemptionExample object with the provided parameters,
     * sets up paths for configuration, artifacts, and outputs,
     * and parses and validates the configuration.
     * 
     * @param ModelInfo ModelInfo for model to run
     * @param verbose Flag to enable verbose output. Default: false
     * @throws std::runtime_error If configuration parsing or validation fails
     */
    PreemptionExample(ModelInfo &modelInfo, bool verbose = false)
    {
        m_modelInfo = modelInfo;
        m_verbose = verbose;
        
        // Determine the source directory
        m_artifactsBaseDir = (fs::path("../../model-artifacts")).string();

        m_artifactsDir = (fs::path(m_artifactsBaseDir) / fs::path(m_modelInfo.model) / fs::path("artifacts")).string();
        if (!fs::exists(m_artifactsDir) || !fs::is_directory(m_artifactsDir))
        {
            printf("[ERROR][%s] artifacts folder %s does not exist or is not a directory\n", m_modelInfo.model.c_str(), m_artifactsDir.c_str());
            throw std::runtime_error("Could not find artifacts directory");
        }

    }

public:

    /**
     * @brief Main method to run inference for all configured models
     * 
     * This method iterates through all configured models, sets up the appropriate
     * runtime session based on the model type, and runs inference using the
     * runInferenceWithSession template method.
     * 
     * @param runConfig RunConfig
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t run(RunConfig &runConfig)
    {
        int32_t status = 0;

        m_outputsBaseDir = runConfig.outputDir;
        
        runConfig.modelInfo = m_modelInfo;
        runConfig.inferOptions["artifacts_folder"] = m_artifactsDir;
        runConfig.inferOptions["priority"] = std::to_string(m_modelInfo.priority);
        runConfig.inferOptions["max_pre_empt_delay"] = std::to_string(m_modelInfo.maxPreemptDelay);

        std::string modelName = runConfig.modelInfo.model;
        std::string runtime = runConfig.modelInfo.runtime;

        if (runConfig.isPreemptionEnabled)
        {
            printf("\nRunning %s with %s runtime - priority = %s and max_pre_empt_delay = %s...\n", modelName.c_str(), runtime.c_str(), runConfig.inferOptions["priority"].c_str(), runConfig.inferOptions["max_pre_empt_delay"].c_str());
        }
        else
        {
            printf("\nRunning %s with %s runtime - pre-emption disabled...\n", modelName.c_str(), runtime.c_str());
        }

        if (runtime == "tidlrt")
        {      
            TIDLRT session;
            status = runInferenceWithSession(runConfig, session);
        }
        else
        {
            printf("[ERROR][%s] Invalid runtime %s. Only 'tidlrt' is supported in this application.\n", modelName.c_str(), runtime.c_str());
            status = -1;
        }
            
        return status;
    }

private:

    /**
     * @brief Template method to run inference with a specific session type
     * 
     * This method encapsulates the common inference workflow for all runtime types:
     * 1. Creates the inference session
     * 2. Allocates memory for input and output tensors
     * 3. Processes each frame (load input, run inference, save output)
     * 4. Frees allocated memory
     * 
     * @tparam SessionType The type of session (ONNXRT, TFLiteRT, TVMRT or TIDLRT)
     * @param runConfig Configuration for the current run
     * @param session The session object to use for inference
     * @return int32_t Status code (0 for success, negative for failure)
     */
    template<typename SessionType>
    int32_t runInferenceWithSession(RunConfig &runConfig, SessionType &session)
    {
        int32_t status = 0;
        int32_t i = 0;
        int32_t memoryAlignent = DEFAULT_MEM_ALIGNMENT;

        std::string modelName = runConfig.modelInfo.model;

        // Create infer session
        pthread_mutex_lock(&gPriorityLock);
        try
        {
            status = session.createInfer(runConfig.inferOptions);
        }
        catch(const std::exception& e)
        {
            status = -1;
        }
        pthread_mutex_unlock(&gPriorityLock);

        if (status != 0)
        {
            printf("[ERROR][%s] Could not create infer session\n", modelName.c_str());
            return -1;
        }

        // Print session information if verbose mode is enabled
        if (m_verbose)
        {
            printf("\n[INFO][%s] Session information:\n", modelName.c_str());
            session.dumpInfo();
            printf("\n");
        }

        // Get input details and prepare input tensors
        const std::vector<DlTensor>* inputInfo = session.getInputDetails();
        std::vector<DlTensor> inputs(inputInfo->size(), DlTensor());
        std::vector<DlTensor *> inputsPtr(inputInfo->size(), nullptr);
        pthread_mutex_lock(&gPriorityLock);
        status = allocateTensors(inputInfo, inputs, inputsPtr, memoryAlignent);
        pthread_mutex_unlock(&gPriorityLock);
        if (status != 0)
        {
            printf("[ERROR][%s] Could not prepare input tensors\n", modelName.c_str());
            return -1;
        }

        // Get output details and prepare output tensors
        const std::vector<DlTensor>* outputInfo = session.getOutputDetails();
        std::vector<DlTensor> outputs(outputInfo->size(), DlTensor());
        std::vector<DlTensor *> outputsPtr(outputInfo->size(), nullptr);
        pthread_mutex_lock(&gPriorityLock);
        status = allocateTensors(outputInfo, outputs, outputsPtr, memoryAlignent);
        pthread_mutex_unlock(&gPriorityLock);
        if (status != 0)
        {
            printf("[ERROR][%s] Could not prepare output tensors\n", modelName.c_str());
            freeTensors(inputsPtr);
            return -1;
        }

        // Load the inputs (pseudo-random)
        pthread_mutex_lock(&gPriorityLock);
        status = loadInput(inputsPtr);
        pthread_mutex_unlock(&gPriorityLock);
        if (status != 0)
        {
            printf("[ERROR][%s] Could not load input data for frame %d\n", modelName.c_str(), i+1);
            return status;
        }

        // Wait for all threads to synchronize before runs start across threads
        pthread_barrier_wait(&gBarrier);

        printf("[%s] Running for %d seconds...\n", modelName.c_str(), runConfig.duration);

        struct timeval startTime, stopTime;
        struct timeval inovkeStartTime, invokeStopTime;
        float totalInvokeTime = 0.0;
        float avgInvokeTime = 0.0;
        int64_t numItr = 0;

        // Run for specified duration
        auto finish = system_clock::now() + seconds(runConfig.duration);
        gettimeofday(&startTime, nullptr);
        do
        {
            // Run inference
            gettimeofday(&inovkeStartTime, nullptr);
            status = session.runInfer(inputsPtr, outputsPtr);
            gettimeofday(&invokeStopTime, nullptr);

            totalInvokeTime += (invokeStopTime.tv_sec * 1000000 + invokeStopTime.tv_usec) - (inovkeStartTime.tv_sec * 1000000 + inovkeStartTime.tv_usec);
            numItr++;

            if (status != 0)
            {
                printf("[ERROR][%s] Could not run inference for frame %d\n", modelName.c_str(), i+1);
                continue;
            }
        } while (system_clock::now() < finish);
        gettimeofday(&stopTime, nullptr);

        avgInvokeTime = (totalInvokeTime) / (numItr * 1000); // Avering across iterations runs and converting to milliseconds
        
        printf("\n");
        printf("[%s] Total Iterations = %ld \n", modelName.c_str(), numItr);
        printf("[%s] Average Processing Time = %f \n", modelName.c_str(), avgInvokeTime);
        printf("\n");

        runConfig.numItr = numItr;
        runConfig.avgInvokeTime = avgInvokeTime;

        // Save output
        pthread_mutex_lock(&gPriorityLock);
        status = saveOutput(outputsPtr);
        pthread_mutex_unlock(&gPriorityLock);
        if (status != 0)
        {
            printf("[ERROR][%s] Could not save output for frame %d\n", modelName.c_str(), i+1);
        }

        freeTensors(inputsPtr);
        freeTensors(outputsPtr);
        return status;
    }

    /**
     * @brief Allocates memory for tensors based on the provided details
     * 
     * This method creates and initializes tensors with the appropriate memory alignment.
     * It copies tensor information from the detail vector and allocates memory for each tensor.
     * 
     * @param detail Vector containing tensor details
     * @param tensor Vector to store the created tensors
     * @param tensorPtr Vector to store pointers to the created tensors
     * @param memAlignment Memory alignment requirement in bytes
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t allocateTensors(const std::vector<DlTensor>* &detail,
                            std::vector<DlTensor> &tensor,
                            std::vector<DlTensor *> &tensorPtr,
                            int32_t memAlignment)
    {
        int32_t status = 0;
        for (int32_t i = 0; i < detail->size(); i++)
        {
            const DlTensor *info = &(detail->at(i));
            tensorPtr[i] = &tensor[i];
            tensorPtr[i]->name = info->name;
            tensorPtr[i]->allocSize = info->allocSize;
            tensorPtr[i]->validSize = info->validSize;
            tensorPtr[i]->shape = info->shape;
            tensorPtr[i]->numDim = info->numDim;
            tensorPtr[i]->type = info->type;
            tensorPtr[i]->numElem = info->numElem;
            tensorPtr[i]->elemSize = info->elemSize;
            tensorPtr[i]->typeName = info->typeName;

            // Allocate memory using allocSize
            tensorPtr[i]->data = allocTensorMem(tensorPtr[i]->allocSize, memAlignment);
            if (tensorPtr[i]->data == NULL)
            {
                status = -1;
                break;
            }
        }
        return status;
    }

    /**
     * @brief Allocates memory for a tensor with specified alignment
     * 
     * This method attempts to allocate memory in shared memory.
     * If that fails or if TIDL offload is disabled, it falls back
     * to regular aligned memory allocation.
     * 
     * @param size Size of memory to allocate in bytes
     * @param alignment Memory alignment requirement in bytes (default: 64)
     * @return void* Pointer to the allocated memory, or NULL if allocation failed
     */
    void *allocTensorMem(size_t size, int32_t alignment = 64)
    {
        void *ptr = NULL;
        
        ptr = TIDLRT_allocSharedMem(alignment, (int32_t)size);
        if (ptr == NULL)
        {
            printf("[WARN] Tensor of size %ld could not be allocate in shared memory. Allocating on default heap region\n", size);
            size_t alignedSize = ((size + alignment - 1) / alignment) * alignment;

            // Use posix_memalign instead of malloc to ensure alignment
            int result = posix_memalign(&ptr, alignment, alignedSize);
            if (result != 0)
            {
                ptr = NULL;
            }
        }

        if (ptr == NULL)
        {
            printf("[ERROR] Could not allocate memory for a tensor of size %ld\n", size);
        }

        return ptr;
    }

    /**
     * @brief Frees memory for all tensors in the provided vector
     * 
     * @param tensorPtr Vector of pointers to tensors whose memory should be freed
     */
    void freeTensors(std::vector<DlTensor *> &tensorPtr)
    {
        for (int32_t i = 0; i < tensorPtr.size(); i++)
        {
            freeTensorMem(tensorPtr[i]->data);
        }
    }

    /**
     * @brief Frees memory allocated for a tensor
     * 
     * This method checks if the memory was allocated in shared memory or regular heap
     * and calls the appropriate deallocation function.
     * 
     * @param ptr Pointer to the memory to free
     */
    void freeTensorMem(void *ptr)
    {
        if (ptr != NULL)
        {
            if(TIDLRT_isSharedMem(ptr))
            {
                TIDLRT_freeSharedMem(ptr);
            }
            else
            {
                free(ptr);
            }
        }
        return;
    }
  
    /**
     * @brief Loads input data from a file into the provided tensors
     * 
     * This method fills the input array with pseudo-random numbers
     * 0,1,2....255,0,1,2...255...
     * 
     * @param inputsPtr Vector of pointers to tensors where input data should be loaded
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t loadInput(std::vector<DlTensor *> &inputsPtr)
    {
        int32_t status = 0;
        int32_t i = 0;
        int32_t j = 0;

        for (i = 0; i < inputsPtr.size(); i++)
        {
            for (j = 0; j < inputsPtr[i]->allocSize; j++)
            {
                *((uint8_t *)inputsPtr[i]->data + j) = (j % 256);
            }
        }

        return status;
    }

    /**
     * @brief Saves output tensors to binary files
     * 
     * This method creates an output directory structure. 
     * and then converts each tensor to float32 format and saves it as a binary file.
     * 
     * @param tensorPtr Vector of pointers to output tensors
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t saveOutput(std::vector<DlTensor *> &tensorPtr)
    {
        int32_t status = 0;
        std::string outputDir = (fs::path(m_outputsBaseDir)).string();
        (void) fs::create_directories(outputDir);
    
        for (int32_t i = 0; i < tensorPtr.size(); i++)
        {
            std::string outputFileName(tensorPtr[i]->name);
            outputFileName += ".bin";

            size_t pos = 0;
            while ((pos = outputFileName.find('/', pos)) != std::string::npos)
            {
                outputFileName.replace(pos, 1, 1, '_');
                pos++;
            }

            std::string outputFilePath = (fs::path(outputDir) / fs::path(outputFileName)).string();
            std::ofstream file(outputFilePath, std::ios::out | std::ios::binary);
            if (file.is_open())
            {
                // Convert data to float32 before saving
                float* buf = new float[tensorPtr[i]->numElem];
                if (buf == nullptr)
                {
                    printf("[ERROR] Could not allocate memory for float conversion\n");
                    status = -1;
                    file.close();
                    break;
                }
                
                // Convert the data to float32
                switch (tensorPtr[i]->type)
                {
                    case TIDL_UnsignedChar:
                        for (int64_t j = 0; j < tensorPtr[i]->numElem; j++)
                            buf[j] = static_cast<float>(reinterpret_cast<const uint8_t*>(tensorPtr[i]->data)[j]);
                        break;
                    case TIDL_SignedChar:
                        for (int64_t j = 0; j < tensorPtr[i]->numElem; j++)
                            buf[j] = static_cast<float>(reinterpret_cast<const int8_t*>(tensorPtr[i]->data)[j]);
                        break;
                    case TIDL_UnsignedShort:
                        for (int64_t j = 0; j < tensorPtr[i]->numElem; j++)
                            buf[j] = static_cast<float>(reinterpret_cast<const uint16_t*>(tensorPtr[i]->data)[j]);
                        break;
                    case TIDL_SignedShort:
                        for (int64_t j = 0; j < tensorPtr[i]->numElem; j++)
                            buf[j] = static_cast<float>(reinterpret_cast<const int16_t*>(tensorPtr[i]->data)[j]);
                        break;
                    case TIDL_UnsignedWord:
                        for (int64_t j = 0; j < tensorPtr[i]->numElem; j++)
                            buf[j] = static_cast<float>(reinterpret_cast<const uint32_t*>(tensorPtr[i]->data)[j]);
                        break;
                    case TIDL_SignedWord:
                        for (int64_t j = 0; j < tensorPtr[i]->numElem; j++)
                            buf[j] = static_cast<float>(reinterpret_cast<const int32_t*>(tensorPtr[i]->data)[j]);
                        break;
                    case TIDL_SignedDoubleWord:
                        for (int64_t j = 0; j < tensorPtr[i]->numElem; j++)
                            buf[j] = static_cast<float>(reinterpret_cast<const int64_t*>(tensorPtr[i]->data)[j]);
                        break;
                    case TIDL_UnsignedDoubleWord:
                        for (int64_t j = 0; j < tensorPtr[i]->numElem; j++)
                            buf[j] = static_cast<float>(reinterpret_cast<const uint64_t*>(tensorPtr[i]->data)[j]);
                        break;
                    case TIDL_SinglePrecFloat:
                        memcpy(buf, tensorPtr[i]->data, tensorPtr[i]->numElem * sizeof(float));
                        break;
                    default:
                        printf("[WARN] Unknown TIDL type %d, copying raw data\n", tensorPtr[i]->type);
                        memcpy(buf, tensorPtr[i]->data, tensorPtr[i]->numElem * sizeof(float));
                        break;
                }
                
                file.write(reinterpret_cast<const char*>(buf), tensorPtr[i]->numElem * sizeof(float));
                file.close();
                
                delete[] buf;
                
                printf("Output saved: %s\n", outputFilePath.c_str());
            }
            else
            {
                status = -1;
                break;
            }
        }
        return status;
    }

};

struct ModelThreadArgs
{
    PreemptionExample   *preemptionClassPtr;
    RunConfig            runConfig;
};

void *threadRunner(void *args)
{
    int32_t status = 0;
    ModelThreadArgs *tArgs = static_cast<ModelThreadArgs*>(args);
    status = tArgs->preemptionClassPtr->run(tArgs->runConfig);
    return NULL;
}

struct TestResult
{
    int testId;
    int n1Priority;
    float n1Delay;
    int n2Priority;
    float n2Delay;
    float n1InvokeTime;
    float n2InvokeTime;
};

// Function to print a formatted results table
void printResultsTable(const std::vector<TestResult>& results)
{
    if (results.empty())
    {
        return;
    }

    // Calculate column widths
    size_t testIdWidth = 8;
    size_t priorityWidth = 8;
    size_t delayWidth = 12;
    size_t timeWidth = 15;

    // Print table header
    printf("\n");
    printf("+-");
    for (size_t i = 0; i < testIdWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < priorityWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < delayWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < priorityWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < delayWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < timeWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < timeWidth; i++) printf("-");
    printf("-+\n");

    printf("| %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s |\n",
           (int)testIdWidth, "Test ID",
           (int)priorityWidth, "N1 - Pri",
           (int)delayWidth, "N1 - Delay",
           (int)priorityWidth, "N2 - Pri",
           (int)delayWidth, "N2 - Delay",
           (int)timeWidth, "N1 - Time (ms)",
           (int)timeWidth, "N2 - Time (ms)");

    printf("+-");
    for (size_t i = 0; i < testIdWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < priorityWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < delayWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < priorityWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < delayWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < timeWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < timeWidth; i++) printf("-");
    printf("-+\n");

    // Print table rows
    for (const auto& result : results)
    {
        std::string n1Delay;
        std::string n2Delay;
        std::stringstream streamn1;
        std::stringstream streamn2;

        if (result.n1Delay == FLT_MAX)
        {
            n1Delay = "FLT_MAX";
        }
        else
        {
            streamn1 << std::fixed << std::setprecision(2) << result.n1Delay;
            n1Delay = streamn1.str();
        }
        if (result.n2Delay == FLT_MAX)
        {
            n2Delay = "FLT_MAX";
        }
        else
        {
            streamn2 << std::fixed << std::setprecision(2) << result.n2Delay;
            n2Delay = streamn2.str();
        }
        
        printf("| %-*d | %-*d | %-*s | %-*d | %-*s | %-*.3f | %-*.3f |\n",
               (int)testIdWidth, result.testId,
               (int)priorityWidth, result.n1Priority,
               (int)delayWidth, n1Delay.c_str(),
               (int)priorityWidth, result.n2Priority,
               (int)delayWidth, n2Delay.c_str(),
               (int)timeWidth, result.n1InvokeTime,
               (int)timeWidth, result.n2InvokeTime);
    }

    printf("+-");
    for (size_t i = 0; i < testIdWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < priorityWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < delayWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < priorityWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < delayWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < timeWidth; i++) printf("-");
    printf("-+-");
    for (size_t i = 0; i < timeWidth; i++) printf("-");
    printf("-+\n");
}

int main(int argc, char *argv[])
{
    int32_t status = 0;

    int32_t numTests = gModelsMap.size();
    std::vector<TestResult> referenceTestResult;
    std::vector<TestResult> preemptionTestResult;

    for (int32_t i = 0; i < numTests; i++)
    {
        printf("\n==================== RUNNING TEST %d ====================\n", (i+1));

        bool skip = false;

        auto& test = gModelsMap[i];
        int32_t numThreads = test.size();
        TestResult testResult;

        std::vector<PreemptionExample> preemptionClasses;

        pthread_t threadId[numThreads];
        pthread_barrierattr_t barrierAttr;
        pthread_attr_t threadAttr;

        ModelThreadArgs ModelThreadArgs[numThreads];

        // Initialize PreemptionExample class
        for(int32_t j = 0; j < numThreads; j++)
        {
            ModelInfo &modelInfo =  test[j];
            try
            {
                preemptionClasses.push_back(PreemptionExample(modelInfo));
            }
            catch(const std::exception& e)
            {
                printf("[ERROR] Could not create PreemptionExample class for %s. Skipping TEST %d.\n", modelInfo.model.c_str(), (i+1));
                skip = true;
                break;
            }
        }

        if (skip)
        {
            continue;
        }

        for(int32_t j = 0; j < numThreads; j++)
        {
            ModelThreadArgs[j].preemptionClassPtr = &preemptionClasses[j];
        }

        (void)pthread_mutex_init(&gPriorityLock, NULL);
        (void)pthread_attr_init(&threadAttr);

        /*
         * Reference run. Run each thread sequentially i.e no pre-emption to
         * establish a reference baseline
         */
        printf("========= RUNNING BASELINE WITH NO PREEMPTION  =========\n", (i+1));
        for(int32_t j = 0; j < numThreads; j++)
        {
            ModelThreadArgs[j].runConfig.isPreemptionEnabled = false;
            ModelThreadArgs[j].runConfig.duration = 1; // Run for 1s
            ModelThreadArgs[j].runConfig.outputDir = "outputs/test_" + std::to_string(i+1) + "/" + test[j].model + "/no_preemption";

            (void)pthread_barrier_init(&gBarrier, &barrierAttr, 1);
            pthread_create(&threadId[j], NULL, threadRunner, &ModelThreadArgs[j]);
            pthread_join(threadId[j], NULL);
            pthread_barrier_destroy(&gBarrier);
        }

        testResult.testId = (i+1);
        testResult.n1Priority = test[0].priority;
        testResult.n1Delay = test[0].maxPreemptDelay;
        testResult.n1InvokeTime = ModelThreadArgs[0].runConfig.avgInvokeTime;
        testResult.n2Priority = test[1].priority;
        testResult.n2Delay = test[1].maxPreemptDelay;
        testResult.n2InvokeTime = ModelThreadArgs[1].runConfig.avgInvokeTime;
        referenceTestResult.push_back(testResult);

        /*
         * Parallel run. Run each thread parallely with pre-emption
         */
        printf("=============== RUNNING WITH PREEMPTION  ===============\n", (i+1));

        (void)pthread_barrier_init(&gBarrier, &barrierAttr, numThreads);

        for(int32_t j = 0; j < numThreads; j++)
        {
            ModelThreadArgs[j].runConfig.isPreemptionEnabled = true;
            ModelThreadArgs[j].runConfig.duration = 10; // Run for 10s
            ModelThreadArgs[j].runConfig.outputDir = "outputs/test_" + std::to_string(i+1) + "/" + test[j].model + "/preemption";
            pthread_create(&threadId[j], NULL, threadRunner, &ModelThreadArgs[j]);
        }
        for(int32_t j = 0; j < numThreads; j++)
        {
            pthread_join(threadId[j], NULL);
        }
        pthread_barrierattr_destroy(&barrierAttr);
        pthread_barrier_destroy(&gBarrier);

        pthread_mutex_destroy(&gPriorityLock);

        testResult.testId = (i+1);
        testResult.n1Priority = test[0].priority;
        testResult.n1Delay = test[0].maxPreemptDelay;
        testResult.n1InvokeTime = ModelThreadArgs[0].runConfig.avgInvokeTime;
        testResult.n2Priority = test[1].priority;
        testResult.n2Delay = test[1].maxPreemptDelay;
        testResult.n2InvokeTime = ModelThreadArgs[1].runConfig.avgInvokeTime;
        preemptionTestResult.push_back(testResult);
    }

    // Print the reference results table
    if (referenceTestResult.size() > 0)
    {
        printf("\nTEST PERFORMANCE WITHOUT PARALLEL PROCESSING");
        printResultsTable(referenceTestResult);
    }

    if (preemptionTestResult.size() > 0)
    {
        printf("\nTEST PERFORMANCE WITH PARALLEL PROCESSING");
        printResultsTable(preemptionTestResult);
    }

    return 0;
}
