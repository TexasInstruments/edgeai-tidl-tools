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

#include "basic_example.h"
#include "argsparser.h"
#include "utils.h"
#include "datasetloader/dataset_loader.h"

#define DEFAULT_MEM_ALIGNMENT (64)

namespace fs = std::filesystem;

struct RunConfig
{
    std::string                         name;

    std::string                         path;

    std::string                         runtime;

    std::vector<std::string>            inputs;

    int32_t                             numFrames;

    std::map<std::string, std::string>  inferOptions;
};

/**
 * @class BasicExample
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
class BasicExample
{
public:
    bool                                m_disableTIDLOffload;
    bool                                m_verbose;
    std::vector<std::string>            m_modelsFilter;
    std::vector<std::string>            m_runtimesFilter;
    std::vector<RunConfig>              m_runConfigs;
    std::string                         m_sourceDir;
    std::string                         m_configFile;
    std::string                         m_artifactsBaseDir;
    std::string                         m_outputsBaseDir;
    std::map<std::string, std::string>  m_commonInferOptions;

public:

    /**
     * @brief Constructor for the BasicExample class
     * 
     * Initializes the BasicExample object with the provided parameters,
     * sets up paths for configuration, artifacts, and outputs,
     * and parses and validates the configuration.
     * 
     * @param disableTIDLOffload Flag to disable TIDL offload
     * @param verbose Flag to enable verbose output
     * @param modelsFilter List of model names to filter (only run these models)
     * @param runtimesFilter List of runtime types to filter (only use these runtimes)
     * @throws std::runtime_error If configuration parsing or validation fails
     */
    BasicExample(bool disableTIDLOffload, bool verbose, std::string configFilePath = "", std::vector<std::string> modelsFilter = {}, std::vector<std::string> runtimesFilter = {})
    {

        m_disableTIDLOffload = disableTIDLOffload;
        m_verbose = verbose;
        m_modelsFilter = modelsFilter;
        m_runtimesFilter = runtimesFilter;
        m_runConfigs = {};
        
        // Determine the source directory and config file path
        if (configFilePath != "")
        {
            m_configFile = configFilePath;
            m_sourceDir = fs::path(m_configFile).parent_path().string();
        }
        else
        {
            m_sourceDir = fs::path(__FILE__).parent_path().string();
            m_configFile = (m_sourceDir / fs::path("config.yaml")).string();
        }
        m_artifactsBaseDir = (m_sourceDir / fs::path("../../model-artifacts")).string();
        m_outputsBaseDir = (m_sourceDir / fs::path("outputs")).string();
        
        if(parseConfig() != 0)
        {
            throw std::runtime_error("Could not parse config");
        }
        if(validateConfig() != 0)
        {
            throw std::runtime_error("Could not validate config");
        }
    }

private:

    /**
     * @brief Parses the configuration from a YAML file
     * 
     * This method loads the configuration from the YAML file specified by m_configFile,
     * extracts model configurations and common inference options, and applies any
     * filters specified during construction.
     * 
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t parseConfig()
    {
        if (!fs::exists(m_configFile))
        {
            printf("[ERROR] %s does not exits\n", m_configFile.c_str());
            return -1;
        }

        const YAML::Node config = YAML::LoadFile(m_configFile);
        const YAML::Node &models = config["models"];
        const YAML::Node &commonInferOptions = config["infer_options"];

        if (!models)
        {
            printf("[ERROR] %s does not have any models to run\n", m_configFile.c_str());
            return -1;
        }

        for (YAML::const_iterator it = models.begin(); it != models.end(); ++it)
        {
            RunConfig runConfig;
            std::vector<std::string> tempInputs;
            int32_t i;
        
            const YAML::Node &path = it->second["path"];
            const YAML::Node &inputs = it->second["inputs"];
            const YAML::Node &runtime = it->second["runtime"];
            const YAML::Node &numFrames = it->second["num_frames"];
            const YAML::Node &inferOptions = it->second["infer_options"];

            runConfig.name = it->first.as<std::string>();

            if (m_modelsFilter.size() > 0 &&
                std::find(m_modelsFilter.begin(), m_modelsFilter.end(), runConfig.name) == m_modelsFilter.end())
            {
                continue;
            }

            if (path)
            {
                runConfig.path = path.as<std::string>();
            }
            else
            {
                runConfig.path = "";
            }

            if (!inputs)
            {
                printf("[ERROR] 'inputs' is missing from %s\n", runConfig.name.c_str());
                return -1;
            }

            if (inputs.IsScalar())
            {
                tempInputs = splitString(inputs.as<std::string>(), ' ');
            }
            else
            {
                tempInputs = inputs.as<std::vector<std::string>>();
            }

            for (auto &input : tempInputs)
            {
                if(fs::path(input).is_relative())
                {
                    input = (fs::path(m_configFile).parent_path() / fs::path(input)).string();
                }
                if (fs::is_directory(input))
                {
                    try
                    {
                        for (const auto &entry : fs::directory_iterator(input))
                        {
                            if (fs::is_regular_file(entry.status()))
                            {
                                fs::path filepath = entry.path();
                                std::string filename = filepath.filename().string();
                                if (endsWith(filename, ".bin") || endsWith(filename, ".npz"))
                                {
                                    runConfig.inputs.push_back(filepath.string());
                                }
                            }
                        }
                    }
                    catch (const fs::filesystem_error& e)
                    {
                        printf("[WARN] Could not iterate over %s directory. Skipping it", input.c_str());
                    }
                }
                else
                {
                    runConfig.inputs.push_back(input);
                }
            }

            if (runtime)
            {
                runConfig.runtime = runtime.as<std::string>();
                std::transform(runConfig.runtime.begin(), runConfig.runtime.end(), runConfig.runtime.begin(),
                               [](unsigned char c){ return std::tolower(c); });
            }
            else
            {
                printf("[ERROR] 'runtime' is missing from %s\n", runConfig.name.c_str());
                return -1;
            }

            if(runConfig.runtime != "tidlrt" && runConfig.path == "")
            {
                printf("[ERROR] 'path' is missing from %s\n", runConfig.name.c_str());
                return -1;
            }

            if(numFrames)
            {
                runConfig.numFrames = numFrames.as<int32_t>();
            }
            else
            {
                runConfig.numFrames = (int32_t)(runConfig.inputs.size());
            }

            if(inferOptions)
            {
                runConfig.inferOptions = inferOptions.as<std::map<std::string, std::string>>();
            }
            
            if (m_runtimesFilter.size() > 0 &&
                std::find(m_runtimesFilter.begin(), m_runtimesFilter.end(), runConfig.runtime) == m_runtimesFilter.end())
            {
                continue;
            }

            m_runConfigs.push_back(runConfig);
        }

        if (commonInferOptions)
        {
            m_commonInferOptions = commonInferOptions.as<std::map<std::string, std::string>>();
        }

        return 0;
    }

    /**
     * @brief Validates the parsed configuration
     * 
     * This method checks that:
     * - All model paths exist and are valid files
     * - Runtime types are supported
     * - Number of frames matches number of inputs
     * - Resolves relative paths to absolute paths
     * 
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t validateConfig()
    {
        for (auto &runConfig : m_runConfigs)
        {
            if (runConfig.runtime != "onnxrt" && runConfig.runtime != "tflitert" && runConfig.runtime != "tidlrt" && runConfig.runtime != "tvmrt")
            {
                printf("[ERROR][%s] Invalid runtime %s\n", runConfig.name.c_str(), runConfig.runtime.c_str());
                return -1;
            }

            if (runConfig.runtime != "tidlrt")
            {
                if(fs::path(runConfig.path).is_relative())
                {
                    runConfig.path = (fs::path(m_configFile).parent_path() / fs::path(runConfig.path)).string();
                }

                if (!fs::exists(runConfig.path) || !fs::is_regular_file(runConfig.path))
                {
                    printf("[ERROR][%s] %s does not exist or is not a file\n", runConfig.name.c_str(), runConfig.path.c_str());
                    return -1;
                }
            }

            for (auto &input : runConfig.inputs)
            {
                if (!fs::exists(input) || !fs::is_regular_file(input))
                {
                    printf("[ERROR][%s] %s does not exist or is not a file\n", runConfig.name.c_str(), input.c_str());
                    return -1;
                }
                if (!endsWith(input, ".bin") && !endsWith(input, ".npz"))
                {
                    printf("[ERROR][%s] %s is not a *.bin or *.npz file\n", runConfig.name.c_str(), input.c_str());
                    return -1;
                }
            }

            if (runConfig.numFrames != (int32_t)runConfig.inputs.size())
            {
                runConfig.numFrames = std::min(runConfig.numFrames, (int32_t)runConfig.inputs.size());
                runConfig.inputs.erase(runConfig.inputs.begin() + runConfig.numFrames, runConfig.inputs.end());
                printf("[WARN][%s] No. of inputs and frames do not match. Running only %d frames\n", runConfig.name.c_str(), runConfig.numFrames);   
            }

            if( (runConfig.runtime == "tvmrt") && m_disableTIDLOffload)
            {
                printf("[ERROR][%s] Flag 'disable_tidl_offload' is valid only for compilation. First compile model with this flag on PC and then run inference without this flag\n", runConfig.name.c_str());
                return -1;
            }
        }
        return 0;
    }

public:

    /**
     * @brief Main method to run inference for all configured models
     * 
     * This method iterates through all configured models, sets up the appropriate
     * runtime session based on the model type, and runs inference using the
     * runInferenceWithSession template method.
     * 
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t run()
    {
        int32_t status = 0;
        for (auto &runConfig : m_runConfigs)
        {
            std::string modelName = runConfig.name;
            std::string artifactsDir;

            artifactsDir = (fs::path(m_artifactsBaseDir) / fs::path(modelName) / fs::path("artifacts")).string();
            if (!fs::exists(artifactsDir) || !fs::is_directory(artifactsDir))
            {
                printf("[ERROR][%s] artifacts folder %s does not exist or is not a directory\n", modelName.c_str(), artifactsDir.c_str());
                status = -1;
                break;
            }

            runConfig.inferOptions["artifacts_folder"] = artifactsDir;
            for (const auto &options : m_commonInferOptions)
            {
                if (runConfig.inferOptions.find(options.first) == runConfig.inferOptions.end())
                {
                    runConfig.inferOptions[options.first] = options.second;
                }
            }
            
            printf("\nRunning %s with %s runtime...\n", modelName.c_str(), runConfig.runtime.c_str());

            if (runConfig.runtime == "onnxrt")
            {
                // Create ONNXRT session
                ONNXRT session(runConfig.path, !m_disableTIDLOffload);
                status = runInferenceWithSession(runConfig, session, modelName);
            }
            else if (runConfig.runtime == "tflitert")
            {
                // Create TFLITERT session
                TFLiteRT session(runConfig.path, !m_disableTIDLOffload);
                status = runInferenceWithSession(runConfig, session, modelName);
            }
            else if (runConfig.runtime == "tidlrt")
            {
                // Create TIDLRT session
                TIDLRT session;
                if (m_disableTIDLOffload)
                {
                    printf("[WARN][%s] TIDLRT does not support disabling offload.\n", modelName.c_str());
                }
                status = runInferenceWithSession(runConfig, session, modelName);
            }
            else if (runConfig.runtime == "tvmrt")
            {
                // Create TVMRT session
                TVMRT session(runConfig.path);
                
                /* Figure out if artifacts are compiled with/without TIDL offlaod and set flag for further consumption */
                std::string tempDir = (fs::path(m_artifactsBaseDir) / fs::path(modelName) / fs::path("artifacts") / fs::path("tempDir")).string();
                if(! (fs::exists(tempDir) && fs::is_directory(tempDir)))
                {
                    /* Default is TIDL offload, if 'tempDir' not present in artifacts, this is no TIDL offload compilation */
                    m_disableTIDLOffload = true;
                }
                else
                {
                    m_disableTIDLOffload = false;
                }
                status = runInferenceWithSession(runConfig, session, modelName);
            }
            else
            {
                printf("[ERROR][%s] Invalid runtime %s\n", modelName.c_str(), runConfig.runtime.c_str());
                status = -1;
            }
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
     * @param modelName Name of the model being processed
     * @return int32_t Status code (0 for success, negative for failure)
     */
    template<typename SessionType>
    int32_t runInferenceWithSession(RunConfig &runConfig, SessionType &session, const std::string &modelName)
    {
        int32_t status = 0;
        int32_t i = 0;
        int32_t memoryAlignent = DEFAULT_MEM_ALIGNMENT;
        std::map<std::string, std::pair<float, std::string>> sumPerformance;

        // Create infer session
        try
        {
            status = session.createInfer(runConfig.inferOptions);
        }
        catch (const std::exception& e)
        {
            status = -1;
        }

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

        // Tflite expects tensor to be aligned as dictated by tflite::kDefaultTensorAlignment
        if (runConfig.runtime == "tflitert")
        {
            memoryAlignent = tflite::kDefaultTensorAlignment;
        }

        // Get input details and prepare input tensors
        const std::vector<DlTensor>* inputInfo = session.getInputDetails();
        std::vector<DlTensor> inputs(inputInfo->size(), DlTensor());
        std::vector<DlTensor *> inputsPtr(inputInfo->size(), nullptr);
        status = allocateTensors(inputInfo, inputs, inputsPtr, memoryAlignent);
        if (status != 0)
        {
            printf("[ERROR][%s] Could not prepare input tensors\n", modelName.c_str());
            return -1;
        }

        // Get output details and prepare output tensors
        const std::vector<DlTensor>* outputInfo = session.getOutputDetails();
        std::vector<DlTensor> outputs(outputInfo->size(), DlTensor());
        std::vector<DlTensor *> outputsPtr(outputInfo->size(), nullptr);
        status = allocateTensors(outputInfo, outputs, outputsPtr, memoryAlignent);
        if (status != 0)
        {
            printf("[ERROR][%s] Could not prepare output tensors\n", modelName.c_str());
            freeTensors(inputsPtr);
            return -1;
        }

        // Process each frame
        for (i = 0; i < runConfig.numFrames; i++)
        {
            // Load the inputs
            status = loadInput(runConfig.inputs[i], inputsPtr);
            if (status != 0)
            {
                printf("[ERROR][%s] Could not load input data for frame %d\n", modelName.c_str(), i+1);
                continue;
            }

            // Run inference
            status = session.runInfer(inputsPtr, outputsPtr);
            if (status != 0)
            {
                printf("[ERROR][%s] Could not run inference for frame %d\n", modelName.c_str(), i+1);
                continue;
            }

            // Get performance data if TIDL offload is enabled
            if (!m_disableTIDLOffload)
            {
                auto performance = session.getPerformance();
                for (const auto &[key, data] : performance)
                {
                    const auto &[val, unit] = data;
                    if (sumPerformance.find(key) != sumPerformance.end())
                    {
                        sumPerformance[key].first += val;
                    }
                    else
                    {
                        sumPerformance[key] = {val, unit};
                    }
                }
            }

            // Save output
            status = saveOutput(outputsPtr, modelName, i+1);
            if (status != 0)
            {
                printf("[ERROR][%s] Could not save output for frame %d\n", modelName.c_str(), i+1);
            }
        }

        // Print average performance metrics after processing all frames
        if (!sumPerformance.empty())
        {
            printf("\n%s\n", std::string(80, '=').c_str());
            printf("Average performance metrics for %s across %d frames:\n", modelName.c_str(), runConfig.numFrames);
            printf("%s\n", std::string(80, '-').c_str());
            
            size_t maxKeyLen = 0;
            for (const auto& [key, _] : sumPerformance)
            {
                maxKeyLen = std::max(maxKeyLen, key.length());
            }
            
            for (const auto& [key, data] : sumPerformance)
            {
                float avg = data.first / runConfig.numFrames;
                printf("%-*s: %.2f %s\n", 
                       static_cast<int>(maxKeyLen + 2), 
                       key.c_str(), 
                       avg, 
                       data.second.c_str());
            }
            printf("%s\n\n", std::string(80, '=').c_str());
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
     * This method attempts to allocate memory in shared memory if TIDL offload
     * is enabled. If that fails or if TIDL offload is disabled, it falls back
     * to regular aligned memory allocation.
     * 
     * @param size Size of memory to allocate in bytes
     * @param alignment Memory alignment requirement in bytes (default: 64)
     * @return void* Pointer to the allocated memory, or NULL if allocation failed
     */
    void *allocTensorMem(size_t size, int32_t alignment = 64)
    {
        bool allocated = false;
        void *ptr = NULL;
        
        if (!m_disableTIDLOffload)
        {
            ptr = TIDLRT_allocSharedMem(alignment, (int32_t)size);
            if (ptr == NULL)
            {
                printf("[WARN] Tensor of size %ld could not be allocate in shared memory. Allocating on default heap region\n", size);
                allocated = false;
            }
            else
            {
                allocated = true;
            }
        }

        if(!allocated)
        {
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
     * This method supports loading from binary (.bin) or NumPy (.npz) files.
     * 
     * @param input Path to the input file
     * @param inputsPtr Vector of pointers to tensors where input data should be loaded
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t loadInput(std::string &input, std::vector<DlTensor *> &inputsPtr)
    {
        int32_t status = 0;
        int32_t i = 0;
        int32_t j = 0;
        std::unique_ptr<DatasetLoaderBase> baseLoader;

        if (endsWith(input, ".bin"))
        {   
            if (inputsPtr[i]->padT != 0 || inputsPtr[i]->padB != 0 ||
                inputsPtr[i]->padL != 0 || inputsPtr[i]->padR != 0)
            {
                printf("[ERROR] Invalid input %s. Binary loader currently does not handle padding. Please use npz file.\n", input.c_str());
                status = -1;
                return status;
            }

            baseLoader = DatasetLoader::createLoader("bin", {{"file_path", input}});
            BinLoader *loader = dynamic_cast<BinLoader*>(baseLoader.get());
            for (i = 0; i < inputsPtr.size(); i++)
            {
                // Fill input tensor (only valid portion)
                try
                {
                    loader->load(inputsPtr[i]->data, inputsPtr[i]->validSize);
                }
                catch (const std::exception& e)
                {
                    status = -1;
                    break;
                }
            }
        }
        else if (endsWith(input, ".npz"))
        {
            baseLoader = DatasetLoader::createLoader("npz", {{"file_path", input}});
            NpzLoader *loader = dynamic_cast<NpzLoader*>(baseLoader.get());
            for (i = 0; i < inputsPtr.size(); i++)
            {
                // Fill input tensor (only valid portion)
                try
                {
                    loader->load(inputsPtr[i]->data, inputsPtr[i]->validSize, inputsPtr[i]->padT, inputsPtr[i]->padB, inputsPtr[i]->padL, inputsPtr[i]->padR);
                }
                catch (const std::exception& e)
                {
                    status = -1;
                    break;
                }
            }
        }
        else
        {
            printf("[ERROR] Invalid input %s. Only *.bin or *.npz is allowed\n", input.c_str());
            status = -1;
        }
        return status;
    }

    /**
     * @brief Saves output tensors to binary files
     * 
     * This method creates an output directory structure based on the model name,
     * offload status, and frame number. It then converts each tensor to float32
     * format and saves it as a binary file.
     * 
     * @param tensorPtr Vector of pointers to output tensors
     * @param modelName Name of the model being processed
     * @param frameNo Frame number for the current output
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t saveOutput(std::vector<DlTensor *> &tensorPtr, std::string modelName, int32_t frameNo)
    {
        int32_t status = 0;
        std::string outputDir = fs::path(m_outputsBaseDir) / fs::path(modelName);
        if (!m_disableTIDLOffload)
        {
            outputDir = outputDir / fs::path("offload");
        }
        else
        {
            outputDir = outputDir / fs::path("no_offload");
        }
        outputDir = (outputDir / fs::path("frame_" +std::to_string(frameNo))).string();
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

int main(int argc, char *argv[])
{
    int32_t status = 0;
    bool disableTIDLOffload = false;
    bool verbose = false;
    std::vector<std::string> modelsFilter = {};
    std::vector<std::string> runtimesFilter = {};
    std::string configPath = "";

    /* Parse arguments */
    try
    {
        ArgParser parser;
        parser.addFlag('h', "help", "Display this help message and exit");
        parser.addFlag('d', "disable_tidl_offload", "Disable offload to TIDL");
        parser.addFlag('v', "verbose", "Enable verbose output");
        parser.addArgument('x', "config", "Path to config.yaml file. Default: <executable_dir>/config.yaml", "");
        parser.addArgument('m', "models", "Filter model keys to run from config file. Default: None (run all models)", "", true);
        parser.addArgument('r', "runtimes", "Filter by runtime types. Values: [onnxrt, tflitert, tvmrt, tidlrt]. Default: None (run all runtimes)", "", true);

        // If parse returns false, help was displayed or there was an error
        if (!parser.parse(argc, argv)) {
            return 0;
        }

        /* Get disable tidl offload */
        disableTIDLOffload = parser.getFlag("disable_tidl_offload");
        
        /* Get verbose flag */
        verbose = parser.getFlag("verbose");
        
        /* Get config file path if specified */
        configPath = parser.getValue("config");

        /* Get models filter and remove dumplicated */
        modelsFilter = parser.getMultiValues("models");
        std::sort(modelsFilter.begin(), modelsFilter.end());
        modelsFilter.erase(std::unique(modelsFilter.begin(), modelsFilter.end()), modelsFilter.end());

        /* Get runtimes filter, keep only allowed values, remove duplicates */
        runtimesFilter = parser.getMultiValues("runtimes");
        for (std::string& s : runtimesFilter)
        {
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return std::tolower(c); });
        }
        runtimesFilter.erase(std::remove_if(runtimesFilter.begin(), runtimesFilter.end(), 
        [&](std::string element)
        {
            return (element != "onnxrt" && element != "tflitert" && element != "tidlrt" && element != "tvmrt");
        }), 
        runtimesFilter.end());
        std::sort(runtimesFilter.begin(), runtimesFilter.end());
        runtimesFilter.erase(std::unique(runtimesFilter.begin(), runtimesFilter.end()), runtimesFilter.end());
    }
    catch (const std::exception& e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return -1;
    }

    try
    {
        BasicExample example = BasicExample(disableTIDLOffload, verbose, configPath, modelsFilter, runtimesFilter);
        status = example.run();
    }
    catch(const std::exception& e)
    {
        return -1;
    }

    return 0;
}
