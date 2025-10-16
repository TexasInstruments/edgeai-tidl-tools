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

#include "tflitert_wrapper.h"

/**
 * @brief Maximum number of inference options that can be passed to the TIDL delegate
 */
#define MAX_INFER_OPTIONS (10u)

namespace tflitert_wrapper
{
    /**
     * @brief Function pointer type for error handling callbacks
     * 
     * This type defines a function signature for error handlers that can be passed to
     * the TIDL delegate creation function. The handler receives error messages as strings.
     * 
     * @param error_message Error message string to be handled
     */
    typedef void (*ErrorHandler)(const char* error_message);
    
    /**
     * @brief Function pointer type for creating a TFLite delegate
     * 
     * This type defines the signature for the delegate creation function that is
     * dynamically loaded from the TIDL delegate library. The function takes configuration
     * options as key-value pairs and an error reporting callback.
     * 
     * @param keys Array of option keys
     * @param values Array of option values corresponding to keys
     * @param num_options Number of options in the arrays
     * @param report_error Function pointer to error reporting callback
     * @return TfLiteDelegate* Pointer to the created delegate
     */
    typedef TfLiteDelegate* (*Create_delegate)(char** keys,
                                               char** values,
                                               size_t num_options,
                                               void (*report_error)(const char *));
    /**
     * @brief Constructor for the TFLiteRT class
     * 
     * Initializes the TFLiteRT object with the specified model path and TIDL offload setting.
     * This constructor only stores the parameters and doesn't perform any resource-intensive
     * operations. The actual model loading and initialization happens in createInfer().
     * 
     * @param modelPath Path to the TensorFlow Lite model file (.tflite)
     * @param tidlOffload Flag to enable TIDL hardware acceleration
     */
    TFLiteRT::TFLiteRT(std::string modelPath, bool tidlOffload):
        m_modelPath(modelPath),
        m_tidlOffload(tidlOffload)
    {
    }

    /**
     * @brief Destructor for the TFLiteRT class
     * 
     * The interpreter and model objects are automatically cleaned up by their
     * respective unique_ptr destructors, so no explicit cleanup is needed here.
     */
    TFLiteRT::~TFLiteRT()
    {
    }

    /**
     * @brief Creates and initializes the inference engine
     * 
     * Sets up the TensorFlow Lite Runtime environment with the specified options and prepares
     * the model for inference. This method performs several critical steps:
     * 
     * 1. Loads the model from the file specified in the constructor
     * 2. Creates an interpreter using the built-in operation resolver
     * 3. If TIDL acceleration is enabled:
     *    - Loads the TIDL delegate library dynamically
     *    - Configures the delegate with the provided options
     *    - Modifies the interpreter graph to use the delegate
     * 4. Allocates memory for all tensors in the model
     * 5. Populates input and output tensor information for later use
     * 
     * @param options Map of configuration options for the inference engine
     *                These are passed directly to the TIDL delegate when enabled
     * @return int32_t Status code (0 for success, negative for failure)
     * @throws std::runtime_error If any step in the initialization process fails
     */
    int32_t TFLiteRT::createInfer(std::map<std::string, std::string> &options)
    {
        int32_t status = 0;
        void *lib = NULL;
        Create_delegate createPlugin;
        TfLiteDelegate *delegatePtr;

        m_model = tflite::FlatBufferModel::BuildFromFile(m_modelPath.c_str());
        if (m_model == nullptr)
        {
            status = -1;
            throw std::runtime_error("TFLiteRT loading model failed");
        }

        tflite::InterpreterBuilder(*m_model, m_resolver)(&m_interpreter);
        if (m_interpreter == nullptr)
        {
            status = -1;
            throw std::runtime_error("TFLiteRT could not build interpreter from loaded model");
        }
    
        if (m_tidlOffload)
        {
            lib = dlopen("libtidl_tfl_delegate.so", RTLD_NOW);
            if(lib == NULL)
            {
                status = -1;
                throw std::runtime_error("TFLiteRT could not open libtidl_tfl_delegate.so");
            }

            createPlugin = (Create_delegate)dlsym(lib, "tflite_plugin_create_delegate");
            if (createPlugin == NULL)
            {
                status = -1;
                throw std::runtime_error("TFLiteRT tflite_plugin_create_delegate lookup in libtidl_tfl_delegate.so failed");
            }

            if (options.find("artifacts_folder") == options.end())
            {
                status = -1;
                throw std::runtime_error("'artifacts_folder' is not provided.");
            }

            const char *keys[MAX_INFER_OPTIONS]; 
            const char *values[MAX_INFER_OPTIONS];
            int32_t numOptions = 0;            
            for (const auto &option : options)
            {
                keys[numOptions] = option.first.c_str();
                values[numOptions] = option.second.c_str();
                numOptions++;
                if (numOptions >= MAX_INFER_OPTIONS)
                {
                    printf("[WARN] No. of infer options exceeds max allowed infer options (%d)", MAX_INFER_OPTIONS);
                    break;
                }
            }

            delegatePtr = createPlugin((char **)keys, (char **)values, numOptions, NULL);

            m_interpreter->ModifyGraphWithDelegate(delegatePtr);
        }

        
        if (m_interpreter->AllocateTensors() != kTfLiteOk)
        {
            status = -1;
            throw std::runtime_error("TFLiteRT Failed to allocate tensors");
        }

        status = populateInputInfo();
        if (status != 0)
        {
            status = -1;
            throw std::runtime_error("TFLiteRT populating input info failed");
        }

        status = populateOutputInfo();
        if (status != 0)
        {
            status = -1;
            throw std::runtime_error("TFLiteRT populating output info failed");
        }

        return status;
    }

    /**
     * @brief Populates information about the model's input tensors
     * 
     * Queries the TensorFlow Lite interpreter for details about input tensors and populates
     * the corresponding member variables. For each input tensor, it retrieves:
     * - Name (identifier for the tensor)
     * - Size (total memory size in bytes)
     * - Number of dimensions (rank of the tensor)
     * - Element size (size of each element in bytes)
     * - Shape (size of each dimension)
     * - Total number of elements
     * 
     * This information is stored in the m_inputs vector for later use during inference
     * and for providing tensor details to the user.
     * 
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t TFLiteRT::populateInputInfo()
    {
        int32_t status = 0;

        /* Get the inputs. */
        const std::vector<int> inputs  = m_interpreter->inputs();
        m_numInputs = inputs.size();

        /* Initialize member variables */
        m_inputs.assign(m_numInputs, DlTensor());

        for (int32_t i = 0; i < m_numInputs; i++)
        {
            DlTensor *info = &m_inputs[i];
            const TfLiteTensor *tensor = m_interpreter->input_tensor(i);
            TfLiteType type = TfLiteTensorType(tensor);

            info->name     = TfLiteTensorName(tensor);
            info->allocSize     = TfLiteTensorByteSize(tensor);
            info->numDim   = TfLiteTensorNumDims(tensor);
            info->elemSize = Tflite2TidlType(type, info->type, info->typeName);
            info->numElem  = 1;
            info->shape.assign(info->numDim, 0);
            for (int32_t j = 0; j < info->numDim; j++)
            {
                info->shape[j] = TfLiteTensorDim(tensor, j);
                info->numElem *= info->shape[j];
            }

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
    const std::vector<DlTensor>* TFLiteRT::getInputDetails()
    {
        return &m_inputs;
    }

    /**
     * @brief Populates information about the model's output tensors
     * 
     * Queries the TensorFlow Lite interpreter for details about output tensors and populates
     * the corresponding member variables. For each output tensor, it retrieves:
     * - Name (identifier for the tensor)
     * - Size (total memory size in bytes)
     * - Number of dimensions (rank of the tensor)
     * - Element size (size of each element in bytes)
     * - Shape (size of each dimension)
     * - Total number of elements
     * 
     * This information is stored in the m_outputs vector for later use during inference
     * and for providing tensor details to the user.
     * 
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t TFLiteRT::populateOutputInfo()
    {
        int32_t status = 0;

        /* Get the inputs. */
        const std::vector<int> outputs  = m_interpreter->outputs();
        m_numOutputs = outputs.size();

        /* Initialize member variables */
        m_outputs.assign(m_numOutputs, DlTensor());

        for (int32_t i = 0; i < m_numOutputs; i++)
        {
            DlTensor *info = &m_outputs[i];
            const TfLiteTensor *tensor = m_interpreter->output_tensor(i);
            TfLiteType type = TfLiteTensorType(tensor);

            info->name     = TfLiteTensorName(tensor);
            info->allocSize     = TfLiteTensorByteSize(tensor);
            info->numDim   = TfLiteTensorNumDims(tensor);
            info->elemSize = Tflite2TidlType(type, info->type, info->typeName);
            info->numElem  = 1;
            info->shape.assign(info->numDim, 0);
            for (int32_t j = 0; j < info->numDim; j++)
            {
                info->shape[j] = TfLiteTensorDim(tensor, j);
                info->numElem *= info->shape[j];
            }

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
    const std::vector<DlTensor>* TFLiteRT::getOutputDetails()
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
     *          Currently not supported for TFLite C++ runtime
     */
    const std::map<std::string, std::pair<float, std::string>> TFLiteRT::getPerformance()
    {
        std::map<std::string, std::pair<float, std::string>> data;
        return data;
    }

    /**
     * @brief Prints detailed information about the model and its tensors
     * 
     * Outputs model path, input/output tensor counts, and detailed information
     * about each tensor including name, type, shape, and size. This is useful for
     * debugging and understanding the model structure. The method calls the dumpInfo()
     * method of each DlTensor to print its detailed properties.
     * 
     * This function is particularly useful when the verbose mode is enabled in the
     * application, allowing users to inspect the model structure and tensor details
     * before running inference.
     */
    void TFLiteRT::dumpInfo()
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
     * 1. Sets custom memory allocations for each input tensor, linking the provided
     *    DlTensor data pointers to the TensorFlow Lite tensors
     * 2. Sets custom memory allocations for each output tensor, allowing direct
     *    writing of results to the provided DlTensor data pointers
     * 3. Invokes the interpreter to perform the actual inference computation
     * 4. Returns the status of the operation
     * 
     * Using custom allocations avoids unnecessary memory copies between the application
     * and the TensorFlow Lite runtime, improving performance.
     * 
     * @param inputs Vector of input tensors containing the data for inference
     * @param outputs Vector of output tensors to store the inference results
     * @return int32_t Status code (0 for success, negative for failure)
     */
    int32_t TFLiteRT::runInfer(const std::vector<DlTensor *> &inputs, std::vector<DlTensor *> &outputs)
    {
        int32_t status = 0;
        TfLiteStatus tfStatus;
        
        for (uint32_t i = 0; i < m_numInputs; i++)
        {
            int tensor_idx = m_interpreter->inputs()[i];
            const TfLiteTensor *tensor = m_interpreter->input_tensor(i);
            m_interpreter->SetCustomAllocationForTensor(tensor_idx, {inputs[i]->data, TfLiteTensorByteSize(tensor)});
        }
        for (uint32_t i = 0; i < m_numOutputs; i++)
        {
            int tensor_idx = m_interpreter->outputs()[i];
            const TfLiteTensor *tensor = m_interpreter->output_tensor(i);
            m_interpreter->SetCustomAllocationForTensor(tensor_idx, {outputs[i]->data, TfLiteTensorByteSize(tensor)});
        }

        tfStatus = m_interpreter->Invoke();
        if (tfStatus != kTfLiteOk)
        {
            status = -1;
        }

        return status;
    }

    /**
     * @brief Converts TFLITE tensor element data type to TIDL type
     * 
     * Maps TensorFlow Lite data types to their corresponding TIDL type identifiers and
     * returns the size in bytes of the data type. This mapping is essential for proper
     * data handling between TensorFlow Lite and TIDL acceleration hardware.
     *
     * 
     * @param tfliteType The TFLITE tensor element data type to convert
     * @param tidlType Reference to store the corresponding TIDL type
     * @param typeName Reference to store the corresponding type name
     * @return int32_t Size in bytes of the data type
     * 
     * @note Future improvement: Replace hardcoded TIDL type integers with proper enum values
     *       from a TIDL header to improve code maintainability and readability.
     */
    int32_t TFLiteRT::Tflite2TidlType(const TfLiteType &tfliteType, int32_t &tidlType, std::string &typeName)
    {
        int32_t size;

        switch (tfliteType)
        {
            case kTfLiteInt8:
                tidlType = TIDL_SignedChar;
                size = sizeof(int8_t);
                typeName = "int8_t";
                break;

            case kTfLiteUInt8:
                tidlType = TIDL_UnsignedChar;
                size = sizeof(uint8_t);
                typeName = "uint8_t";
                break;

            case kTfLiteInt16:
                tidlType = TIDL_SignedShort;
                size = sizeof(int16_t);
                typeName = "int16_t";
                break;

            case kTfLiteUInt16:
                tidlType = TIDL_UnsignedShort;
                size = sizeof(uint16_t);
                typeName = "uint16_t";
                break;

            case kTfLiteInt32:
                tidlType = TIDL_SignedWord;
                size = sizeof(int32_t);
                typeName = "int32_t";
                break;

            case kTfLiteUInt32:
                tidlType = TIDL_UnsignedWord;
                size = sizeof(uint32_t);
                typeName = "uint32_t";
                break;

            case kTfLiteInt64:
                tidlType = TIDL_SignedDoubleWord;
                size = sizeof(int64_t);
                typeName = "int64_t";
                break;
            
            case kTfLiteUInt64:
                tidlType = TIDL_UnsignedDoubleWord;
                size = sizeof(uint64_t);
                typeName = "uint64_t";
                break;

            case kTfLiteFloat32:
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
     * @brief Converts TIDL type to TFLITE tensor element data type
     * 
     * Maps TIDL type identifiers to their corresponding TensorFlow Lite data types and
     * returns the size in bytes of the data type. This is the inverse operation of
     * Tflite2TidlType and is used when interfacing between TIDL and TensorFlow Lite.
     * 
     * @param tidlType The TIDL type to convert
     * @param tfliteType Reference to store the corresponding TensorFlow Lite tensor element data type
     * @return int32_t Size in bytes of the data type
     */
    int32_t TFLiteRT::Tidl2TfliteType(const int32_t &tidlType, TfLiteType &tfliteType)
    {
        int32_t size;

        switch (tidlType)
        {
            case TIDL_SignedChar:
                tfliteType = kTfLiteInt8;
                size = sizeof(int8_t);
                break;

            case TIDL_UnsignedChar:
                tfliteType = kTfLiteUInt8;
                size = sizeof(uint8_t);
                break;

            case TIDL_SignedShort:
                tfliteType = kTfLiteInt16;
                size = sizeof(int16_t);
                break;

            case TIDL_UnsignedShort:
                tfliteType = kTfLiteUInt16;
                size = sizeof(uint16_t);
                break;

            case TIDL_SignedWord:
                tfliteType = kTfLiteInt32;
                size = sizeof(int32_t);
                break;

            case TIDL_UnsignedWord:
                tfliteType = kTfLiteUInt32;
                size = sizeof(uint32_t);
                break;

            case TIDL_SignedDoubleWord:
                tfliteType = kTfLiteInt64;
                size = sizeof(int64_t);
                break;
            
            case TIDL_UnsignedDoubleWord:
                tfliteType = kTfLiteUInt64;
                size = sizeof(uint64_t);
                break;

            case TIDL_SinglePrecFloat:
                tfliteType = kTfLiteFloat32;
                size = sizeof(float);
                break;

            default:
                size = 0;
        }

        return size;
    }

} // namespace tflitert_wrapper
