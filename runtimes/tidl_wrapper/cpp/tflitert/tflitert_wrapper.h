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

#ifndef TFLiteRT_WRAPPER_H_
#define TFLiteRT_WRAPPER_H_

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <dlfcn.h>

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/util.h>


#include "itidl_rt.h"
#include "itidl_io.h"
#include "dltensor.h"

using namespace dl_tensor;

namespace tflitert_wrapper
{
    /**
     * @class TFLiteRT
     * @brief A wrapper class for TensorFlow Lite Runtime that provides an interface for model inference
     * 
     * This class encapsulates the functionality required to load, initialize, and run
     * inference on TensorFlow Lite models. It supports both CPU execution and TIDL 
     * (Texas Instruments Deep Learning) hardware acceleration for optimized performance 
     * on TI devices. The class handles model loading, tensor allocation, delegate 
     * configuration, and inference execution with proper error handling.
     * 
     * The TIDL acceleration is implemented through a delegate mechanism that offloads
     * compatible operations to specialized hardware, significantly improving inference
     * performance while maintaining accuracy.
     */
    class TFLiteRT
    {
        public:
            /**
             * @brief Constructor for the TFLiteRT class
             * 
             * Initializes a new instance with the specified model path and acceleration setting.
             * Does not load the model immediately - call createInfer() to load and prepare the model.
             * 
             * @param modelPath Path to the TensorFlow Lite model file (.tflite)
             * @param tidlOffload Flag to enable TIDL hardware acceleration (default: true)
             *                    When true, compatible operations are offloaded to TIDL hardware
             */
            TFLiteRT(std::string modelPath, bool tidlOffload = true);

            /**
             * @brief Destructor for the TFLiteRT class
             * 
             * Cleans up resources allocated during the lifetime of the object.
             * The interpreter and model objects are automatically cleaned up by their
             * respective unique_ptr destructors, so no explicit cleanup is needed here.
             */
            ~TFLiteRT();

            /**
             * @brief Creates and initializes the inference engine
             * 
             * Sets up the TensorFlow Lite Runtime session with the specified options and prepares
             * the model for inference. This includes:
             * 1. Loading the model from the specified path
             * 2. Building an interpreter with the appropriate operation resolver
             * 3. Configuring TIDL acceleration delegate if enabled
             * 4. Allocating tensors for input and output
             * 5. Populating tensor information for later use
             * 
             * @param options Map of configuration options for the inference engine
             *                These options are passed to the TIDL delegate when enabled
             * @return int32_t Status code (0 for success, negative for failure)
             * @throws std::runtime_error If model loading, interpreter creation, or tensor allocation fails
             */
            int32_t createInfer(std::map<std::string, std::string> &options);

            /**
             * @brief Runs inference on the loaded model
             * 
             * Executes the model with the provided input tensors and populates the output tensors
             * with the inference results. This method:
             * 1. Sets custom memory allocations for input and output tensors
             * 2. Invokes the TensorFlow Lite interpreter to perform the actual inference
             * 3. Returns the status of the operation
             * 
             * Using custom allocations avoids unnecessary memory copies between the application
             * and the TensorFlow Lite runtime, improving performance.
             * 
             * @param inputs Vector of input tensors containing the data for inference
             *               The number and order must match the model's expected inputs
             * @param outputs Vector of output tensors to store the inference results
             *                The number and order must match the model's outputs
             * @return int32_t Status code (0 for success, negative for failure)
             */
            int32_t runInfer(const std::vector<DlTensor *> &inputs, std::vector<DlTensor *> &outputs);

            /**
             * @brief Gets details about the model's input tensors
             * 
             * Returns information about all input tensors including their names,
             * shapes, data types, and memory requirements.
             * 
             * @return const std::vector<DlTensor>* Pointer to vector of input tensor details
             */
            const std::vector<DlTensor>* getInputDetails();

            /**
             * @brief Gets details about the model's output tensors
             * 
             * Returns information about all output tensors including their names,
             * shapes, data types, and memory requirements.
             * 
             * @return const std::vector<DlTensor>* Pointer to vector of output tensor details
             */
            const std::vector<DlTensor>* getOutputDetails();

            /**
             * @brief Gets the performance data
             *
             * @return  const std::map<std::string, std::pair<float, std::string>>
             *          Map containing performance data where key is performance
             *          metric, and values are (data, unit) 
             * 
             *          Currently not supported for TFLite C++ runtime
             */
            const std::map<std::string, std::pair<float, std::string>> getPerformance();

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

            void dumpInfo();

            /**
             * @brief Converts TFLITE tensor element data type to TIDL type
             * 
             * Maps TensorFlow Lite data types to their corresponding TIDL type identifiers and
             * returns the size in bytes of the data type. This mapping is essential for proper
             * data handling between TensorFlow Lite and TIDL acceleration hardware.
             * 
             * @param tfliteType The TFLITE tensor element data type to convert
             * @param tidlType Reference to store the corresponding TIDL type
             * @param typeName Reference to store the corresponding type name
             * @return int32_t Size in bytes of the data type
             * 
             * @note Future improvement: Replace hardcoded TIDL type integers with proper enum values
             *       from a TIDL header to improve code maintainability and readability.
             */
            static int32_t Tflite2TidlType(const TfLiteType &tfliteType, int32_t &tidlType, std::string &typeName);

            /**
             * @brief Converts TIDL type to TFLITE tensor element data type
             * 
             * Maps TIDL type identifiers to their corresponding TensorFlow Lite data types and
             * returns the size in bytes of the data type. This is the inverse operation of
             * Tflite2TidlType and is used when interfacing between TIDL and TensorFlow Lite.
             * 
             * @param tidlType The TIDL type to convert
             * @param tfliteType Reference to store the corresponding TFLITE tensor element data type
             * @return int32_t Size in bytes of the data type
             */
            static int32_t Tidl2TfliteType(const int32_t &tidlType, TfLiteType &tfliteType);
        
        private:
            /**
             * @brief Populates information about the model's input tensors
             * 
             * Queries the TensorFlow Lite interpreter for details about input tensors
             * and populates the corresponding member variables. For each input tensor,
             * it extracts the name, shape, dimensions, data type, and calculates the
             * total size and number of elements.
             * 
             * @return int32_t Status code (0 for success, negative for failure)
             */
            int32_t populateInputInfo();

            /**
             * @brief Populates information about the model's output tensors
             * 
             * Queries the TensorFlow Lite interpreter for details about output tensors
             * and populates the corresponding member variables. For each output tensor,
             * it extracts the name, shape, dimensions, data type, and calculates the
             * total size and number of elements.
             * 
             * @return int32_t Status code (0 for success, negative for failure)
             */
            int32_t populateOutputInfo();

        private:
            /** @brief Path to the TFLITE model file */
            std::string                                 m_modelPath;
    
            /** @brief Flag indicating whether TIDL hardware acceleration is enabled */
            bool                                        m_tidlOffload;

            /** @brief A pointer to the model representation in memory */
            std::unique_ptr<tflite::FlatBufferModel>    m_model;

            /** @brief Resolver for built-in TFLite operations that maps op codes to implementations */
            tflite::ops::builtin::BuiltinOpResolver     m_resolver;

            /** @brief A pointer to the model interpreter that executes the loaded model */
            std::unique_ptr<tflite::Interpreter>        m_interpreter;

            /** @brief Number of input tensors in the model */
            int32_t                                     m_numInputs;

            /** @brief Vector of input tensor details */
            std::vector<DlTensor>                       m_inputs;

            /** @brief Number of output tensors in the model */
            int32_t                                     m_numOutputs;

            /** @brief Vector of output tensor details */
            std::vector<DlTensor>                       m_outputs;
    };
} // namespace tflitert_wrapper

#endif /* TFLiteRT_WRAPPER_H_ */
