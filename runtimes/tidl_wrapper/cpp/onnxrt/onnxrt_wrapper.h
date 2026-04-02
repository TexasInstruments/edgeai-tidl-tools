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

#ifndef ONNXRT_WRAPPER_H_
#define ONNXRT_WRAPPER_H_

#include <iostream>
#include <map>
#include <vector>
#include <string>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/onnxruntime_session_options_config_keys.h>
#include <onnxruntime/core/providers/tidl/tidl_provider_factory.h>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>

#include "itidl_rt.h"
#include "itidl_io.h"
#include "dltensor.h"

using namespace dl_tensor;

namespace onnxrt_wrapper
{
    /**
     * @class ONNXRT
     * @brief A wrapper class for ONNX Runtime that provides an interface for model inference
     * 
     * This class encapsulates the functionality required to load, initialize, and run
     * inference on ONNX models. It supports both CPU execution and TIDL (Texas Instruments
     * Deep Learning) hardware acceleration for optimized performance on TI devices.
     */
    class ONNXRT
    {
        public:
            /**
             * @brief Constructor for the ONNXRT class
             * 
             * @param modelPath Path to the ONNX model file
             * @param tidlOffload Flag to enable TIDL hardware acceleration (default: true)
             */
            ONNXRT(std::string modelPath, bool tidlOffload = true);

            /**
             * @brief Destructor for the ONNXRT class
             * 
             * Cleans up resources allocated during the lifetime of the object
             */
            ~ONNXRT();

            /**
             * @brief Creates and initializes the inference
             * 
             * Sets up the ONNX Runtime session with the specified options and prepares
             * the model for inference. This includes configuring TIDL acceleration if enabled.
             * 
             * @param options Map of configuration options for the inference
             * @return int32_t Status code (0 for success, negative for failure)
             */
            int32_t createInfer(std::map<std::string, std::string> &options);

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
            int32_t runInfer(const std::vector<DlTensor *> &inputs, std::vector<DlTensor *> &outputs);

            /**
             * @brief Gets details about the model's input tensors
             * 
             * @return const std::vector<DlTensor>* Pointer to vector of input tensor details
             */
            const std::vector<DlTensor>* getInputDetails();

            /**
             * @brief Gets details about the model's output tensors
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
             *          'total_time': Total time taken for run (ms)
             *          'core_time': Total time taken barring the io copy time (ms)
             *          'subgraph_time': Total TIDL Subgraphs processing time (ms)
             *          'read_total': Total DDR Read bytes [X for x86 runs]
             *          'write_total': Total DDR Write bytes [X for x86 runs]
             */
            const std::map<std::string, std::pair<float, std::string>> getPerformance();

            /**
             * @brief Prints detailed information about the model and its tensors
             * 
             * Outputs model path, input/output tensor counts, and detailed information
             * about each tensor including name, type, shape, and size.
             */
            void dumpInfo();

            /**
             * @brief Converts ONNX tensor element data type to TIDL type
             * 
             * @param onnxType The ONNX tensor element data type to convert
             * @param tidlType Reference to store the corresponding TIDL type
             * @param typeName Reference to store the corresponding type name
             * @return int32_t Size in bytes of the data type
             */
            static int32_t Onnx2TidlType(const ONNXTensorElementDataType &onnxType, int32_t &tidlType, std::string &typeName);

            /**
             * @brief Converts TIDL type to ONNX tensor element data type
             * 
             * @param tidlType The TIDL type to convert
             * @param onnxType Reference to store the corresponding ONNX tensor element data type
             * @return int32_t Size in bytes of the data type
             */
            static int32_t Tidl2OnnxType(const int32_t &tidlType, ONNXTensorElementDataType &onnxType);
        
        private:
            /**
             * @brief Populates information about the model's input tensors
             * 
             * Queries the ONNX Runtime session for details about input tensors
             * and populates the corresponding member variables.
             * 
             * @return int32_t Status code (0 for success, negative for failure)
             */
            int32_t populateInputInfo();

            /**
             * @brief Populates information about the model's output tensors
             * 
             * Queries the ONNX Runtime session for details about output tensors
             * and populates the corresponding member variables. Requires a warmup
             * inference run to determine output tensor properties.
             * 
             * @return int32_t Status code (0 for success, negative for failure)
             */
            int32_t populateOutputInfo();

        private:
            /** @brief Path to the ONNX model file */
            std::string                             m_modelPath;
    
            /** @brief Flag indicating whether TIDL hardware acceleration is enabled */
            bool                                    m_tidlOffload;

            /** @brief Pointer to the ONNX Runtime session object */
            Ort::Session                            *m_session;

            /** @brief ONNX Runtime environment with logging configuration */
            Ort::Env                                m_env;

            /** @brief Default memory allocator for ONNX Runtime */
            Ort::AllocatorWithDefaultOptions        m_allocator;

            /** @brief Memory information for tensor allocation */
            Ort::MemoryInfo                         m_memInfo;

            /** @brief Number of input tensors in the model */
            int32_t                                 m_numInputs;

            /** @brief Vector of input tensor names */
            std::vector<const char*>                m_inputNames;

            /** @brief Vector of allocated string pointers for input names (ownership management) */
            std::vector<Ort::AllocatedStringPtr>    m_inputNamesPtr;

            /** @brief Vector of input tensor details */
            std::vector<DlTensor>                   m_inputs;

            /** @brief Vector of input tensor data types in ONNX format */
            std::vector<ONNXTensorElementDataType>  m_inputTypes;

            /** @brief Number of output tensors in the model */
            int32_t                                 m_numOutputs;

            /** @brief Vector of output tensor names */
            std::vector<const char*>                m_outputNames;

            /** @brief Vector of allocated string pointers for output names (ownership management) */
            std::vector<Ort::AllocatedStringPtr>    m_outputNamesPtr;

            /** @brief Vector of output tensor data types in ONNX format */
            std::vector<ONNXTensorElementDataType>  m_outputTypes;

            /** @brief Vector of output tensor details */
            std::vector<DlTensor>                   m_outputs;
    };
} // namespace onnxrt_wrapper

#endif /* ONNXRT_WRAPPER_H_*/
