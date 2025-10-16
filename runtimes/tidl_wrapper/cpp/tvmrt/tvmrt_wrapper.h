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

#ifndef TVMRT_WRAPPER_H_
#define TVMRT_WRAPPER_H_

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <fstream>

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <dlpack/dlpack.h>

#include "itidl_rt.h"
#include "itidl_io.h"
#include "dltensor.h"

using namespace dl_tensor;

namespace tvmrt_wrapper
{
    /**
     * @class TVMRT
     * @brief A wrapper class for TVM Runtime that provides a uniform interface for model inference
     *
     * This class encapsulates the functionality required to load, initialize, and run
     * inference on TVM compiled models. It provides a uniform interface compatible with
     * ONNXRT and TFLiteRT wrappers.
     */
    class TVMRT
    {
        public:
            /**
             * @brief Constructor for the TVMRT class
             *
             * @param modelPath Path to the ONNX model file (used to derive artifact paths)
             */
            TVMRT(std::string modelPath);

            /**
             * @brief Destructor for the TVMRT class
             *
             * Cleans up resources allocated during the lifetime of the object
             */
            ~TVMRT();

            /**
             * @brief Creates and initializes the inference
             *
             * Sets up the TVM Runtime session with the specified options and prepares
             * the model for inference. This includes loading the compiled model artifacts.
             *
             * @param options Map of configuration options for the inference
             *                Expected keys:
             *                - "artifacts_folder": Path to TVM artifacts (deploy_lib.so, deploy_graph.json, deploy_param.params)
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
             * @brief Converts DLDataType to TIDL type
             *
             * @param dlType The DLDataType to convert
             * @param tidlType Reference to store the corresponding TIDL type
             * @param typeName Reference to store the corresponding type name
             * @return int32_t Size in bytes of the data type
             */
            static int32_t DLType2TidlType(const DLDataType &dlType, int32_t &tidlType, std::string &typeName);

        private:
            /**
             * @brief Populates information about the model's input tensors
             *
             * Queries the TVM Runtime module for details about input tensors
             * and populates the corresponding member variables.
             *
             * @return int32_t Status code (0 for success, negative for failure)
             */
            int32_t populateInputInfo();

            /**
             * @brief Populates information about the model's output tensors
             *
             * Queries the TVM Runtime module for details about output tensors
             * and populates the corresponding member variables.
             *
             * @return int32_t Status code (0 for success, negative for failure)
             */
            int32_t populateOutputInfo();

            /**
             * @brief Retrieves input names from the TVM graph module
             *
             * Uses TVM's get_input_info() API to extract input tensor names.
             * This approach is more robust than parsing JSON and doesn't require hardcoded name patterns.
             *
             * @return std::vector<std::string> Vector of input names
             */
            std::vector<std::string> getInputNames();

        private:
            /** @brief Path to the model file */
            std::string                             m_modelPath;

            /** @brief Flag indicating whether TIDL hardware acceleration is enabled (unused for TVM) */
            bool                                    m_tidlOffload;

            /** @brief TVM runtime module */
            tvm::runtime::Module                    m_graphModule;

            /** @brief TVM device context */
            DLDevice                                m_device;

            /** @brief Number of input tensors in the model */
            int32_t                                 m_numInputs;

            /** @brief Vector of input tensor names */
            std::vector<std::string>                m_inputNames;

            /** @brief Vector of input tensor details */
            std::vector<DlTensor>                   m_inputs;

            /** @brief Number of output tensors in the model */
            int32_t                                 m_numOutputs;

            /** @brief Vector of output tensor details */
            std::vector<DlTensor>                   m_outputs;

            /** @brief Artifacts folder path */
            std::string                             m_artifactsPath;

            /** @brief Vector of output tensor names */
            std::vector<std::string>                m_outputNames;
    };
} // namespace tvmrt_wrapper

#endif /* TVMRT_WRAPPER_H_*/
