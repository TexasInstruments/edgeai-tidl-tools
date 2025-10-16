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

#ifndef TIDLRT_WRAPPER_H_
#define TIDLRT_WRAPPER_H_

#include <iostream>
#include <map>
#include <vector>
#include <string>

#include "itidl_rt.h"
#include "itidl_io.h"
#include "dltensor.h"

using namespace dl_tensor;

namespace tidlrt_wrapper
{
    /**
     * @class TIDLRT
     * @brief A wrapper class for TIDL Runtime that provides an interface for model inference
     * 
     * This class encapsulates the functionality required to load, initialize, and run
     * inference on TIDL models.
     */
    class TIDLRT
    {
        public:
            /**
             * @brief Constructor for the TIDLRT class
             * 
             */
            TIDLRT();

            /**
             * @brief Destructor for the TIDLRT class
             * 
             * Cleans up resources allocated during the lifetime of the object
             */
            ~TIDLRT();

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
            int32_t createInfer(std::map<std::string, std::string> &options);

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
             *          'graph_time': Total TIDL graph processing time (ms)
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
        
        private:
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
            int32_t populateInputInfo();

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
            int32_t populateOutputInfo();

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
             */
            static int32_t Tidl2TidlType(const int32_t &type, int32_t &tidlType, std::string &typeName);

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
            static bool endsWith(const std::string& str, const std::string& end);

        private:
            /** @brief Path to the network binary file */
            std::string                             m_netBinPath;

            /** @brief Path to the IO buffer descriptor binary file */
            std::string                             m_ioBinPath;

            /** @brief TIDL Runtime parameters structure */
            sTIDLRT_Params_t                        m_params;

            /** @brief TIDL Runtime handle */
            void                                    *m_handle;

            /** @brief Vector of input tensor structures for TIDL Runtime */
            std::vector<sTIDLRT_Tensor_t>           m_inTensor;

            /** @brief Array of pointers to input tensor structures */
            sTIDLRT_Tensor_t                        *m_inTensorPtr[TIDL_MAX_ALG_IN_BUFS];

            /** @brief Vector of output tensor structures for TIDL Runtime */
            std::vector<sTIDLRT_Tensor_t>           m_outTensor;

            /** @brief Array of pointers to output tensor structures */
            sTIDLRT_Tensor_t                        *m_outTensorPtr[TIDL_MAX_ALG_OUT_BUFS];

            /** @brief Number of input tensors in the model */
            int32_t                                 m_numInputs;

            /** @brief Vector of input tensor details */
            std::vector<DlTensor>                   m_inputs;

            /** @brief Number of output tensors in the model */
            int32_t                                 m_numOutputs;

            /** @brief Vector of output tensor details */
            std::vector<DlTensor>                   m_outputs;

            /** @brief Invoke start timestamp */
            uint64_t                                m_invokeStart;

            /** @brief Invoke end timestamp */
            uint64_t                                m_invokeEnd;

            /** @brief DDR Read B/W before run start */
            uint64_t                                m_ddrReadStart;

            /** @brief DDR Read B/W after run end */
            uint64_t                                m_ddrReadEnd;

            /** @brief DDR Write B/W before run start */
            uint64_t                                m_ddrWriteStart;

            /** @brief DDR Write B/W after run end */
            uint64_t                                m_ddrWriteEnd;
    };
} // namespace tidlrt_wrapper

#endif /* TIDLRT_WRAPPER_H_*/
