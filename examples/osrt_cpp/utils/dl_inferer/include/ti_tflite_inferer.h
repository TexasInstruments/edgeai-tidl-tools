/*
 *  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#if !defined(_TI_TFLITE_INFERER_)
#define _TI_TFLITE_INFERER_

/* Third-party headers. */
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

/* Module headers. */
#include "ti_dl_inferer.h"

/**
 * \defgroup group_tflite_inferer TFLITE runtime API
 *
 * \brief A class for encapsulating the TFLITE runtime API.
 * \ingroup group_dl_inferer
 */

namespace tidl
{
    namespace dlInferer
    {
            
    /** \brief A concrete class for TFLITE RT API.
     *
     * \ingroup group_tflite_inferer
     */
    class TFLiteInferer: public DLInferer
    {
        public:
            /**
             * Constructor.
             *
             * @param modelPath Path to the model.
             * @param artifactPath Path to the directory containing the model
             *                     artifacts.
             */
            TFLiteInferer(const std::string &modelPath,
                          const std::string &artifactPath);

            /**
             * Runs the model. This should be called only after all the inputs
             * have been set using setInput() call(s).
             *
             * @param inputs Input buffers to set for inference run
             * @param outputs Output buffers to set for inference run
             *
             * @returns 0 upon success. A nagative value otherwise.
             */
            virtual int32_t run(const VecDlTensorPtr &inputs,
                                VecDlTensorPtr       &outputs);

            /**
             * Dumps the model information to the screen.
             */
            virtual void dumpInfo();

            /**
             * Returns a pointer to an array containing detailed information on
             * the inputs of the model.
             *
             * @returns A pointer to an array of input interface parameters.
             */
            virtual const VecDlTensor *getInputInfo();

            /**
             * Returns an array containing detailed information on the outputs
             * of the model.
             *
             * @returns A pointer to an array of output interface parameters.
             */
            virtual const VecDlTensor *getOutputInfo();

            /**
             * Returns an allocated pointer that can be consumed by inference
             * of the model by this framework.
             *
             * @returns An pointer to allocated memory.
             */
            virtual void *allocate(int64_t size) override;

            /**
             * Destructor
             */
            ~TFLiteInferer();

        private:
            /** Path to the model. */
            std::string                                 m_modelPath;

            /** Path to the directory containing the model artifacts. */
            std::string                                 m_artifactPath;

            /**  A pointer to the model representation in memory. */
            std::unique_ptr<tflite::FlatBufferModel>    m_model;

            /** TODO. */
            tflite::ops::builtin::BuiltinOpResolver     m_resolver;

            /** A pointer to the model interpreter. */
            std::unique_ptr<tflite::Interpreter>        m_interpreter;
            
            /** Input tensor count. */
            uint32_t                                    m_numInputs;

            /** Output tensor count. */
            uint32_t                                    m_numOutputs;

            /** A list of input interface details. */
            VecDlTensor                                 m_inputs;

            /** A list of output interface details. */
            VecDlTensor                                 m_outputs;

        private:
            /**
             * Quesries the model and extracts the details of the input parameters.
             *
             * @returns 0 upon success. A negative value otherwise.
             */
            int32_t populateInputInfo();

            /**
             * Quesries the model and extracts the details of the output parameters.
             *
             * @returns 0 upon success. A negative value otherwise.
             */
            int32_t populateOutputInfo();
    };

    } // namespace dlInferer

} // namespace tidl

#endif // _TI_TFLITE_INFERER_

