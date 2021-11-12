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
/* Standard headers. */
#include <stdio.h>
#include <dlfcn.h>

/* Third-party headers. */
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/util.h>

/* Module headers. */
#include "../include/ti_tflite_inferer.h"

namespace tidl
{
    namespace dlInferer
    {
        typedef void (*ErrorHandler)(const char *);
        typedef TfLiteDelegate *(*Create_delegate)(char **,
                                                   char **,
                                                   size_t,
                                                   void (*report_error)(const char *));

        DlInferType Tflite2TiInferType(TfLiteType type)
        {
            DlInferType tiType = DlInferType_Invalid;

            switch (type)
            {
            case kTfLiteFloat32:
                tiType = DlInferType_Float32;
                break;

            case kTfLiteInt16:
                tiType = DlInferType_Int16;
                break;

            case kTfLiteInt32:
                tiType = DlInferType_Int32;
                break;

            case kTfLiteInt64:
                tiType = DlInferType_Int64;
                break;

            case kTfLiteUInt8:
                tiType = DlInferType_UInt8;
                break;

            case kTfLiteInt8:
                tiType = DlInferType_Int8;
                break;

            case kTfLiteFloat16:
                tiType = DlInferType_Float16;
                break;
            }

            return tiType;
        }

        TFLiteInferer::TFLiteInferer(const std::string &modelPath,
                                     const std::string &artifactPath) : m_modelPath(modelPath),
                                                                        m_artifactPath(artifactPath)
        {
            Create_delegate createPlugin;
            const char *keys[2];
            const char *values[2];
            TfLiteDelegate *dlgPtr;
            ErrorHandler *errorHandler;
            void *lib;
            std::string path;
            size_t numOptions = 0;
            int32_t status = 0;

            m_model = tflite::FlatBufferModel::BuildFromFile(m_modelPath.c_str());

            if (m_model == nullptr)
            {
                LOG_ERROR("Model build failed.\n");
                status = -1;
            }

            if (status == 0)
            {
                tflite::InterpreterBuilder(*m_model, m_resolver)(&m_interpreter);
                if (m_interpreter == nullptr)
                {
                    LOG_ERROR("Inferer construction failed.\n");
                    status = -1;
                }
            }

            // Setup delegate
            if (status == 0)
            {
                path = "/usr/lib/libtidl_tfl_delegate.so";

                lib = dlopen(path.c_str(), RTLD_NOW);

                if (lib == NULL)
                {
                    LOG_ERROR("Opening TIDL delegate shared library failed [%s].\n",
                              path.c_str());
                    LOG_ERROR("Error: %s\n", dlerror());
                    status = -1;
                }
            }

            if (status == 0)
            {
                createPlugin =
                    (Create_delegate)dlsym(lib, "tflite_plugin_create_delegate");

                if (createPlugin == NULL)
                {
                    LOG_ERROR("Symbol lookup in delegate shared library failed.\n");
                    status = -1;
                }
            }

            if (status == 0)
            {
                // Currently, we program just one option
                keys[0] = "artifacts_folder";
                values[0] = m_artifactPath.c_str();
                numOptions++;

                // **keys, **values, num_options, error_handler
                dlgPtr = createPlugin((char **)keys, (char **)values, numOptions, NULL);

                m_interpreter->ModifyGraphWithDelegate(dlgPtr);

                // Allocate input and output tensors
                if (m_interpreter->AllocateTensors() != kTfLiteOk)
                {
                    LOG_ERROR("Tensor allocation failed.\n");
                    status = -1;
                }
            }

            if (status == 0)
            {
                // Query the input information
                status = populateInputInfo();
            }

            // Query the output information
            if (status == 0)
            {
                status = populateOutputInfo();
            }

            if (status < 0)
            {
                throw std::runtime_error("TFLiteInferer object creation failed.");
            }

            LOG_DEBUG("CONSTRUCTOR\n");
        }

        int32_t TFLiteInferer::populateInputInfo()
        {
            const std::vector<int> inputs = m_interpreter->inputs();

            m_numInputs = inputs.size();

            // Reserve the storage
            m_inputs.assign(m_numInputs, DlTensor());

            for (uint32_t i = 0; i < m_numInputs; i++)
            {
                DlTensor *info;
                TfLiteIntArray *dims;
                const TfLiteTensor *tensor;
                TfLiteType type;

                info = &m_inputs[i];

                // Total size of the data in bytes
                tensor = m_interpreter->input_tensor(i);
                dims = tensor->dims;
                type = TfLiteTensorType(tensor);
                info->name = TfLiteTensorName(tensor);
                info->size = TfLiteTensorByteSize(tensor);
                info->dim = TfLiteTensorNumDims(tensor);
                info->typeName = TfLiteTypeGetName(type);
                info->type = Tflite2TiInferType(type);
                info->numElem = 1;

                info->shape.assign(info->dim, 0);
                for (uint32_t j = 0; j < info->dim; j++)
                {
                    info->shape[j] = TfLiteTensorDim(tensor, j);
                    info->numElem *= info->shape[j];
                }

                info->elemSize = info->size / info->numElem;

            } // for (uint32_t i = 0; i < m_numInputs; i++)

            return 0;
        }

        int32_t TFLiteInferer::populateOutputInfo()
        {
            const std::vector<int> outputs = m_interpreter->outputs();

            m_numOutputs = outputs.size();

            // Reserve the storage
            m_outputs.assign(m_numOutputs, DlTensor());

            for (uint32_t i = 0; i < m_numOutputs; i++)
            {
                DlTensor *info;
                TfLiteIntArray *dims;
                const TfLiteTensor *tensor;
                TfLiteType type;

                info = &m_outputs[i];

                // Total size of the data in bytes
                tensor = m_interpreter->output_tensor(i);
                dims = tensor->dims;
                type = TfLiteTensorType(tensor);
                info->name = TfLiteTensorName(tensor);
                info->size = TfLiteTensorByteSize(tensor);
                info->dim = TfLiteTensorNumDims(tensor);
                info->typeName = TfLiteTypeGetName(type);
                info->type = Tflite2TiInferType(type);
                info->numElem = 1;

                info->shape.assign(info->dim, 0);
                for (uint32_t j = 0; j < info->dim; j++)
                {
                    info->shape[j] = TfLiteTensorDim(tensor, j);
                    info->numElem *= info->shape[j];
                }

                info->elemSize = info->size / info->numElem;

            } // for (uint32_t i = 0; i < m_numOutputs; i++)

            return 0;
        }

        int32_t TFLiteInferer::run(const VecDlTensorPtr &inputs,
                                   VecDlTensorPtr &outputs)
        {
            DL_INFER_GET_EXCL_ACCESS;
            TfLiteStatus tfStatus;
            int32_t status = 0;

            if (m_numInputs != inputs.size())
            {
                LOG_ERROR("Number of inputs does not match.\n");
                status = -1;
            }
            else if (m_numOutputs != outputs.size())
            {
                LOG_ERROR("Number of outputs does not match.\n");
                status = -1;
            }

            /* Set inputs and outputs (zero-copy). */
            if (status == 0)
            {
                for (uint32_t i = 0; i < m_numInputs; i++)
                {
                    int tensor_idx = m_interpreter->inputs()[i];
                    const TfLiteTensor *tensor = m_interpreter->input_tensor(i);
                    m_interpreter->SetCustomAllocationForTensor(tensor_idx,
                                                                {inputs[i]->data, TfLiteTensorByteSize(tensor)});
                }
                for (uint32_t i = 0; i < m_numOutputs; i++)
                {
                    int tensor_idx = m_interpreter->outputs()[i];
                    const TfLiteTensor *tensor = m_interpreter->output_tensor(i);
                    m_interpreter->SetCustomAllocationForTensor(tensor_idx,
                                                                {outputs[i]->data, TfLiteTensorByteSize(tensor)});
                }
            }

            /* Run the model. */
            if (status == 0)
            {
                tfStatus = m_interpreter->Invoke();

                if (tfStatus != kTfLiteOk)
                {
                    LOG_ERROR("Invoke() failed.\n");
                    status = -1;
                }
            }

            return status;
        }

        void TFLiteInferer::dumpInfo()
        {
            LOG_INFO("Model Path        = %s\n", m_modelPath.c_str());
            LOG_INFO("Number of Inputs  = %d\n", m_numInputs);

            for (uint32_t i = 0; i < m_numInputs; i++)
            {
                DlTensor *info = &m_inputs[i];

                LOG_INFO("INPUT [%d]: \n", i);
                info->dumpInfo();
            }

            LOG_INFO("Number of Outputs  = %d\n", m_numOutputs);

            for (uint32_t i = 0; i < m_numOutputs; i++)
            {
                DlTensor *info = &m_outputs[i];

                LOG_INFO("OUTPUT [%d]: \n", i);
                info->dumpInfo();
            }
        }

        const VecDlTensor *TFLiteInferer::getInputInfo()
        {
            return &m_inputs;
        }

        const VecDlTensor *TFLiteInferer::getOutputInfo()
        {
            return &m_outputs;
        }

        void *TFLiteInferer::allocate(int64_t size)
        {
            auto align = [](int alignment, int64_t size)
            {
                return alignment * ((size + (alignment - 1)) / alignment);
            };
            return aligned_alloc(tflite::kDefaultTensorAlignment,
                                 align(tflite::kDefaultTensorAlignment, size));
        }

        TFLiteInferer::~TFLiteInferer()
        {
            LOG_DEBUG("DESTRUCTOR\n");
        }

    }

} // namespace tidl
