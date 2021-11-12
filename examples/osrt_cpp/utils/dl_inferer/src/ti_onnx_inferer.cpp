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
#include <cstring>
#include <string>
#include <algorithm>

/* Third-party headers. */
#include <core/providers/tidl/tidl_provider_factory.h>

/* Module headers. */
#include "../include/ti_onnx_inferer.h"

namespace tidl
{
    namespace dlInferer
    {
#define ONNX_NUM_DEFAULT_ELEM (200)

        int32_t Onnx2TiInferType(ONNXTensorElementDataType type,
                                 const char **typeName,
                                 DlInferType &tiType)
        {
            int32_t size;

            switch (type)
            {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                tiType = DlInferType_Int8;
                *typeName = "int8";
                size = sizeof(int8_t);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                tiType = DlInferType_UInt8;
                *typeName = "uint8";
                size = sizeof(uint8_t);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                tiType = DlInferType_Int16;
                *typeName = "int16";
                size = sizeof(int16_t);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                tiType = DlInferType_UInt16;
                *typeName = "uint16";
                size = sizeof(uint16_t);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                tiType = DlInferType_Int32;
                *typeName = "int32";
                size = sizeof(int32_t);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                tiType = DlInferType_UInt32;
                *typeName = "uint32";
                size = sizeof(uint32_t);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                tiType = DlInferType_Int64;
                *typeName = "int64";
                size = sizeof(int64_t);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                tiType = DlInferType_Float16;
                *typeName = "float16";
                size = sizeof(float);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                tiType = DlInferType_Float32;
                *typeName = "float32";
                size = sizeof(float);
                break;

            default:
                tiType = DlInferType_Invalid;
                *typeName = "invalid";
                size = 0;
            }

            return size;
        }

        ORTInferer::ORTInferer(const std::string &modelPath,
                               const std::string &artifactPath) : m_modelPath(modelPath),
                                                                  m_artifactPath(artifactPath),
                                                                  m_env(ORT_LOGGING_LEVEL_ERROR, __FUNCTION__),
                                                                  m_memInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
        {
            OrtStatus *ortStatus;
            Ort::SessionOptions sessionOpts;
            c_api_tidl_options tidlOpts{};
            int32_t status;

            sessionOpts.SetIntraOpNumThreads(1);

            sessionOpts.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            strcpy(tidlOpts.artifacts_folder, m_artifactPath.c_str());

            ortStatus = OrtSessionOptionsAppendExecutionProvider_Tidl(sessionOpts,
                                                                      &tidlOpts);

            //m_session = new Ort::Session(m_env, m_modelPath, sessionOpts);
            m_session = new Ort::Session(m_env, m_modelPath.c_str(), sessionOpts);

            // Query the input information
            status = populateInputInfo();

            // Query the output information
            if (status == 0)
            {
                status = populateOutputInfo();
            }

            if (status < 0)
            {
                throw std::runtime_error("ORTInferer object creation failed.");
            }

            LOG_DEBUG("CONSTRUCTOR\n");
        }

        int32_t ORTInferer::populateInputInfo()
        {
            int32_t numInfo;
            int32_t status = 0;

            /* Query the number of inputs. */
            numInfo = m_session->GetInputCount();
            m_numInputs = numInfo;

            /* Reserve the storage. */
            m_inputs.assign(numInfo, DlTensor());
            m_inputTypes.assign(numInfo, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
            m_inputNames.assign(numInfo, nullptr);

            for (int32_t i = 0; i < numInfo; i++)
            {
                DlTensor *info;
                int32_t status1;

                info = &m_inputs[i];

                /* Query input name. */
                info->name = m_session->GetInputName(i, m_allocator);

                if (info->name == nullptr)
                {
                    LOG_ERROR("GetInputName(%d) failed.\n", i);
                    status = -1;
                    break;
                }

                m_inputNames[i] = info->name;

                auto typeInfo = m_session->GetInputTypeInfo(i);
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

                /* Query input shape. */
                info->shape = tensorInfo.GetShape();

                /* Query input dimensions. */
                info->dim = tensorInfo.GetDimensionsCount();
                info->numElem = tensorInfo.GetElementCount();

                /* Get the type, type name, and size. */
                m_inputTypes[i] = tensorInfo.GetElementType();

                info->elemSize = Onnx2TiInferType(m_inputTypes[i],
                                                  &info->typeName,
                                                  info->type);

                info->size = info->numElem * info->elemSize;

                if (info->size == 0)
                {
                    LOG_ERROR("Invalid data size(%d).\n", i);
                    status = -1;
                    break;
                }

            } // for (int32_t i = 0; i < numInfo; i++)

            return status;
        }

        int32_t ORTInferer::populateOutputInfo()
        {
            int32_t numInfo;
            int32_t status = 0;

            /* Query the number of outputs. */
            numInfo = m_session->GetOutputCount();
            m_numOutputs = numInfo;

            /* Reserve the storage. */
            m_outputs.assign(numInfo, DlTensor());
            m_outputTypes.assign(numInfo, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
            m_outputNames.assign(numInfo, nullptr);

            for (int32_t i = 0; i < numInfo; i++)
            {
                DlTensor *info;
                int32_t status1;

                info = &m_outputs[i];

                /* Query input name. */
                info->name = m_session->GetOutputName(i, m_allocator);

                if (info->name == nullptr)
                {
                    LOG_ERROR("GetInputName(%d) failed.\n", i);
                    status = -1;
                    break;
                }

                m_outputNames[i] = info->name;

                auto typeInfo = m_session->GetOutputTypeInfo(i);
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

                /* Query input shape. */
                info->shape = tensorInfo.GetShape();

                /* Query input dimensions. */
                info->dim = tensorInfo.GetDimensionsCount();
                info->numElem = tensorInfo.GetElementCount();

                /* Get the type, type name, and size. */
                m_outputTypes[i] = tensorInfo.GetElementType();
                info->elemSize = Onnx2TiInferType(m_outputTypes[i],
                                                  &info->typeName,
                                                  info->type);

                info->size = info->numElem * info->elemSize;
            } // for (int32_t i = 0; i < numInfo; i++)

            return status;
        }

        int32_t ORTInferer::run(const VecDlTensorPtr &inputs,
                                VecDlTensorPtr &outputs)
        {
            DL_INFER_GET_EXCL_ACCESS;
            //auto null_outs = std::find_if(m_outputs.begin(), m_outputs.end(), [](auto t) { return t.size < 0; });
            //auto null_ins = std::find_if(m_inputs.begin(), m_inputs.end(), [](auto t) { return t.size < 0; });

            /*if(null_outs == m_outputs.end() && null_ins == m_inputs.end())
    {
        return run_zerocopy(inputs, outputs);
    }
    else*/
            {
                return run_memcopy(inputs, outputs);
            }
        }

        int32_t ORTInferer::run_memcopy(const VecDlTensorPtr &inputs,
                                        VecDlTensorPtr &outputs)
        {
            std::vector<Ort::Value> inputValues;
            std::vector<Ort::Value> outputValues;
            const Ort::RunOptions &runOpts = Ort::RunOptions();
            int32_t status = 0;

            for (uint32_t i = 0; i < m_numInputs; i++)
            {
                const DlTensor *info = inputs[i];
                Ort::Value v = Ort::Value::CreateTensor(m_memInfo,
                                                        (void *)info->data,
                                                        (size_t)info->size,
                                                        info->shape.data(),
                                                        info->shape.size(),
                                                        m_inputTypes[i]);

                inputValues.push_back(std::move(v));
            }

            for (uint32_t i = 0; i < m_numOutputs; i++)
            {
                outputValues.emplace_back(nullptr);
            }

            outputValues = m_session->Run(runOpts,
                                          m_inputNames.data(),
                                          inputValues.data(),
                                          m_numInputs,
                                          m_outputNames.data(),
                                          m_numOutputs);

            /* Copy the output buffers. */
            for (uint32_t i = 0; i < m_numOutputs; i++)
            {
                const DlTensor *origInfo = &m_outputs[i];
                DlTensor *info = outputs[i];
                auto &tensor = outputValues[i];
                void *src = tensor.GetTensorMutableData<void>();
                const auto &tsInfo = tensor.GetTensorTypeAndShapeInfo();
                int32_t newSize;

                /* Update the output tensor object details. */
                info->shape = tsInfo.GetShape();
                info->dim = tsInfo.GetDimensionsCount();
                info->numElem = tsInfo.GetElementCount();
                newSize = info->numElem * info->elemSize;

                /* Allocate new buffers if the new size is not the same as the old one.
         * This is typically the case for the detection models where the actual
         * tensor output dimensions are not known until one inference is run.
         */
                if (newSize != info->size)
                {
                    LOG_DEBUG("NEW_SIZE = %d OLD_SIZE = %d\n", newSize, info->size);
                    info->size = newSize;
                    info->allocateDataBuffer(*this);
                }

                memcpy(info->data, src, info->size);
            }

            return status;
        }

        int32_t ORTInferer::run_zerocopy(const VecDlTensorPtr &inputs,
                                         VecDlTensorPtr &outputs)
        {
            Ort::IoBinding binding(*m_session);
            const Ort::RunOptions &runOpts = Ort::RunOptions();
            int32_t status = 0;

            for (uint32_t i = 0; i < m_numInputs; i++)
            {
                const DlTensor *info = inputs[i];
                Ort::Value v = Ort::Value::CreateTensor(m_memInfo,
                                                        (void *)info->data,
                                                        (size_t)info->size,
                                                        info->shape.data(),
                                                        info->shape.size(),
                                                        m_inputTypes[i]);
                binding.BindInput(info->name, v);
            }

            for (uint32_t i = 0; i < m_numOutputs; i++)
            {
                const DlTensor *info = outputs[i];
                Ort::Value v = Ort::Value::CreateTensor(m_memInfo,
                                                        (void *)info->data,
                                                        (size_t)info->size,
                                                        info->shape.data(),
                                                        info->shape.size(),
                                                        m_outputTypes[i]);
                binding.BindOutput(info->name, v);
            }

            m_session->Run(runOpts, binding);

            return status;
        }

        void ORTInferer::dumpInfo()
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

        const VecDlTensor *ORTInferer::getInputInfo()
        {
            return &m_inputs;
        }

        const VecDlTensor *ORTInferer::getOutputInfo()
        {
            return &m_outputs;
        }

        ORTInferer::~ORTInferer()
        {
            LOG_DEBUG("DESTRUCTOR\n");

            /* Releast the memory allocated for the strings. */
            for (const auto &vec : {m_inputNames, m_outputNames})
            {
                for (const auto &s : vec)
                {
                    m_allocator.Free((void *)s);
                }
            }

            delete m_session;
        }

        void *ORTInferer::allocate(int64_t size)
        {
            void *mem = NULL;

            if (size > 0)
            {
                mem = new uint8_t[size];
            }

            return mem;
        }

    }

} // namespace ti::dl
