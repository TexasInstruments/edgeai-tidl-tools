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
#include <map>

/* Module headers. */
#include "../include/ti_dlr_inferer.h"

/* External headers */
#include <dlpack/dlpack.h>

namespace tidl
{
    namespace dlInferer
    {
        static int32_t type2Size(const char *s, DlInferType &tiType)
        {
            int32_t size = 1;

            tiType = DlInferType_Invalid;

            if (!strcmp(s, "float32"))
            {
                size = sizeof(float);
                tiType = DlInferType_Float32;
            }
            else if (!strcmp(s, "float16"))
            {
                size = sizeof(float);
                tiType = DlInferType_Float16;
            }
            else if (!strcmp(s, "int32"))
            {
                size = sizeof(int32_t);
                tiType = DlInferType_Int32;
            }
            else if (!strcmp(s, "int64"))
            {
                size = sizeof(int64_t);
                tiType = DlInferType_Int64;
            }
            else if (!strcmp(s, "int16"))
            {
                size = sizeof(int16_t);
                tiType = DlInferType_Int16;
            }
            else if (!strcmp(s, "int8"))
            {
                size = sizeof(int8_t);
                tiType = DlInferType_Int8;
            }
            else if (!strcmp(s, "uint8"))
            {
                size = sizeof(int8_t);
                tiType = DlInferType_UInt8;
            }

            return size;
        }

        DLRInferer::DLRInferer(const std::string &fileName, int32_t devType, int32_t devId) : m_name(fileName),
                                                                                              m_devType(devType),
                                                                                              m_devId(devId)
        {
            int32_t status = 0;

            status = CreateDLRModel(&m_handle,
                                    m_name.c_str(),
                                    m_devType,
                                    m_devId);

            if (status < 0)
            {
                LOG_ERROR("CreateDLRModel() failed. Error [%s].\n", DLRGetLastError());
            }

            // Query the input information
            if (status == 0)
            {
                status = populateInputInfo();
            }

            // Query the output information
            if (status == 0)
            {
                status = populateOutputInfo();
            }

            if (status < 0)
            {
                throw std::runtime_error("DLRInferer object creation failed.");
            }

            LOG_DEBUG("CONSTRUCTOR\n");
        }

        int32_t DLRInferer::populateInputInfo()
        {
            int32_t numInfo;
            int32_t status;

            /* Query the number of inputs. */
            status = GetDLRNumInputs(&m_handle, &numInfo);

            if (status < 0)
            {
                LOG_ERROR("GetDLRNumInputs() failed. Error [%s].\n", DLRGetLastError());
            }

            if (status == 0)
            {
                // Reserve the storage
                m_inputs.assign(numInfo, DlTensor());

                for (int32_t i = 0; i < numInfo; i++)
                {
                    DlTensor *info;
                    int32_t status1;

                    info = &m_inputs[i];

                    /* Query input name. */
                    status1 = GetDLRInputName(&m_handle,
                                              i,
                                              &info->name);

                    if (status1 < 0)
                    {
                        LOG_ERROR("GetDLRInputName(%d) failed. Error [%s].\n",
                                  i, DLRGetLastError());
                        status = status1;
                        break;
                    }

                    /* Query input type name. */
                    status1 = GetDLRInputType(&m_handle,
                                              i,
                                              &info->typeName);

                    if (status1 < 0)
                    {
                        LOG_ERROR("GetDLRInputType(%d) failed. Error [%s].\n",
                                  i, DLRGetLastError());
                        status = status1;
                        break;
                    }

                    /* Query input dimensions. */
                    status1 = GetDLRInputSizeDim(&m_handle,
                                                 i,
                                                 &info->numElem,
                                                 &info->dim);

                    if (status1 < 0)
                    {
                        LOG_ERROR("GetDLRInputSizeDim(%d) failed. Error [%s].\n",
                                  i, DLRGetLastError());
                        status = status1;
                        break;
                    }

                    info->elemSize = type2Size(info->typeName, info->type);
                    info->size = info->numElem * info->elemSize;

                    /* Query input shape. */
                    info->shape.assign(info->dim, 0);
                    status1 = GetDLRInputShape(&m_handle,
                                               i,
                                               info->shape.data());

                    if (status1 < 0)
                    {
                        LOG_ERROR("GetDLRInputShape(%d) failed. Error [%s].\n",
                                  i, DLRGetLastError());
                        status = status1;
                        break;
                    }

                } // for (int32_t i = 0; i < dlrObj->input.numInfo; i++)
            }

            return status;
        }

        int32_t DLRInferer::populateOutputInfo()
        {
            int32_t numInfo;
            int32_t status;

            /* Query the number of outputs. */
            status = GetDLRNumOutputs(&m_handle, &numInfo);

            if (status < 0)
            {
                LOG_ERROR("GetDLRNumOutputs() failed. Error [%s].\n",
                          DLRGetLastError());
            }

            if (status == 0)
            {
                // Reserve the storage
                m_outputs.assign(numInfo, DlTensor());

                for (int32_t i = 0; i < numInfo; i++)
                {
                    DlTensor *info;
                    int32_t status1;

                    info = &m_outputs[i];

                    /* Query output name. */
                    status1 = GetDLROutputName(&m_handle,
                                               i,
                                               &info->name);

                    if (status1 < 0)
                    {
                        LOG_WARN("GetDLROutputName(%d) NULL.\n", i);
                    }

                    /* Query output type name. */
                    status1 = GetDLROutputType(&m_handle,
                                               i,
                                               &info->typeName);

                    if (status1 < 0)
                    {
                        LOG_ERROR("GetDLROutputType(%d) failed. Error [%s].\n",
                                  i, DLRGetLastError());
                    }

                    /* Query output dimensions. */
                    status1 = GetDLROutputSizeDim(&m_handle,
                                                  i,
                                                  &info->numElem,
                                                  &info->dim);

                    if (status1 < 0)
                    {
                        LOG_ERROR("GetDLROutputSizeDim(%d) failed. Error [%s].\n",
                                  i, DLRGetLastError());
                        status = status1;
                        break;
                    }

                    info->elemSize = type2Size(info->typeName, info->type);
                    info->size = info->numElem * info->elemSize;

                    /* Query output shape. */
                    info->shape.assign(info->dim, 0);
                    status1 = GetDLROutputShape(&m_handle,
                                                i,
                                                info->shape.data());

                    if (status1 < 0)
                    {
                        LOG_ERROR("GetDLROutputSizeDim(%d) failed. Error [%s].\n",
                                  i, DLRGetLastError());
                        status = status1;
                        break;
                    }

                } // for (int32_t i = 0; i < numInfo; i++)
            }

            return status;
        }

        int32_t DLRInferer::run(const VecDlTensorPtr &inputs,
                                VecDlTensorPtr &outputs)
        {
            DL_INFER_GET_EXCL_ACCESS;
            int32_t status = 0;

            if (m_inputs.size() != inputs.size())
            {
                LOG_ERROR("Number of inputs does not match.\n");
                status = -1;
            }
            else if (m_outputs.size() != outputs.size())
            {
                LOG_ERROR("Number of outputs does not match.\n");
                status = -1;
            }

            /* Set inputs. */
            if (status == 0)
            {
                for (uint32_t i = 0; i < m_inputs.size(); i++)
                {
                    tidl::dlInferer::DlTensor *info = &m_inputs[i];
                    DLTensor dltensor;
                    auto cfunc = [info]()
                    {
                        std::map<DlInferType, DLDataTypeCode> cmap = {
                            {DlInferType_Int8, kDLInt},
                            {DlInferType_Int16, kDLInt},
                            {DlInferType_Int32, kDLInt},
                            {DlInferType_Int64, kDLInt},
                            {DlInferType_UInt8, kDLUInt},
                            {DlInferType_UInt16, kDLUInt},
                            {DlInferType_UInt32, kDLUInt},
                            {DlInferType_Float16, kDLFloat},
                            {DlInferType_Float32, kDLFloat},
                        };
                        return cmap[info->type];
                    };

                    dltensor.ctx = {kDLCPU, 0};
                    dltensor.ndim = info->dim;
                    dltensor.shape = info->shape.data();
                    dltensor.strides = nullptr;
                    dltensor.byte_offset = 0;
                    dltensor.dtype = {cfunc(),
                                      static_cast<uint8_t>(info->elemSize * 8),
                                      1};
                    dltensor.data = inputs[i]->data;

                    status = SetDLRInputTensorZeroCopy(&m_handle,
                                                       info->name,
                                                       &dltensor);

                    if (status < 0)
                    {
                        LOG_ERROR("SetDLRInputTensorZeroCopy(%d) failed.\n", i);
                        break;
                    }
                }
            }

            /* Run the model. */
            if (status == 0)
            {
                status = RunDLRModel(&m_handle);

                if (status < 0)
                {
                    LOG_ERROR("RunDLRModel() failed.\n");
                }
            }

            /* Get the outputs. */
            if (status == 0)
            {
                for (uint32_t i = 0; i < m_outputs.size(); i++)
                {
                    DlTensor *info = &m_outputs[i];

                    status = GetDLROutput(&m_handle, i, outputs[i]->data);

                    if (status < 0)
                    {
                        LOG_ERROR("SetDLRInput(%d) failed.\n", i);
                        break;
                    }
                }
            }

            return status;
        }

        void DLRInferer::dumpInfo()
        {
            LOG_INFO("Model Path        = %s\n", m_name.c_str());
            LOG_INFO("Model Device Type = %d\n", m_devType);
            LOG_INFO("Model Device Id   = %d\n", m_devId);
            LOG_INFO("Number of Inputs  = %ld\n", m_inputs.size());

            for (uint32_t i = 0; i < m_inputs.size(); i++)
            {
                DlTensor *info = &m_inputs[i];

                LOG_INFO("INPUT [%d]: \n", i);
                info->dumpInfo();
            }

            LOG_INFO("Number of Outputs  = %ld\n", m_outputs.size());

            for (uint32_t i = 0; i < m_outputs.size(); i++)
            {
                DlTensor *info = &m_outputs[i];

                LOG_INFO("OUTPUT [%d]: \n", i);
                info->dumpInfo();
            }
        }

        const VecDlTensor *DLRInferer::getInputInfo()
        {
            return &m_inputs;
        }

        const VecDlTensor *DLRInferer::getOutputInfo()
        {
            return &m_outputs;
        }

        DLRInferer::~DLRInferer()
        {
            int32_t status;

            LOG_DEBUG("DESTRUCTOR\n");

            /* Delete the DLR Model handle. */
            status = DeleteDLRModel(&m_handle);

            if (status < 0)
            {
                LOG_ERROR("DeleteDLRModel() failed. Error [%s].\n", DLRGetLastError());
            }
        }

        void *DLRInferer::allocate(int64_t size)
        {
            auto align = [](int alignment, int64_t size)
            {
                return alignment * ((size + (alignment - 1)) / alignment);
            };
            return aligned_alloc(128, align(128, size));
        }
    }

} // namespace ti::dl
