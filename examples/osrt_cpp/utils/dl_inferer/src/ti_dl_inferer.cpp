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
#include <string>

/* Module headers. */
// #include "../include/ti_dlr_inferer.h"
#include "../include/ti_tflite_inferer.h"
#include "../include/ti_onnx_inferer.h"

/**
 * \brief Constant for DLR device type CPU.
 * \ingroup group_dl_inferer
 */
#define DLR_DEVTYPE_CPU (1)

/**
 * \brief Constant for DLR device type GPU.
 * \ingroup group_dl_inferer
 */
#define DLR_DEVTYPE_GPU (2)

/**
 * \brief Constant for DLR device type OPENCL.
 * \ingroup group_dl_inferer
 */
#define DLR_DEVTYPE_OPENCL (3)

namespace tidl
{
    namespace dlInferer
    {
        DlTensor::DlTensor()
        {
            LOG_DEBUG("DEFAULT CONSTRUCTOR\n");
        }

        DlTensor::DlTensor(const DlTensor &rhs) : name(rhs.name),
                                                  typeName(rhs.typeName),
                                                  type(rhs.type),
                                                  size(rhs.size),
                                                  elemSize(rhs.elemSize),
                                                  numElem(rhs.numElem),
                                                  dim(rhs.dim),
                                                  shape(rhs.shape),
                                                  dataAllocated(rhs.dataAllocated),
                                                  data(nullptr)
        {
            /* Do not point to the other object's data pointer. It
     * should be allocated as needed in the current object.
     */
            LOG_DEBUG("COPY CONSTRUCTOR\n");
        }

        void DlTensor::allocateDataBuffer(DLInferer &inferer)
        {
            if (dataAllocated)
            {
                delete[] reinterpret_cast<uint8_t *>(data);
            }

            data = inferer.allocate(size);
            dataAllocated = true;
        }

        void DlTensor::dumpInfo() const
        {
            if (name != nullptr)
            {
                LOG_INFO("    Name          = %s\n", name);
            }

            LOG_INFO("    Num Elements  = %ld\n", numElem);
            LOG_INFO("    Element Size  = %d bytes\n", elemSize);
            LOG_INFO("    Total Size    = %ld bytes\n", size);
            LOG_INFO("    Num Dims      = %d\n", dim);
            LOG_INFO("    Type          = %s (Enum: %d)\n", typeName, type);
            LOG_INFO("    Shape         = ");

            for (int32_t j = 0; j < dim; j++)
            {
                LOG_INFO_RAW("[%ld] ", shape[j]);
            }

            LOG_INFO_RAW("\n\n");
        }

        DlTensor &DlTensor::operator=(const DlTensor &rhs)
        {
            if (this != &rhs)
            {
                name = rhs.name;
                typeName = rhs.typeName;
                type = rhs.type;
                size = rhs.size;
                elemSize = rhs.elemSize;
                numElem = rhs.numElem;
                dim = rhs.dim;
                shape = rhs.shape;
                data = nullptr;
            }

            return *this;
        }

        DlTensor::~DlTensor()
        {
            LOG_DEBUG("DESTRUCTOR\n");

            /* The allocation is based on the size in bytes andof type
     * uint8_t * to be generic. Given this, the following should
     * be safe to do so.
     */
            if (dataAllocated)
            {
                delete[] reinterpret_cast<uint8_t *>(data);
            }
        }

        void InfererConfig::dumpInfo()
        {
            LOG_INFO("InfererConfig::Model Path        = %s\n", modelFile.c_str());
            LOG_INFO("InfererConfig::Artifacts Path    = %s\n", artifactsPath.c_str());
            LOG_INFO("InfererConfig::Runtime API       = %s\n", rtType.c_str());
            LOG_INFO("InfererConfig::Device Type       = %s\n", devType.c_str());
            LOG_INFO_RAW("\n");
        }

        DLInferer *DLInferer::makeInferer(const InfererConfig &config)
        {
            DLInferer *inter = nullptr;
            int32_t status = 0;

            if (config.modelFile.empty())
            {
                LOG_ERROR("Please specifiy a valid model path.\n");
                status = -1;
            }
            else if (config.rtType.empty())
            {
                LOG_ERROR("Please specifiy a valid run-time API type.\n");
                status = -1;
            }
            else if (config.rtType == DL_INFER_RTTYPE_TFLITE)
            {
                if (config.artifactsPath.empty())
                {
                    LOG_ERROR("Missing model artifacts path.\n");
                    status = -1;
                }
                else
                {
                    inter = new TFLiteInferer(config.modelFile, config.artifactsPath);
                }
            }
            // else if (config.rtType == DL_INFER_RTTYPE_DLR)
            // {
            //     const std::string &s = config.devType;
            //     int32_t devType;

            //     if (config.devId < 0)
            //     {
            //         LOG_ERROR("Missing device Id.\n");
            //         status = -1;
            //     }
            //     else if (s.empty())
            //     {
            //         LOG_ERROR("Missing device type.\n");
            //         status = -1;
            //     }
            //     else if (s == "CPU")
            //     {
            //         devType = DLR_DEVTYPE_CPU;
            //     }
            //     else if (s == "GPU")
            //     {
            //         devType = DLR_DEVTYPE_GPU;
            //     }
            //     else if (s == "OPENCL")
            //     {
            //         devType = DLR_DEVTYPE_OPENCL;
            //     }

            //     if (status == 0)
            //     {
            //         inter = new DLRInferer(config.artifactsPath, devType, config.devId);
            //     }
            // }
            else if (config.rtType == DL_INFER_RTTYPE_ONNX)
            {
                if (config.artifactsPath.empty())
                {
                    LOG_ERROR("Missing model artifacts path.\n");
                    status = -1;
                }
                else
                {
                    inter = new ORTInferer(config.modelFile,
                                           config.artifactsPath);
                }
            }
            else
            {
                LOG_ERROR("Unsupported RT API.\n");
            }

            return inter;
        }

    }

} // namespace tidl
