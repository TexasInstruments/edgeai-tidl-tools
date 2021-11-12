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
#if !defined(_TI_DL_INFERER_)
#define _TI_DL_INFERER_

/* Standard headers. */
#include <string>
#include <cstdarg>
#include <vector>
#include <mutex>
#include <stdexcept>

/* Module headers. */
#include "../../include/ti_logger.h"

/**
 * \defgroup group_dl_inferer Deep Learning Inference engine
 *
 * \brief Unified interface for running different DL run time APIs. The goal of
 *        this class to provide a common interface to the application for
 *        different runtime API (ex:- DLR, TFLITE,..). These underlying
 *        runtime libraries provide different capabilities and when designing
 *        the applications to use differne models, it is important that the
 *        application has a common API that it can use for DL inference and
 *        this class provides just that.
 */

/**
 * \brief Constant for DLR invalid device Id.
 * \ingroup group_dl_inferer
 */
#define DLR_DEVID_INVALID (-1)

/**
 * \brief Constant for TFLITE RT API
 * \ingroup group_dl_inferer
 */
#define DL_INFER_RTTYPE_TFLITE "tflitert"

/**
 * \brief Constant for DLR RT API
 * \ingroup group_dl_inferer
 */
#define DL_INFER_RTTYPE_DLR "tvmdlr"

/**
 * \brief Constant for ONNX RT API
 * \ingroup group_dl_inferer
 */
#define DL_INFER_RTTYPE_ONNX "onnxrt"

namespace tidl
{
    namespace dlInferer
    {

        /**
     * \brief Enumeration for the different data types used for identifying
     *        the data types at the interface.
     *
     * \ingroup group_dl_inferer
     */
        typedef enum
        {
            /**  Invalid type. */
            DlInferType_Invalid = 0,

            /** Data type signed 8 bit integer. */
            DlInferType_Int8 = 2,

            /** Data type unsigned 8 bit integer. */
            DlInferType_UInt8 = 3,

            /** Data type signed 16 bit integer. */
            DlInferType_Int16 = 4,

            /** Data type unsigned 16 bit integer. */
            DlInferType_UInt16 = 5,

            /** Data type signed 32 bit integer. */
            DlInferType_Int32 = 6,

            /** Data type unsigned 32 bit integer. */
            DlInferType_UInt32 = 7,

            /** Data type signed 64 bit integer. */
            DlInferType_Int64 = 8,

            /** Data type 16 bit floating point. */
            DlInferType_Float16 = 9,

            /** Data type 32 bit floating point. */
            DlInferType_Float32 = 10,

        } DlInferType;

        /**
     * \brief Configuration for the DL inferer.
     *
     * \ingroup group_dl_inferer
     */
        struct InfererConfig
        {
            /** Path to the model directory or a file.
         *  - file for TFLITE and ONNX
         *  - directory for DLR
         **/
            std::string modelFile{};

            /** Path to the directory containing the model artifacts. This is only
         *  valid for TFLITE models and is not looked at for the other ones.
         */
            std::string artifactsPath{};

            /** Type of the runtime API to invoke. The valid values are:
         * - DL_INFER_RTTYPE_DLR
         * - DL_INFER_RTTYPE_TFLITE
         * - DL_INFER_RTTYPE_ONNX
         */
            std::string rtType{};

            /** Type of the device. This field is specific to the DLR API and is
         * is not looked at for the other ones. Please refer to the DLR API
         * specification for valid values this field can take.
         */
            std::string devType{};

            /** Id of the device. This field is specific to the DLR API and is
         * is not looked at for the other ones. Please refer to the DLR API
         * specification for valid values this field can take.
         */
            int32_t devId{DLR_DEVID_INVALID};

            /**
         * Helper function to dump the configuration information.
         */
            void dumpInfo();
        };

        /* Forward declaration. */
        class DLInferer;

        /**
     * \brief DLR Interface context. This captures various attributes associated
     *        with the inputs and outputs of the model. The 'data' field may be
     *        undefined but all other parameters will have valid values, if the
     *        model defines them.
     *
     *        For TVM generated models, the 'name' field could be NULL for the
     *        output paramaters.
     *
     * \ingroup group_dl_inferer
     */
        class DlTensor
        {
        public:
            /** Name of the element. */
            const char *name{nullptr};

            /** String representation of the element type. */
            const char *typeName{nullptr};

            /** Unified type across APIs. */
            DlInferType type;

            /** Total size in bytes of the input. This should be equal to
             *  numElem * sizeof(type).
             */
            int64_t size{};

            /** Total number of elements in the input. This should be equal
             *  to the product of all dimensions. The size of the type is
             *  not accounted in this.
             */
            int64_t numElem{};

            /** Element size in bytes. */
            int32_t elemSize{};

            /** Dimensions. */
            int32_t dim{};

            /** Shape information. */
            std::vector<int64_t> shape;

            /** Data buffer. */
            void *data{nullptr};

            /**
             * Default constructor.
             */
            DlTensor();

            /**
             * Copy constructor.
             */
            DlTensor(const DlTensor &rhs);

            /**
             * Dumps the information to the screen.
             */
            void dumpInfo() const;

            /**
             * Allocate memory for the buffer.
             */
            void allocateDataBuffer(DLInferer &inferer);

            /**
             * Assignment operator.
             *
             */
            DlTensor &operator=(const DlTensor &rhs);

            /**
             * Destructor
             */
            ~DlTensor();

        private:
            /** Flag to track if memory has been allocated. */
            bool dataAllocated{false};
        };

        /**
     * \brief Alias for a vector of DL interface info objects.
     *
     * \ingroup group_dl_inferer
     */
        using VecDlTensor = std::vector<DlTensor>;

        /** Alias for a vector of DlTensor pointers
     * \ingroup group_dl_inferer
     */
        using VecDlTensorPtr = std::vector<DlTensor *>;

        /** \brief An abstract base class for different class of RT inference API.
     *
     * \ingroup group_dl_inferer
     */
        class DLInferer
        {
        public:
            /**
             * Runs the model.
             *
             * @param inputs Input buffers to set for inference run
             * @param outputs Output buffers to set for inference run
             *
             * @returns 0 upon success. A nagative value otherwise.
             */
            virtual int32_t run(const VecDlTensorPtr &inputs,
                                VecDlTensorPtr &outputs) = 0;

            /**
             * Dumps the model information to the screen.
             */
            virtual void dumpInfo() = 0;

            /**
             * Returns an array containing detailed information on the inputs
             * of the model.
             *
             * @returns An array of input interface parameters.
             */
            virtual const VecDlTensor *getInputInfo() = 0;

            /**
             * Returns an array containing detailed information on the outputs
             * of the model.
             *
             * @returns An array of output interface parameters.
             */
            virtual const VecDlTensor *getOutputInfo() = 0;

            /**
             * Returns an allocated pointer that can be consumed by inference
             * of the model by this framework.
             *
             * @returns An pointer to allocated memory.
             */
            virtual void *allocate(int64_t size)
            {
                return new uint8_t[size];
            }

            /** Factory method for making a specifc inferer based on the
             * configuration passed.
             *
             * @param config Configuration specifying the type of inferer and
             *               associated parameters.
             * @returns A valid inferer if success. A nullptr otherwise.
             */
            static DLInferer *makeInferer(const InfererConfig &config);

            /**
             * Destructor.
             */
            virtual ~DLInferer() {}

        protected:
            /** Mutex for multi-thread access control. */
            std::mutex m_mutex;
        };

#define DL_INFER_GET_EXCL_ACCESS std::unique_lock<std::mutex> lock(this->m_mutex)

    } // namespace dlInferer

} // namespace tidl

#endif // _TI_DL_INFERER_
