/*
Copyright (c) 2020 – 2021 Texas Instruments Incorporated

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
#ifndef _UTILS_MODEL_INFO_H_
#define _UTILS_MODEL_INFO_H_

/* Standard headers. */
#include <map>
#include <set>
#include <fstream>
#include <experimental/filesystem>
#include <iostream>

/* Third-party headers. */
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

/*module header */
#include "model_info.h"
#include "ti_logger.h"
#include "edgeai_classnames.h"

namespace tidl
{
   namespace modelInfo
   {

      /**
       * \brief Constant for DLR invalid device Id.
       * \ingroup group_dl_inferer
       */
#define DLR_DEVID_INVALID (-1)

#define TI_PREPROC_DEFAULT_WIDTH 320
#define TI_PREPROC_DEFAULT_HEIGHT 240

#define TI_POSTPROC_DEFAULT_WIDTH 1280
#define TI_POSTPROC_DEFAULT_HEIGHT 720
#define TI_DEFAULT_DISP_WIDTH 1920
#define TI_DEFAULT_DISP_HEIGHT 1080

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
       * \ingroup group_edgeai_cpp_apps_post_proc
       */
      struct PreprocessImageConfig
      {
         /** Name of the model. */
         std::string modelName{};

         /** Type of the runtime API to invoke. The valid values are:
          * - DL_INFER_RTTYPE_DLR
          * - DL_INFER_RTTYPE_TFLITE
          * - DL_INFER_RTTYPE_ONNX
          */
         std::string rtType{};

         /** Task type.
          *  - detection
          *  - segmentation
          *  - classification
          */
         std::string taskType{};

         /** Width of the input data. */
         int32_t inDataWidth{TI_PREPROC_DEFAULT_WIDTH};

         /** Height of the input data. */
         int32_t inDataHeight{TI_PREPROC_DEFAULT_HEIGHT};

         /** Out width. */
         int32_t outDataWidth{TI_PREPROC_DEFAULT_WIDTH};

         /** Out height. */
         int32_t outDataHeight{TI_PREPROC_DEFAULT_HEIGHT};

         /** Mean values to apply during normalization. */
         std::vector<float> mean;

         /** Scale values to apply during normalization. */
         std::vector<float> scale;

         /** Resize width. */
         int32_t resizeWidth{TI_PREPROC_DEFAULT_WIDTH};

         /** Resize height. */
         int32_t resizeHeight{TI_PREPROC_DEFAULT_HEIGHT};

         /** Layout of the data. Allowed values. */
         std::string dataLayout{"NCHW"};

         /** Number of channels. */
         int32_t numChans{0};

         /** Data type of Input tensor. */
         tidl::modelInfo::DlInferType inputTensorType{tidl::modelInfo::DlInferType_Invalid};

         /**
          * Helper function to dump the configuration information.
          */
         void dumpInfo() const
         {
            LOG_INFO_RAW("\n");
            LOG_INFO("PreprocessImageConfig::modelName       = %s\n", modelName.c_str());
            LOG_INFO("PreprocessImageConfig::rtType          = %s\n", rtType.c_str());
            LOG_INFO("PreprocessImageConfig::taskType        = %s\n", taskType.c_str());
            LOG_INFO("PreprocessImageConfig::dataLayout      = %s\n", dataLayout.c_str());
            LOG_INFO("PreprocessImageConfig::inDataWidth     = %d\n", inDataWidth);
            LOG_INFO("PreprocessImageConfig::inDataHeight    = %d\n", inDataHeight);
            LOG_INFO("PreprocessImageConfig::resizeWidth     = %d\n", resizeWidth);
            LOG_INFO("PreprocessImageConfig::resizeHeight    = %d\n", resizeHeight);
            LOG_INFO("PreprocessImageConfig::outDataWidth    = %d\n", outDataWidth);
            LOG_INFO("PreprocessImageConfig::outDataHeight   = %d\n", outDataHeight);
            LOG_INFO("PreprocessImageConfig::numChannels     = %d\n", numChans);

            LOG_INFO("PreprocessImageConfig::mean          = [");
            for (uint32_t i = 0; i < mean.size(); i++)
            {
               LOG_INFO_RAW(" %f", mean[i]);
            }

            LOG_INFO_RAW(" ]\n");

            LOG_INFO("PreprocessImageConfig::scale         = [");
            for (uint32_t i = 0; i < scale.size(); i++)
            {
               LOG_INFO_RAW(" %f", scale[i]);
            }

            LOG_INFO_RAW(" ]\n\n");
         }
      };

      /**
       * \brief Configuration for the DL inferer.
       *
       * \ingroup group_edgeai_cpp_apps_post_proc
       */
      struct PostprocessImageConfig
      {
         /** Name of the model. */
         std::string modelName{};

         /** Type of the runtime API to invoke. The valid values are:
          * - DL_INFER_RTTYPE_DLR
          * - DL_INFER_RTTYPE_TFLITE
          * - DL_INFER_RTTYPE_ONNX
          */
         std::string rtType{};

         /** Task type.
          *  - detection
          *  - segmentation
          *  - classification
          */
         std::string taskType{};

         /** Layout of the data. Allowed values. */
         std::string dataLayout{"NCHW"};

         /** Optional offset to be applied when detecting the output
          * class. This is applicable for image classification and
          * detection cases only.
          * Classification - a single scalar value
          * Detection      - a map
          */
         std::map<int32_t, int32_t> labelOffsetMap{{0, 0}};

         /** Order of results for detection use case
          * default is assumed to be [0 1 2 3 4 5] which means
          * [x1y1 x2y2 label score]
          */
         std::vector<int32_t> formatter{0, 1, 2, 3, 4, 5};
         
         /** Formatter Name : DetectionYXYX2XYXY*/
         std::string formatterName;

         /** If detections are normalized to 0-1 */
         bool normDetect{false};

         /** Order of tensors for detection results */
         std::vector<int32_t> resultIndices{0, 1, 2};

         /** Multiplicative factor to be applied to Y co-ordinates. This is used
          * for visualization of the bounding boxes for object detection post-
          * processing only.
          */
         float vizThreshold{0.50f};

         /** Alpha value for blending. This is used for semantic segmentation
          *  post-processing only.
          */
         float alpha{0.5f};

         /** Number of classification results to pick from the top of the
         model output. */
         int32_t topN{5};

         /** Width of the output to display after adding tile. */
         int32_t dispWidth{TI_DEFAULT_DISP_WIDTH};

         /** Height of the output to display after adding tile. */
         int32_t dispHeight{TI_DEFAULT_DISP_HEIGHT};

         /** Width of the input data. */
         int32_t inDataWidth{TI_POSTPROC_DEFAULT_WIDTH};

         /** Height of the input data. */
         int32_t inDataHeight{TI_POSTPROC_DEFAULT_HEIGHT};

         /** Width of the output data. */
         int32_t outDataWidth{TI_POSTPROC_DEFAULT_WIDTH};

         /** Height of the output data. */
         int32_t outDataHeight{TI_POSTPROC_DEFAULT_HEIGHT};

         /** An array of strings for object class names. */
         const std::string *classnames{nullptr};

         /**
          * Helper function to dump the configuration information.
          */
         void dumpInfo() const
         {
            LOG_INFO_RAW("\n");
            LOG_INFO("PostprocessImageConfig::modelName      = %s\n", modelName.c_str());
            LOG_INFO("PostprocessImageConfig::rtType         = %s\n", rtType.c_str());
            LOG_INFO("PostprocessImageConfig::taskType       = %s\n", taskType.c_str());
            LOG_INFO("PostprocessImageConfig::dataLayout     = %s\n", dataLayout.c_str());
            LOG_INFO("PostprocessImageConfig::inDataWidth    = %d\n", inDataWidth);
            LOG_INFO("PostprocessImageConfig::inDataHeight   = %d\n", inDataHeight);
            LOG_INFO("PostprocessImageConfig::outDataWidth   = %d\n", outDataWidth);
            LOG_INFO("PostprocessImageConfig::outDataHeight  = %d\n", outDataHeight);
            LOG_INFO("PostprocessImageConfig::dispWidth      = %d\n", dispWidth);
            LOG_INFO("PostprocessImageConfig::dispHeight     = %d\n", dispHeight);
            LOG_INFO("PostprocessImageConfig::vizThreshold   = %f\n", vizThreshold);
            LOG_INFO("PostprocessImageConfig::alpha          = %f\n", alpha);
            LOG_INFO("PostprocessImageConfig::normDetect     = %d\n", normDetect);
            LOG_INFO("PostprocessImageConfig::labelOffsetMap = [ ");

            for (const auto labelOffset : labelOffsetMap)
            {
               int32_t key = labelOffset.first, value = labelOffset.second;
               LOG_INFO_RAW("(%d, %d) ", key, value);
            }

            LOG_INFO_RAW("]\n");

            LOG_INFO("PostprocessImageConfig::formatter = [ ");

            for (uint32_t i = 0; i < formatter.size(); i++)
            {
               LOG_INFO_RAW(" %d", formatter[i]);
            }

            LOG_INFO_RAW("]\n");

            LOG_INFO("PostprocessImageConfig::resultIndices = [ ");

            for (uint32_t i = 0; i < resultIndices.size(); i++)
            {
               LOG_INFO_RAW(" %d", resultIndices[i]);
            }

            LOG_INFO_RAW("]\n\n");
         }
      };

      /**
       * \brief Class for holding DL inference model parameters for setting up
       *        inference pipeline flow.
       *
       * \ingroup group_edgeai_demo_config
       */
      class ModelInfo
      {
      public:
         /** Default constructor. Use the compiler generated default one. */
         ModelInfo() = default;

         /** Constructor.
          *
          * @param mdoel_path model path.
          */
         ModelInfo(std::string m_modelPath);

         /** Destructor. */
         ~ModelInfo();

         /** Initializes the object. The following is done during
          * initialization:
          * - Parse the model specific param.yaml file and extract the
          *   model parameters.
          * - Instantiate an inference object
          * - Create the pre-processing
          */
         int32_t initialize();

         /**
          * Helper function to dump the configuration information.
          *
          * @param prefix Prefix to be added to the log outputs.
          */
         void dumpInfo(const char *prefix = "") const;

      public:
         /* Pre-process configuration. */
         PreprocessImageConfig m_preProcCfg;

         /* Post-processing configuration.*/
         PostprocessImageConfig m_postProcCfg;

         /* inferer configuration.*/
         InfererConfig m_infConfig;

         /** Path to the model. */
         std::string m_modelPath;

         /** Path to the filename with classnames. */
         std::string m_labelsPath;

         /** Alpha value used for blending the sementic segmentation output. */
         float m_alpha{0.5f};

         /** Threshold for visualizing the output from the detection models. */
         float m_vizThreshold{0.4f};

         /** Number of classification results to pick from the top of the model
         output. */
         int32_t m_topN{5};
      };

   } // namespace modelInfo

} // namespace tidl

#endif //_UTILS_MODEL_INFO_H_
