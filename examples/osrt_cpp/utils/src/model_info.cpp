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

#include "../include/model_info.h"

namespace tidl
{
    namespace modelInfo
    {
        int32_t getInfererConfig(const YAML::Node &appConfig,
                                 const std::string &modelBasePath,
                                 InfererConfig &infConfig)
        {
            const YAML::Node &n = appConfig["session"];
            int32_t status = 0;

            // Validate the parsed yaml configuration and create the configuration
            // for the inference object creation.
            if (!n)
            {
                LOG_ERROR("Inference configuration parameters  missing.\n");
                status = -1;
            }
            else if (!n["model_path"])
            {
                LOG_ERROR("Please specifiy a valid model path.\n");
                status = -1;
            }
            else if (!n["artifacts_folder"])
            {
                LOG_ERROR("Artifacts directory path missing.\n");
                status = -1;
            }
            else if (!n["session_name"])
            {
                LOG_ERROR("Please specifiy a valid run-time API type.\n");
                status = -1;
            }
            else
            {
                // Initialize the inference configuration parameter object
                infConfig.rtType = n["session_name"].as<std::string>();

                if (infConfig.rtType == "tvmdlr")
                {
                    infConfig.modelFile = modelBasePath + "/model";
                }
                else if ((infConfig.rtType == "tflitert") ||
                         (infConfig.rtType == "onnxrt"))
                {
                    const std::string &modSrc = n["model_path"].as<std::string>();
                    infConfig.modelFile = modelBasePath + "/" + modSrc;
                }

                const std::string &artSrc = n["artifacts_folder"].as<std::string>();
                infConfig.artifactsPath = modelBasePath + "/artifacts";

                if (n["device_type"])
                {
                    infConfig.devType = n["device_type"].as<std::string>();
                }
                else
                {
                    infConfig.devType = "CPU";
                }

                if (n["device_id"])
                {
                    infConfig.devId = n["device_id"].as<int32_t>();
                }
                else
                {
                    infConfig.devId = 0;
                }
            }

            return status;
        }

        int32_t getPreprocessImageConfig(const YAML::Node &yaml,
                                         PreprocessImageConfig &config)
        {
            const YAML::Node &session = yaml["session"];
            const YAML::Node &taskType = yaml["task_type"];
            const YAML::Node &preProc = yaml["preprocess"];
            int32_t status = 0;

            // Validate the parsed yaml configuration and create the configuration
            // for the inference object creation.
            if (!preProc)
            {
                LOG_ERROR("Preprocess configuration parameters missing.\n");
                status = -1;
            }
            else if (!preProc["mean"])
            {
                LOG_ERROR("Mean value specification missing.\n");
                status = -1;
            }
            else if (!preProc["scale"])
            {
                LOG_ERROR("Scale value specification missing.\n");
                status = -1;
            }
            else if (!preProc["data_layout"])
            {
                LOG_ERROR("Data layout specification missing.\n");
                status = -1;
            }
            else if (!preProc["resize"])
            {
                LOG_ERROR("Resize specification missing.\n");
                status = -1;
            }
            else if (!preProc["crop"])
            {
                LOG_ERROR("Crop specification missing.\n");
                status = -1;
            }

            // Check if crop information exists
            if (status == 0)
            {
                config.rtType = session["session_name"].as<std::string>();
                config.taskType = taskType.as<std::string>();

                // Read the width and height values
                const YAML::Node &cropNode = preProc["crop"];

                if (cropNode.Type() == YAML::NodeType::Sequence)
                {
                    config.outDataHeight = cropNode[0].as<int32_t>();
                    config.outDataWidth = cropNode[1].as<int32_t>();
                }
                else if (cropNode.Type() == YAML::NodeType::Scalar)
                {
                    config.outDataHeight = cropNode.as<int32_t>();
                    config.outDataWidth = config.outDataHeight;
                }

                // Read the data layout
                config.dataLayout = preProc["data_layout"].as<std::string>();

                // Read the mean values
                const YAML::Node &meanNode = preProc["mean"];
                for (uint32_t i = 0; i < meanNode.size(); i++)
                {
                    config.mean.push_back(meanNode[i].as<float>());
                }

                // Read the scale values
                const YAML::Node &scaleNode = preProc["scale"];
                for (uint32_t i = 0; i < scaleNode.size(); i++)
                {
                    config.scale.push_back(scaleNode[i].as<float>());
                }

                // Read the width and height values
                const YAML::Node &resizeNode = preProc["resize"];

                if (resizeNode.Type() == YAML::NodeType::Sequence)
                {
                    config.resizeHeight = resizeNode[0].as<int32_t>();
                    config.resizeWidth = resizeNode[1].as<int32_t>();
                }
                else if (resizeNode.Type() == YAML::NodeType::Scalar)
                {
                    int32_t resize = resizeNode.as<int32_t>();
                    int32_t minVal = std::min(config.inDataHeight, config.inDataWidth);

                    /* tiovxmultiscaler dosen't support odd resolutions */
                    config.resizeHeight = (((config.inDataHeight * resize) / minVal) >> 1) << 1;
                    config.resizeWidth = (((config.inDataWidth * resize) / minVal) >> 1) << 1;
                }

                if (config.mean.size() != config.scale.size())
                {
                    LOG_ERROR("The sizes of mean and scale vectors do not match.\n");
                    status = -1;
                }
            }

            config.numChans = config.mean.size();

            return status;
        }

        int32_t getPostprocessImageConfig(const YAML::Node &yaml,
                                          PostprocessImageConfig &config)
        {
            const YAML::Node &session = yaml["session"];
            const YAML::Node &taskType = yaml["task_type"];
            const YAML::Node &postProc = yaml["postprocess"];
            int32_t status = 0;

            // Validate the parsed yaml configuration and create the configuration
            // for the inference object creation.
            if (!session)
            {
                LOG_ERROR("Inference configuration parameters  missing.\n");
                status = -1;
            }
            else if (!postProc)
            {
                LOG_WARN("Postprocess configuration parameters missing.\n");
                status = -1;
            }
            else if (!taskType)
            {
                LOG_WARN("Tasktype configuration parameters missing.\n");
                status = -1;
            }

            if (status == 0)
            {
                config.rtType = session["session_name"].as<std::string>();
                config.taskType = taskType.as<std::string>();

                // Read the data layout
                if (postProc["data_layout"])
                {
                    config.dataLayout = postProc["data_layout"].as<std::string>();
                }

                if (postProc["formatter"] && postProc["formatter"]["src_indices"])
                {
                    const YAML::Node &formatterNode = postProc["formatter"]["src_indices"];

                    /* default is assumed to be [0 1 2 3 4 5] which means
             * [x1y1 x2y2 label score].
             *
             * CASE 1: Only 2 values are specified. These are assumed to
             *         be "label" and "score". Keep [0..3] same as the default
             *         values but overwrite [4,5] with these two values.
             *
             * CASE 2: Only 4 values are specified. These are assumed to
             *         be "x1y1" and "x2y2". Keep [4,5] same as the default
             *         values but overwrite [0..3] with these four values.
             *
             * CASE 3: All 6 values are specified. Overwrite the defaults.
             *
             */
                    if (formatterNode.size() == 2)
                    {
                        config.formatter[4] = formatterNode[0].as<int32_t>();
                        config.formatter[5] = formatterNode[1].as<int32_t>();
                    }
                    else if ((formatterNode.size() == 6) ||
                             (formatterNode.size() == 4))
                    {
                        for (uint8_t i = 0; i < formatterNode.size(); i++)
                        {
                            config.formatter[i] = formatterNode[i].as<int32_t>();
                        }
                    }
                    else
                    {
                        LOG_ERROR("formatter specification incorrect.\n");
                        status = -1;
                    }
                }

                if (postProc["normalized_detections"])
                {
                    config.normDetect = postProc["normalized_detections"].as<bool>();
                }

                if (postProc["shuffle_indices"])
                {
                    const YAML::Node indicesNode = postProc["shuffle_indices"];

                    for (uint8_t i = 0; i < indicesNode.size(); i++)
                    {
                        config.resultIndices[i] = indicesNode[i].as<int32_t>();
                    }
                }

                const YAML::Node &metric = yaml["metric"];

                if (metric && metric["label_offset_pred"])
                {
                    // Read the width and height values
                    const YAML::Node &offset = metric["label_offset_pred"];

                    if (offset.Type() == YAML::NodeType::Scalar)
                    {
                        /* Use "0" key to store the value. */
                        config.labelOffsetMap[0] = offset.as<int32_t>();
                    }
                    else if (offset.Type() == YAML::NodeType::Map)
                    {
                        for (const auto &it : offset)
                        {
                            if (it.second.Type() == YAML::NodeType::Scalar)
                            {
                                config.labelOffsetMap[it.first.as<int32_t>()] =
                                    it.second.as<int32_t>();
                            }
                        }
                    }
                    else
                    {
                        LOG_ERROR("label_offset_pred specification incorrect.\n");
                        status = -1;
                    }
                }
            }

            return status;
        }

        void ModelInfo::dumpInfo(const char *prefix) const
        {
            LOG_INFO("%sModelInfo::modelPath     = %s\n", prefix, m_modelPath.c_str());
            LOG_INFO("%sModelInfo::labelsPath    = %s\n", prefix, m_labelsPath.c_str());
            LOG_INFO("%sModelInfo::vizThreshold  = %f\n", prefix, m_vizThreshold);
            LOG_INFO("%sModelInfo::alpha         = %f\n", prefix, m_alpha);
            LOG_INFO("%sModelInfo::topN          = %d\n", prefix, m_topN);
            LOG_INFO_RAW("\n");
        }

        int32_t ModelInfo::initialize()
        {
            YAML::Node yaml;
            int32_t status = 0;
            InfererConfig infConfig;
            const std::string &configFile = m_modelPath + "/param.yaml";

            // Check if the specified configuration file exists
            std::ifstream file(configFile);
            if (!file)
            {
                LOG(FATAL) << "Labels file " << configFile << " not found\n";
                status = -1;
            }

            if (status == 0)
            {
                yaml = YAML::LoadFile(configFile.c_str());

                // Populate infConfig from yaml
                status = getInfererConfig(yaml, m_modelPath, infConfig);

                if (status < 0)
                {
                    LOG_ERROR("getInfererConfig() failed.\n");
                }
            }

            // Populate pre-process config from yaml
            if (status == 0)
            {
                status = getPreprocessImageConfig(yaml, m_preProcCfg);

                if (status < 0)
                {
                    LOG_ERROR("getPreprocessImageConfig() failed.\n");
                }
            }

            // Populate post-process config from yaml
            if (status == 0)
            {
                status = getPostprocessImageConfig(yaml, m_postProcCfg);

                if (status < 0)
                {
                    LOG_ERROR("getPostprocessImageConfig() failed.\n");
                }
            }

            // Populate post-process config from yaml
            if (status == 0)
            {

                /* Query the input information for setting the tensor type in pre process. */
                // dlInfInputs = m_infererObj->getInputInfo();
                // ifInfo = &dlInfInputs->at(0);
                // m_preProcCfg.inputTensorType = ifInfo->type;

                /* Set input data width and height based on the infererence engine
         * information. This is only used for semantic segmentation models
         * which have 4 dimensions. The logic is extended to any models that
         * have atleast three dimensions which has the following
         * - Num channels (C)
         * - Height (H)
         * - Width (W)
         *
         * The semantic segmentation model output will have one extra dimension
         * which leads to NCHW dimensions in the output.
         * - Batch (N)
         *
         * For all other cases, the default values (set in the post-process
         * obhect are used.
         */
                // ifInfo = &dlInfOutputs->at(0);

                if (m_postProcCfg.taskType == "segmentation")
                {
                    /* Either NCHW or CHW. Width is the last dimention and the height 
             * is the previous to last.
            //  */
            //         m_postProcCfg.inDataWidth = ifInfo->shape[ifInfo->dim - 1];
            //         m_postProcCfg.inDataHeight = ifInfo->shape[ifInfo->dim - 2];
                    m_postProcCfg.classnames = nullptr;
                    m_postProcCfg.alpha = m_alpha;
                }
                else
                {
                    // Query the output data dimension ofrom the pre-process module.
                    m_postProcCfg.inDataWidth = m_preProcCfg.outDataWidth;
                    m_postProcCfg.inDataHeight = m_preProcCfg.outDataHeight;

                    if (m_postProcCfg.taskType == "classification")
                    {
                        m_postProcCfg.classnames = ti::common::gClassNameMap["imagenet"];
                        m_postProcCfg.topN = m_topN;
                    }
                    else
                    {
                        m_postProcCfg.classnames = ti::common::gClassNameMap["coco"];
                        m_postProcCfg.vizThreshold = m_vizThreshold;
                    }
                }

                std::string modelName = m_modelPath;

                if (modelName.back() == '/')
                {
                    modelName.pop_back();
                }

                modelName = std::experimental::filesystem::path(modelName).filename();

                m_preProcCfg.modelName = modelName;
                m_postProcCfg.modelName = modelName;
            }

            return status;
        }

    } // namespace modelInfo

} // namespace tidl
