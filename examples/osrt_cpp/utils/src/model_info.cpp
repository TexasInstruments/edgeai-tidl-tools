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

#include "../include/model_info.h"

namespace tidl
{
    namespace modelInfo
    {

        void InfererConfig::dumpInfo()
        {
            LOG_INFO("InfererConfig::Model Path        = %s\n", modelFile.c_str());
            LOG_INFO("InfererConfig::Artifacts Path    = %s\n", artifactsPath.c_str());
            LOG_INFO("InfererConfig::Runtime API       = %s\n", rtType.c_str());
            LOG_INFO("InfererConfig::Device Type       = %s\n", devType.c_str());
            LOG_INFO_RAW("\n");
        }

        int32_t getInfererConfig(const YAML::Node &appConfig,
                                 const std::string &modelBasePath,
                                 InfererConfig &infConfig)
        {
            const YAML::Node &n = appConfig["session"];
            int32_t status = 0;

            /* Validate the parsed yaml configuration and create the
            configuration for the inference object creation. */
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
                /*Initialize the inference configuration parameter object*/
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
                infConfig.artifactsPath = modelBasePath + "/" + artSrc;

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

            /*Validate the parsed yaml configuration and create the
            configuration for the inference object creation.*/
            if (!preProc)
            {
                LOG_ERROR("Preprocess configuration parameters missing.\n");
                status = -1;
            }
            else if (!session["input_mean"])
            {
                LOG_ERROR("Mean value specification missing. Setting deafult mean: 0\n");
            }
            else if (!session["input_scale"])
            {
                LOG_ERROR("Scale value specification missing.Setting default scale: 1\n");
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

            /* Check if crop information exists */
            if (status == 0)
            {
                config.rtType = session["session_name"].as<std::string>();
                config.taskType = taskType.as<std::string>();

                /* Read the width and height values */
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

                /* Read the data layout */
                config.dataLayout = preProc["data_layout"].as<std::string>();

                /* Read the mean values */
                if(session["input_mean"]){
                    const YAML::Node &meanNode = session["input_mean"];
                    for (uint32_t i = 0; i < meanNode.size(); i++)
                    {
                        config.mean.push_back(meanNode[i].as<float>());
                    }
                }else{
                    /*setting default mean to 0 , assuming 3 ch input */
                    config.mean.insert(config.mean.end(), { 0, 0, 0 });
                }

                /* Read the scale values */
                if(session["input_scale"]){
                    const YAML::Node &scaleNode = session["input_scale"];
                    for (uint32_t i = 0; i < scaleNode.size(); i++)
                    {
                        config.scale.push_back(scaleNode[i].as<float>());
                    }
                }else{
                     /*setting default scale to 1 , assuming 3 ch input */
                    config.scale.insert(config.scale.end(), { 1, 1, 1 });
                }
                /* Read the width and height values */
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

            /* Validate the parsed yaml configuration and create the configuration
            configuration for the inference object creation. */
            if (!session)
            {
                LOG_ERROR("Inference configuration parameters  missing.\n");
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

                /* Read the data layout */
                if (postProc && postProc["data_layout"])
                {
                    config.dataLayout = postProc["data_layout"].as<std::string>();
                }
                if (postProc && postProc["detection_thr"]){
                    config.vizThreshold = postProc["detection_thr"].as<float>();
                }
                if (postProc && postProc["formatter"] && postProc["formatter"]["name"]){

                    config.formatterName = postProc["formatter"]["name"].as<std::string>();
                }
                if (postProc && postProc["formatter"] && postProc["formatter"]["src_indices"])
                {
                    const YAML::Node &formatterNode = postProc["formatter"]["src_indices"];

                    /* default is assumed to be [0 1 2 3 4 5] which means
                     * [x1y1 x2y2 label score].
                     *
                     * CASE 1: Only 2 values are specified. These are assumed to
                     *         be "label" and "score". Keep [0..3] same as the
                     *         default values but overwrite [4,5] with these two
                     *         values.
                     *
                     * CASE 2: Only 4 values are specified. These are assumed to
                     *         be "x1y1" and "x2y2". Keep [4,5] same as the
                     *         default values but overwrite [0..3] with these
                     *         four values.
                     *
                     * CASE 3: All 6 values are specified. Overwrite the defaults.
                     * */
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

                if (postProc && postProc["normalized_detections"])
                {
                    config.normDetect = postProc["normalized_detections"].as<bool>();
                }

                if (postProc && postProc["shuffle_indices"])
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
                    /* Read the width and height values */
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

        ModelInfo::~ModelInfo()
        {
            LOG_DEBUG("DESTRUCTOR\n");
        }

        ModelInfo::ModelInfo(std::string modelPath)
        {
            this->m_modelPath = modelPath;
        }

        int32_t ModelInfo::initialize(tidl::utils::LogLevel logLevel)
        {
            YAML::Node yaml;
            int32_t status = 0;
            // m_modelPath points to artifacts folder, but param.yaml is present outside of artifacts folder 
            const std::string path_to_artifacts = "/artifacts";
            m_modelPath = m_modelPath.erase(m_modelPath.size()-path_to_artifacts.size());
            const std::string &configFile = m_modelPath + "/param.yaml";

            /* Check if the specified configuration file exists */
            std::ifstream file(configFile);
            if (!file)
            {
                LOG(FATAL) << "Labels file " << configFile << " not found\n";
                status = -1;
            }

            if (status == 0)
            {
                yaml = YAML::LoadFile(configFile.c_str());

                /* Populate m_infConfig from yaml */
                status = getInfererConfig(yaml, m_modelPath, m_infConfig);
                if (status < 0)
                {
                    LOG_ERROR("getInfererConfig() failed.\n");
                }
            }

            /* Populate pre-process config from yaml */
            if (status == 0)
            {
                status = getPreprocessImageConfig(yaml, m_preProcCfg);

                if (status < 0)
                {
                    LOG_ERROR("getPreprocessImageConfig() failed.\n");
                }
            }

            /* Populate post-process config from yaml */
            if (status == 0)
            {
                status = getPostprocessImageConfig(yaml, m_postProcCfg);

                if (status < 0)
                {
                    LOG_ERROR("getPostprocessImageConfig() failed.\n");
                }
            }
            /* Populate post-process config from yaml */
            if (status == 0)
            {
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
                if (m_postProcCfg.taskType == "segmentation")
                {
                    /* Either NCHW or CHW. Width is the last dimention and the height
                    is the previous to last. */
                    m_postProcCfg.classnames = nullptr;
                    m_postProcCfg.alpha = m_alpha;
                }
                else
                {
                    /* Query the output data dimension ofrom the pre-process module. */
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
                    }
                }

                std::string modelName = m_modelPath;

                if (modelName.back() == '/')
                {
                    modelName.pop_back();
                }

                //modelName = std::experimental::filesystem::path(modelName).filename();
                size_t sep = modelName.find_last_of("\\/");
                if (sep != std::string::npos)
                    modelName = modelName.substr(sep + 1, modelName.size() - sep - 1);

                m_preProcCfg.modelName = modelName;
                m_postProcCfg.modelName = modelName;
            }
            if(logLevel <= tidl::utils::INFO ){
                m_infConfig.dumpInfo();
                m_preProcCfg.dumpInfo();
                m_postProcCfg.dumpInfo();
            }
            return status;
        }

    } // namespace modelInfo

} // namespace tidl
