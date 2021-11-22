
#include "dlr_main.h"
#define LOG(x) std::cerr
namespace dlr
{
    namespace main
    {

 /**
  
  *  \brief  Actual infernce happening 
  *  \param  ModelInfo YAML parsed model info
  *  \param  Settings user input options  and default values of setting if any
  * @returns void
  */
    void RunInference(tidl::modelInfo::ModelInfo *modelInfo, tidl::arg_parsing::Settings *s)
        {
            int index_top_results[1]; //Only Top1 results for now.. but can be expanded to top5 results
            int frame_cnt = 0;
            int num_outputs, num_inputs;
            double fp_ms_avg = 0.0; //Initial inference time
            DLRModelHandle model;
            if (CreateDLRModel(&model, modelInfo->m_infConfig.artifactsPath.c_str(), 1, 0) != 0)
            {
                throw std::runtime_error("Could not load DLR Model");
            }
            //output vector infering
            GetDLRNumOutputs(&model, &num_outputs);
            std::vector<std::vector<float>> outputs;
            for (int i = 0; i < num_outputs; i++)
            {
                int64_t cur_size = 0;
                int cur_dim = 0;
                GetDLROutputSizeDim(&model, i, &cur_size, &cur_dim);
                std::vector<float> output(cur_size, 0);
                outputs.push_back(output);
            }

            //input vector infering -assuming single input
            GetDLRNumInputs(&model, &num_inputs);
            if (num_inputs != 1)
            {
                throw std::runtime_error("Model with more than one input not supported");
            }
            int64_t input_size = 0;
            int input_dim = 0;
            GetDLRInputSizeDim(&model, 0, &input_size, &input_dim);
            int64_t input_shape[input_dim];
            GetDLRInputShape(&model, 0, input_shape);
            int wanted_batch_size = input_shape[0];
            int wanted_height = modelInfo->m_preProcCfg.outDataHeight;
            int wanted_width = modelInfo->m_preProcCfg.outDataWidth;
            int wanted_channels = modelInfo->m_preProcCfg.numChans;

            std::cout << "\nInference call (initialization) started...\n";
            uint8_t *in_data_resize = (uint8_t *)malloc(wanted_channels * (wanted_width * wanted_height));
            std::vector<uint8_t> in;
            std::string last_label = "None";
            cv::Mat img;
            float image_data[wanted_height * wanted_width * wanted_channels];
            const char* input_type_feild[] = {};
            const char** input_type = &input_type_feild[0];

            GetDLRInputType(&model,0, input_type);
            printf("type: %s\n",*input_type);
            if(!strcmp(*input_type, "float32")){
                img = tidl::preprocess::preprocImage<float>(s->input_bmp_path, image_data, modelInfo->m_preProcCfg);
            }else{
                std::cout << "cannot handle input type " << *input_type << " yet";
                exit(-1);
            }
            std::cout << "Classifying input:" << s->input_bmp_path << std::endl;

            
            //Running inference
            if (SetDLRInput(&model, s->input_node_name.c_str(), input_shape, image_data, 4) != 0)
            {
                throw std::runtime_error("Could not set input '" + s->input_node_name + "'");
            }
            if (RunDLRModel(&model) != 0)
            {
                throw std::runtime_error("Could not run");
            }
            for (int i = 0; i < num_outputs; i++)
            {
                if (GetDLROutput(&model, i, outputs[i].data()) != 0)
                {
                    throw std::runtime_error("Could not get output" + std::to_string(i));
                }
            }
            
            if (!strcmp(modelInfo->m_preProcCfg.taskType.c_str(), "classification"))
            {
                const float threshold = 0.001f;
                std::vector<std::pair<float, int>> top_results;

                //assuming 1 output vector
                tidl::postprocess::get_top_n<float>(outputs[0].data(),
                                                    1000, s->number_of_results, threshold,
                                                    &top_results, true);
                std::vector<std::string> labels;
                size_t label_count;

                if (tidl::postprocess::ReadLabelsFile(s->labels_file_path, &labels, &label_count) != 0)
                {
                    throw std::runtime_error("Failed to load labels file\n");
                    exit(-1);
                }

                for (const auto &result : top_results)
                {
                    const float confidence = result.first;
                    const int index = result.second;
                    if(!strcmp(modelInfo->m_preProcCfg.dataLayout.c_str(),"NCHW")){
                        LOG(INFO) << confidence << ": " << index << " " << labels[index+1] << "\n";
                    }else{
                        //TODO fix from osrt_python 
                        LOG(INFO) << confidence << ": " << index-1 << " " << labels[index] << "\n";
                    }
                }
                int num_results = 5;
                img.data = tidl::postprocess::overlayTopNClasses(img.data, top_results, &labels, img.cols, img.rows, num_results);
            }

            std::cout << std::endl;
        }
    } // namespace main
} // namespace dlr

int main(int argc, char **argv)
{
  tidl::arg_parsing::Settings s;
  tidl::arg_parsing::parse_args(argc, argv, &s);
  tidl::arg_parsing::dump_args(&s);
  tidl::utils::logSetLevel((tidl::utils::LogLevel)s.log_level);
  // Parse the input configuration file
  tidl::modelInfo::ModelInfo model(s.model_zoo_path);

  int status = model.initialize();
  dlr::main::RunInference(&model, &s);

  return 0;
}
