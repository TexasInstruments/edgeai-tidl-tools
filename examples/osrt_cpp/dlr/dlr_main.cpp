
#include "dlr_main.h"
#define LOG(x) std::cerr
namespace dlr
{
    namespace main
    {

        //===============================================================================
        /*! \brief Do inference on acquired images
 */
        int RunInference(Settings *s)
        {
            int index_top_results[1]; //Only Top1 results for now.. but can be expanded to top5 results
            int frame_cnt = 0;
            int num_outputs, num_inputs;
            double fp_ms_avg = 0.0; //Initial inference time
            DLRModelHandle model;
            int wanted_batch_size;
            int wanted_channels;
            int wanted_height;
            int wanted_width;
            std::cout << s->model_path;
            if (CreateDLRModel(&model, s->model_path.c_str(), 1, 0) != 0)
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
            if(s->isFormatNCHW){
                wanted_batch_size = input_shape[0];
                wanted_channels = input_shape[1];
                wanted_height = input_shape[2];
                wanted_width = input_shape[3];                
            }else{
                wanted_batch_size = input_shape[0];
                wanted_height = input_shape[1];
                wanted_width = input_shape[2];     
                wanted_channels = input_shape[3];
            }
            std::cout << "\nInference call (initialization) started...\n";
            uint8_t *in_data_resize = (uint8_t *)malloc(wanted_channels * (wanted_width * wanted_height));
            std::vector<uint8_t> in;
            std::string last_label = "None";
            std::vector<float> image_data(wanted_height * wanted_width * wanted_channels);
            cv::Mat img = tidl::preprocess::preprocImage<float>(s->input_bmp_path, image_data, wanted_height, wanted_width, wanted_channels, s->input_mean, s->input_std);

            std::cout << "Classifying input:" << s->input_bmp_path << std::endl;

            
            //Running inference
            if (SetDLRInput(&model, s->input_node_name.c_str(), input_shape, image_data.data(), 4) != 0)
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
            
            if (s->model_type == tidl::config::CLF)
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
                    if(s->isFormatNCHW){
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

        /**
  *  \brief  options parsing and infernce calling
  * @returns int
  */
        int DLR_Main(int argc, char **argv)
        {
            Settings s;
            int c;
            while (1)
            {
                static struct option long_options[] = {
                    {"accelerated", required_argument, nullptr, 'a'},
                    {"device_mem", required_argument, nullptr, 'd'},
                    {"count", required_argument, nullptr, 'c'},
                    {"verbose", required_argument, nullptr, 'v'},
                    {"threads", required_argument, nullptr, 't'},
                    {"warmup_runs", required_argument, nullptr, 'w'},
                    {nullptr, 0, nullptr, 0}};

                /* getopt_long stores the option index here. */
                int option_index = 0;

                c = getopt_long(argc, argv,
                                "a:c:d:t:v:w:", long_options,
                                &option_index);

                /* Detect the end of the options. */
                if (c == -1)
                    break;

                switch (c)
                {
                case 'a':
                    s.accel = strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
                    break;
                case 'c':
                    s.loop_count =
                        strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
                    break;
                case 'd':
                    s.device_mem =
                        strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
                    break;
                case 't':
                    s.number_of_threads = strtol( // NOLINT(runtime/deprecated_fn)
                        optarg, nullptr, 10);
                    break;
                case 'v':
                    s.verbose =
                        strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
                    break;
                case 'w':
                    s.number_of_warmup_runs =
                        strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
                    break;
                case 'h':
                case '?':
                    /* getopt_long already printed an error message. */
                    dlr::main::display_usage();
                    exit(-1);
                default:
                    exit(-1);
                }
            }
            for (int i = 0; i < NUM_CONFIGS; i++)
            {
                bool isTflModel = endsWith(tidl::config::model_configs[i].model_path, ".tflite");
                bool isOnnxModel = endsWith(tidl::config::model_configs[i].model_path, ".onnx");
                if (!isTflModel && !isOnnxModel)
                {
                    
                    s.artifact_path = tidl::config::model_configs[i].artifact_path;
                    s.model_path = tidl::config::model_configs[i].model_path;
                    //hard codign - should fix 
                    if(endsWith(s.model_path,"onnx_mobilenetv2")){
                        s.isFormatNCHW = true;
                    }
                    else{
                        s.isFormatNCHW = false;
                    }
                    s.labels_file_path = tidl::config::model_configs[i].labels_path;
                    s.input_bmp_path = tidl::config::model_configs[i].image_path;
                    s.input_mean = tidl::config::model_configs[i].mean;
                    s.input_std = tidl::config::model_configs[i].std;
                    s.model_type = tidl::config::model_configs[i].model_type;
                    s.input_node_name = tidl::config::model_configs[i].input_name;
                    RunInference(&s);
                }
            }

            return 0;
        }

    } // namespace main
} // namespace dlr

int main(int argc, char **argv)
{
    return dlr::main::DLR_Main(argc, argv);
}
