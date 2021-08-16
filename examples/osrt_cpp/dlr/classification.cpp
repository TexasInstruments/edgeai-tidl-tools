#include <iostream>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <vector>
#include <limits>
#include <stdexcept>
#include <dlr.h>
#include <libgen.h>
#include <utility>
#include <thread>
#include <chrono>
#include <mutex>
#include <pthread.h>

#include <memory.h>
#include <unistd.h>
#include <getopt.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#define RES_X 224
#define RES_Y 224
#define BATCH_SIZE 1
#define NUM_CHANNELS 3

namespace dlr {
namespace classification {

//===============================================================================
/*! \DLR Classification
 */

char dlr_model[320];
std::string input_node_name("input"); //Mobilenet is default
std::string input_image("input");
std::vector<std::string> vecOfLabels;
char imagenet_win[160];
bool set_verbose = false;

template <class T>
int preprocImage(const std::string &input_bmp_name, T * outPtr, int wanted_height, int wanted_width, int wanted_channels, std::vector<float> mean, std::vector<float> scale)
{
    int i,j;
    uint8_t *pSrc;
    cv::Mat spl[3];;
    cv::Mat image = cv::imread(input_bmp_name, cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(wanted_width, wanted_height), 0, 0, cv::INTER_AREA);

    if (image.channels() != wanted_channels)
    {
      printf("Warning : Number of channels wanted differs from number of channels in the actual image \n");
      return (-1);
    }
    cv::split(image, spl);

    for (j = 0; j < wanted_channels; j++)
    {
      pSrc = (uint8_t *)spl[j].data;
      for (i = 0; i < wanted_height * wanted_width; i++)
      {
        outPtr[j*(wanted_height * wanted_width) + i] = ((T)pSrc[i] - mean[j]) * scale[j];
      }
    }
    return 0;
}

//------------------------------------------------------------------------------------------------------------------------------------------
int getTop_n(std::vector<std::vector<float>> outputs, int bcnt, int *index_top_results)
{
  const int single_inference_size =  outputs[0].size() / BATCH_SIZE;
    float max_pred = 0.0;
    int argmax = -1;

  for (int i = 0; i < single_inference_size; i++) {
    if (outputs[0][i] > max_pred) {
    max_pred = outputs[0][i];
    argmax = i;
    }
  }
  std::cout << " Max probability at " << argmax << " with probability " << max_pred;

  index_top_results[0]= argmax;

  if(argmax < vecOfLabels.size()) {
    std::cout << " label:" << vecOfLabels[argmax];
    std::cout << std::endl;
  }

  return argmax;

}

//===============================================================================
/*! \brief Do inference on acquired images
 */
void ProcessFrames(void ) {
  int index_top_results[1]; //Only Top1 results for now.. but can be expanded to top5 results
  std::vector<int64_t> output_sizes;
  int frame_cnt = 0;
  char tmp_string[160];
  int num_outputs;
  const int batch_size = 1;
  float *input_data = (float *)malloc(batch_size * RES_X * RES_Y * NUM_CHANNELS * sizeof(float));
  int64_t image_shape[4] = { batch_size, RES_Y, RES_X, NUM_CHANNELS };
  double fp_ms_avg = 0.0; //Initial inference time
  DLRModelHandle model;

  if (CreateDLRModel(&model, dlr_model, 1, 0) != 0) 
  {
    throw std::runtime_error("Could not load DLR Model");
  }

    GetDLRNumOutputs(&model, &num_outputs);

    for (int i = 0; i < num_outputs; i++) 
    {
      int64_t cur_size = 0;
      int cur_dim = 0;
      GetDLROutputSizeDim(&model, i, &cur_size, &cur_dim);
      output_sizes.push_back(cur_size);
    }

    std::vector<std::vector<float>> outputs;
    for (auto i : output_sizes) {
      outputs.push_back(std::vector<float>(i, 0));
    }
    //First inference is dummy, initialization call!!
    std::cout << "DUMMY inference call (initialization) started...\n";
    memset(input_data, 0, batch_size * RES_X * RES_Y * NUM_CHANNELS * sizeof(float));
    if (SetDLRInput(&model, input_node_name.c_str(), image_shape, input_data, 4) != 0) 
    {
        throw std::runtime_error("Could not set input '" + input_node_name + "'");
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
    std::cout << "...DUMMY inference call ended\n";

    //Used harcoded RES_X and RES_Y values
    uint8_t *in_data_resize = (uint8_t*)malloc(NUM_CHANNELS*(RES_X*RES_Y));
    std::vector<uint8_t> in;



    std::string last_label = "None";

    int wanted_height   = 224;
    int wanted_width    = 224;
    int wanted_channels = 3;
    std::vector<float> mean = {123.675, 116.28, 103.53};
    std::vector<float> scale = {0.017125, 0.017507, 0.017429};

    preprocImage<float>(input_image, (float*)input_data, wanted_height, wanted_width, wanted_channels, mean, scale);

      std::cout << "Classifying input:" << input_image << std::endl;


       //----------------------------------------------------------------------------
       // Single batch, runs BATCH_SIZE inferences
       //----------------------------------------------------------------------------

        if (SetDLRInput(&model, input_node_name.c_str(), image_shape, input_data, 4) != 0) 
        {
        throw std::runtime_error("Could not set input '" + input_node_name + "'");
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

      for(int bcnt = 0; bcnt < BATCH_SIZE; bcnt++)
      {
        int argmax = 0;
        std::cout << "[" << (frame_cnt + bcnt) << "]";
        argmax = getTop_n(outputs, bcnt, index_top_results);
        if(argmax < vecOfLabels.size()) {
        last_label = vecOfLabels[argmax];
      }



      }
      frame_cnt += BATCH_SIZE;
      std::cout << "processFrame: esc key is pressed by user" << std::endl;
}

/*! \brief Get all class labels from the provided file (one per line)
 */
bool getFileContent(std::string fileName, std::vector<std::string> & vecOfStrs)
{
  // Open the File
  std::ifstream in(fileName.c_str());
  // Check if object is valid
  if(!in)
  {
    std::cerr << "Cannot open the File : "<<fileName<<std::endl;
    return false;
  }
  std::string str;
  // Read the next line from File untill it reaches the end.
  while (std::getline(in, str))
  {
    // Line contains string of length > 0 then save it in vector
    if(str.size() > 0)
    vecOfStrs.push_back(str);
  }
  //Close The File
  in.close();
  return true;
}
//------------------------------------------------------------------------------------------------------------------------------------------
void print_usage(int argc, char **argv)
{
  std::cout << "Usage: " << argv[0] << std::endl;
  std::cout << "  --model (-m)  ... path to compiled TVM model\n";
  std::cout << "  --labels (-l) ... file with labels (mapping of class ID to human readable classes)\n";
  std::cout << "  --input_node (-n) ... name of TVM input node\n";
  std::cout << "  --input_image (-i) ... Path of input image file\n";
  std::cout << "  --verbose (-v) ... (flag) show frame processing time\n";
  std::cout << "  --help (-h) ... (flag) this help\n"; 
}



//------------------------------------------------------------------------------------------------------------------------------------------

int Main_dlr(int argc, char** argv) {
bool labels_ok = false;
int c;

  while (1) {
    int option_index = 0;
    static struct option long_options[] = {
      {"model",            required_argument, 0, 'm' },
      {"labels",           required_argument, 0, 'l' },
      {"input_image",      required_argument, 0, 'i' },
      {"input_node",       required_argument, 0, 'n' },
      {"verbose",          no_argument,       0, 'v' },
      {"help",             no_argument,       0, 'h' },
      {0,                  0,                 0,  0  }
    };

    c = getopt_long(argc, argv, "m:l:b:i:n:v:h", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
      case 'm':
        strcpy(dlr_model, optarg);
        break;
      case 'l':
        labels_ok = getFileContent(optarg, vecOfLabels);
        break;
      case 'b':
        std::cout << "Batch size hardcoded to:" << BATCH_SIZE << std::endl;
        break;
      case 'i':
        input_image = std::string(optarg); 
        break;
      case 'n':
        input_node_name = std::string(optarg); 
        break;
      case 'v':
        set_verbose = true;
        break;
      case 'h':
      case '?':
        print_usage(argc, argv);
        exit(EXIT_FAILURE);

      default:
        printf("!? getopt returned character code 0%o ??\n", c);
     }
  }
  std::cout << "DUMP CONFIGURATION:\n";
  std::cout << "Model:" << dlr_model << std::endl;
  std::cout << "Labels :" << labels_ok << std::endl;
  std::cout << "Input node name:" << input_node_name << std::endl;
  std::cout << "Input Image Path :" << input_image << std::endl;

  ProcessFrames();

  return 0;
}

}  // namespace classification
}  // namespace dlr

int main(int argc, char** argv) {
  return dlr::classification::Main_dlr(argc, argv);
}


