#include <assert.h>
#include <getopt.h>
#include <iostream>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <vector>
#include <limits>
#include <stdexcept>
#include <libgen.h>
#include <utility>
#include <sys/time.h>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/tidl/tidl_provider_factory.h>

#include "validator.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>
#include  <cstring>
#include <algorithm>
#include <functional>
#include <queue>
#include "itidl_rt.h"
#define ORT_ZERO_COPY_API (1)

template <class T>
std::vector<T> preprocImage(const std::string &input_bmp_name, int wanted_height, int wanted_width, int wanted_channels, std::vector<float> mean, std::vector<float> scale)
{
    int i,j;
    uint8_t *pSrc;
    cv::Mat spl[3];;
    std::vector<T> out(wanted_height * wanted_width * wanted_channels);
    cv::Mat image = cv::imread(input_bmp_name, cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(wanted_width, wanted_height), 0, 0, cv::INTER_AREA);

    if (image.channels() != wanted_channels)
    {
      printf("Warning : Number of channels wanted differs from number of channels in the actual image \n");
      exit(-1);
    }
    cv::split(image, spl);

    for (j = 0; j < wanted_channels; j++)
    {
      pSrc = (uint8_t *)spl[j].data;
      for (i = 0; i < wanted_height * wanted_width; i++)
      {
        out[j*(wanted_height * wanted_width) + i] = ((T)pSrc[i] - mean[j]) * scale[j];
      }
    }
    return out;
}

Validator::Validator(Ort::Env& env,
                     std::string model_path,
                     std::string image_path,
                     std::string labels_path,
                     Ort::SessionOptions& session_options)
    : _session(env, model_path.c_str(), session_options),
      _num_input_nodes{_session.GetInputCount()},
      _input_node_names(_num_input_nodes),
      _image_path(image_path),
      _labels_path(labels_path)
{
    Validate();
}

int Validator::GetImageSize() const
{
    return _image_size;
}

void Validator::PrepareInputs()
{
    Ort::AllocatorWithDefaultOptions allocator;
    
    printf("Number of inputs = %zu\n", _num_input_nodes);
    
    // iterate over all input nodes
    for (int i = 0; i < _num_input_nodes; i++) {
        // print input node names
        char* input_name = _session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        _input_node_names[i] = input_name;
        
        // print input node types
        Ort::TypeInfo type_info = _session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);
        
        // print input shapes/dims
        _input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, _input_node_dims.size());
        for (int j = 0; j < _input_node_dims.size(); j++)
        {
            printf("Input %d : dim %d=%jd\n", i, j, _input_node_dims[j]);
        }
    }
}
double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

void Validator::ScoreModel()
{


    int wanted_height   = 224;
    int wanted_width    = 224;
    int wanted_channels = 3;
    int num_iter = 100;
    std::vector<float> mean = {123.675, 116.28, 103.53};
    std::vector<float> scale = {0.017125, 0.017507, 0.017429};

    std::vector<float> image_data = preprocImage<float>(_image_path, wanted_height, wanted_width, wanted_channels, mean, scale);

    size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
    size_t output_tensor_size = 1000;  // simplify ... using known dim values to calculate size
    // use OrtGetTensorShapeElementCount() to get official size!

    // std::vector<float> input_tensor_values(input_tensor_size);
    std::vector<const char*> output_node_names = {"resnetv15_dense0_fwd"};
    printf("Output name -- %s \n", *(output_node_names.data()));

    // create input tensor object from data values
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    void *inData = TIDLRT_allocSharedMem(16, input_tensor_size * sizeof(float));
    if(inData == NULL)
    {
        printf("Could not allocate memory for inData \n ");
        exit(0);
    }
    memcpy(inData, image_data.data(), input_tensor_size * sizeof(float));

    void *outData = TIDLRT_allocSharedMem(16, output_tensor_size * sizeof(float));
    if(outData == NULL)
    {
        printf("Could not allocate memory for outData \n ");
        exit(0);
    }
    
 #if ORT_ZERO_COPY_API
    Ort::IoBinding binding(_session);
    const Ort::RunOptions  &runOpts = Ort::RunOptions();
    Ort::Value input_tensor  = Ort::Value::CreateTensor<float>(memory_info, (float*)inData, input_tensor_size, _input_node_dims.data(), 4);
    assert(input_tensor.IsTensor());
    std::vector<int64_t> _output_node_dims = {1,1,1,1000};

    Ort::Value output_tensors = Ort::Value::CreateTensor<float>(memory_info, (float*)outData, output_tensor_size, _output_node_dims.data(), 4);
    assert(output_tensors.IsTensor());

    binding.BindInput(_input_node_names[0], input_tensor);
    binding.BindOutput(output_node_names[0], output_tensors);
 
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, nullptr);
    for (int i = 0; i < num_iter; i++)
    {
      _session.Run(runOpts, binding);
    }
    gettimeofday(&stop_time, nullptr);
    float *floatarr = (float*)outData;
   
#else
    //Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, image_data.data(), input_tensor_size, _input_node_dims.data(), 4);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)inData, input_tensor_size, _input_node_dims.data(), 4);
    
    assert(input_tensor.IsTensor());

    // score model & input tensor, get back output tensor
    auto run_options = Ort::RunOptions();
    run_options.SetRunLogVerbosityLevel(2);

    auto output_tensors = _session.Run(run_options, _input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);

    struct timeval start_time, stop_time;
    gettimeofday(&start_time, nullptr);
    for (int i = 0; i < num_iter; i++)
    {
      output_tensors = _session.Run(run_options, _input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    }
    gettimeofday(&stop_time, nullptr);

    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
    // Get pointer to output tensor float values
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
#endif
    std::cout << "invoked \n";
    std::cout << "average time: "
              << (get_us(stop_time) - get_us(start_time)) / (num_iter * 1000)
              << " ms \n";

    // Determine most common index
    float max_val = 0.0;
    int max_index = 0;
    for (int i = 0; i < 1000; i++)
    {
        if (floatarr[i] > max_val)
        {
            max_val = floatarr[i];
            max_index = i;
        }
    }
    std::cout << "MAX: class [" << max_index << "] = " << max_val << std::endl;

    std::vector<std::string> labels = ReadFileToVec(_labels_path);
    std::cout << labels[max_index + 1] << std::endl;
}

void Validator::Validate()
{
    PrepareInputs();
    ScoreModel();
}

std::vector<std::string> Validator::ReadFileToVec(std::string fname)
{
    // Open the File
    std::ifstream file(fname.c_str());
  
    // Check if object is valid
    if(!file)
    {
        throw std::runtime_error("Cannot open file: " + fname);
    }

    // Read the next line from File untill it reaches the end.
    std::string line;
    std::vector<std::string> labels;
    while (std::getline(file, line))
    {
        // Line contains string of length > 0 then save it in vector
        if(!line.empty())
        {
            labels.push_back(line);
        }
    }
  
    // Close The File
    file.close();
    return labels;
}
