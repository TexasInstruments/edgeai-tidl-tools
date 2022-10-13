/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "classification.h"
#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>



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

#define LOG(x) std::cerr

void* in_ptrs[16] = {NULL};
void* out_ptrs[16]= {NULL};

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
int32_t ReadLabelsFile(const std::string& file_name,
                            std::vector<std::string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(FATAL) << "Labels file " << file_name << " not found\n";
    return -1;
  }
  result->clear();
  std::string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return 0;
}

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
template <class T>
void get_top_n(T* prediction, int prediction_size, size_t num_results,
               float threshold, std::vector<std::pair<float, int>>* top_results,
               bool input_floating) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>>
      top_result_pq;

  const long count = prediction_size;  // NOLINT(runtime/int)
  for (int i = 0; i < count; ++i) {
    float value;
    if (input_floating)
      value = prediction[i];
    else
      value = prediction[i] / 255.0;
    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }

  std::reverse(top_results->begin(), top_results->end());
}

template <class T>
int preprocImage(const std::string &input_image_name, T *out, int wanted_height, int wanted_width, int wanted_channels, float mean, float scale)
{
    int i;
    uint8_t *pSrc;
    cv::Mat image = cv::imread(input_image_name, cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(wanted_width, wanted_height), 0, 0, cv::INTER_AREA);

    if (image.channels() != wanted_channels)
    {
      printf("Warning : Number of channels wanted differs from number of channels in the actual image \n");
      return (-1);
    }
    pSrc = (uint8_t *)image.data;
    for (i = 0; i < wanted_height * wanted_width * wanted_channels; i++)
      out[i] = ((T)pSrc[i] - mean) / scale;
    return 0;
}


void getModelNameromArtifactsDir(char* path, char * net_name, char *io_name)
{
  char sys_cmd[500];
  sprintf(sys_cmd, "ls %s/*net.bin | head -1", path);
  FILE * fp = popen(sys_cmd,  "r");
  if (fp == NULL)
  {
    printf("Error while runing command : %s", sys_cmd);
  }
  fscanf(fp, "%s", net_name);
  fclose(fp);

  sprintf(sys_cmd, "ls %s/*io_1.bin | head -1", path);
  fp = popen(sys_cmd,  "r");
  if (fp == NULL)
  {
    printf("Error while runing command : %s", sys_cmd);
  }
  fscanf(fp, "%s", io_name);
  fclose(fp);
  return;
}
int32_t TIDLReadBinFromFile(const char *fileName, void *addr, int32_t size)
{
    FILE *fptr = NULL;
    fptr = fopen((const char *)fileName, "rb");
    if (fptr)
    {
      fread(addr, size, 1, fptr);
      fclose(fptr);
      return 0;
    }
    else
    {
      printf("Could not open %s file for reading \n", fileName);
    }
    return -1;
}

int RunInference(Settings* s) {

  char net_name[512];
  char io_name[512];

  getModelNameromArtifactsDir((char *)s->artifact_path.c_str(), net_name, io_name);

  printf("Model Files names : %s,%s\n", net_name, io_name);

  sTIDLRT_Params_t prms;
  void *handle = NULL;
  int32_t status;

  status = TIDLRT_setParamsDefault(&prms);


  FILE * fp_network = fopen(net_name, "rb");
  if (fp_network == NULL)
  {
    printf("Invoke  : ERROR: Unable to open network file %s \n", net_name);
    return -1;
  }
  prms.stats = (sTIDLRT_PerfStats_t*)malloc(sizeof(sTIDLRT_PerfStats_t));

  fseek(fp_network, 0, SEEK_END);
  prms.net_capacity = ftell(fp_network);
  fseek(fp_network, 0, SEEK_SET);
  fclose(fp_network);
  prms.netPtr = malloc(prms.net_capacity);

  status = TIDLReadBinFromFile(net_name, prms.netPtr, prms.net_capacity);

  FILE * fp_config = fopen(io_name, "rb");
  if (fp_config == NULL)
  {
    printf("Invoke  : ERROR: Unable to open IO config file %s \n", io_name);
    return -1;
  }
  fseek(fp_config, 0, SEEK_END);
  prms.io_capacity = ftell(fp_config);
  fseek(fp_config, 0, SEEK_SET);
  fclose(fp_config);
  prms.ioBufDescPtr = malloc(prms.io_capacity);
  status = TIDLReadBinFromFile(io_name, prms.ioBufDescPtr, prms.io_capacity);

  //prms.traceLogLevel = 3;
  //prms.traceWriteLevel = 3;

  status = TIDLRT_create(&prms, &handle);

  sTIDLRT_Tensor_t *in[16];
  sTIDLRT_Tensor_t *out[16];

  sTIDLRT_Tensor_t in_tensor;
  sTIDLRT_Tensor_t out_tensor;

  int32_t j = 0;
  in[j] = &in_tensor;
  status = TIDLRT_setTensorDefault(in[j]);
  in[j]->layout = TIDLRT_LT_NHWC;
  //strcpy((char *)in[j]->name, tensor->name);
  in[j]->elementType = TIDLRT_Uint8;
  int32_t in_tensor_szie = 224 * 224 * 3 * sizeof(uint8_t);

  if (s->device_mem)
  { 
      in[j]->ptr =  TIDLRT_allocSharedMem(64, in_tensor_szie);
      in[j]->memType = TIDLRT_MEM_SHARED;
  }
  else
  {
      in[j]->ptr =  malloc(in_tensor_szie);
  }


  out[j] = &out_tensor;
  status = TIDLRT_setTensorDefault(out[j]);
  out[j]->layout = TIDLRT_LT_NHWC;
  //strcpy((char *)in[j]->name, tensor->name);
  out[j]->elementType = TIDLRT_Float32;
  int32_t out_tensor_szie = 1001 * sizeof(float);

  if (s->device_mem)
  { 
      out[j]->ptr =  TIDLRT_allocSharedMem(64, out_tensor_szie);
      out[j]->memType = TIDLRT_MEM_SHARED;
  }
  else
  {
      out[j]->ptr =  malloc(out_tensor_szie);
  }

  status = preprocImage<uint8_t>(s->input_image_name, (uint8_t*)in[j]->ptr, 224, 224, 3, s->input_mean, s->input_std);
  LOG(INFO) << "invoked \n";

  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr);

  for (int i = 0; i < s->loop_count; i++)
  {
    TIDLRT_invoke(handle, in, out);
  }

  gettimeofday(&stop_time, nullptr);

  LOG(INFO) << "average time: "
            << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000)
            << " ms \n";
  const float threshold = 0.001f;

  std::vector<std::pair<float, int>> top_results;

  float *output = (float *)out[j]->ptr;
  int output_size = 1001;

  get_top_n<float>(output, output_size,
                    s->number_of_results, threshold, &top_results, true);

  std::vector<std::string> labels;
  size_t label_count;

  if (ReadLabelsFile(s->labels_file_name, &labels, &label_count) != 0)
    exit(-1);

  for (const auto &result : top_results)
  {
    const float confidence = result.first;
    int index = result.second;
    LOG(INFO) << confidence << ": " << index << " " << labels[index] << "\n";
  }

  status = TIDLRT_deactivate(handle);
  status = TIDLRT_delete(handle);

  if (s->device_mem)
  {
    for (uint32_t i = 0; i < 1; i++)
    {
      if (in_ptrs[i])
      {
        TIDLRT_freeSharedMem(in[i]->ptr);
      }
    }
    for (uint32_t i = 0; i < 1; i++)
    {
      if (out_ptrs[i])
      {
        TIDLRT_freeSharedMem(in[i]->ptr);
      }
    }
  }
  return 0;
}

void display_usage() {
  LOG(INFO)
      << "label_image\n"
      << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
      << "--old_accelerated, -d: [0|1], use old Android NNAPI delegate or not\n"
      << "--artifact_path, -f: [0|1], Path for Delegate artifacts folder \n"
      << "--count, -c: loop interpreter->Invoke() for certain times\n"
      << "--gl_backend, -g: use GL GPU Delegate on Android\n"
      << "--input_mean, -b: input mean\n"
      << "--input_std, -s: input standard deviation\n"
      << "--image, -i: image_name.bmp\n"
      << "--labels, -l: labels for the model\n"
      << "--tflite_model, -m: model_name.tflite\n"
      << "--profiling, -p: [0|1], profiling or not\n"
      << "--num_results, -r: number of results to show\n"
      << "--threads, -t: number of threads\n"
      << "--verbose, -v: [0|1] print more information\n"
      << "--warmup_runs, -w: number of warmup runs\n"
      << "\n";
}

int main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {"device_mem", required_argument, nullptr, 'd'},
        {"artifact_path", required_argument, nullptr, 'f'},
        {"count", required_argument, nullptr, 'c'},
        {"verbose", required_argument, nullptr, 'v'},
        {"image", required_argument, nullptr, 'i'},
        {"labels", required_argument, nullptr, 'l'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"profiling", required_argument, nullptr, 'p'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"num_results", required_argument, nullptr, 'r'},
        {"max_profiling_buffer_entries", required_argument, nullptr, 'e'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"gl_backend", required_argument, nullptr, 'g'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:d:e:f:g:i:l:m:p:r:s:t:v:w:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.accel = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'd':
        s.device_mem =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'e':
        s.max_profiling_buffer_entries =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'f':
        s.artifact_path = optarg;
        break;
      case 'g':
        s.gl_backend =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_image_name = optarg;
        break;
      case 'l':
        s.labels_file_name = optarg;
        break;
      case 'p':
        s.profiling =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'r':
        s.number_of_results =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
      default:
        exit(-1);
    }
  }
  return (RunInference(&s));
}
