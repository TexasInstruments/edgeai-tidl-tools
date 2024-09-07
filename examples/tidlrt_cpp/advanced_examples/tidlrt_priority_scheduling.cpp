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
#include <fcntl.h>    
#include <getopt.h>   
#include <sys/time.h> 
#include <sys/types.h>
#include <sys/uio.h>  
#include <unistd.h>   

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
#include <fstream>


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
#include "tidlrt_priority_scheduling.h"

#include "../../osrt_cpp/utils/include/ti_logger.h"
#include "../../osrt_cpp/utils/include/utility_functs.h"

using namespace std::chrono;

#define LOG(x) std::cerr

void* in_ptrs[16] = {NULL};
void* out_ptrs[16]= {NULL};

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

pthread_mutex_t priority_lock;
pthread_barrier_t barrier;

#define NUM_PRIORITIES 2
#define MAX_THREADS 8


#define MAX_MODELS_PER_THREAD 8

/* This struct specifies the arguments expected to be provided by user */
typedef struct
{
    std::string model_artifacts_path;
    int priority;
    float max_pre_empt_delay;
    int in_width;
    int in_height;
    int in_numCh;
    int in_element_size_in_bytes;
    int in_element_type;
    int out_width;
    int out_height;
    int out_numCh;
    int out_element_size_in_bytes;
    int out_element_type;
} model_input_info;

typedef struct 
{
  std::string model_name;
  int test_id;
  int thread_id;
  int model_id;
  float actual_times;
  int num_iterations_run;
  float avg_time;
} model_generic_info;

/* These are arguments passed to infer function call as part of the pthread call */
typedef struct
{
  int num_models_in_thread;
  model_input_info * model_input_args[MAX_MODELS_PER_THREAD];    /* Info provided for individual model */
  Settings * s;               /* common argument across threads - pass pointer */
  int is_reference_run;
  model_generic_info * model_info[MAX_MODELS_PER_THREAD];
} thread_arguments;

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

void * infer(void * argument) {

  thread_arguments *arg = (thread_arguments *)argument;
  Settings *s = arg->s;
  int num_models = arg->num_models_in_thread;

  void * handles[MAX_MODELS_PER_THREAD];
  sTIDLRT_Tensor_t *in[MAX_MODELS_PER_THREAD][16];
  sTIDLRT_Tensor_t *out[MAX_MODELS_PER_THREAD][16];
  int out_tensor_sizes[MAX_MODELS_PER_THREAD][16];

  int32_t status;

  for(int i = 0; i < num_models; i++) /* Loop for creation of all models */
  {
    model_generic_info * model_info = arg->model_info[i];
    model_input_info * model_input_args = arg->model_input_args[i];

    std::string artifacts_path = model_input_args->model_artifacts_path;
    
    char net_name[512];
    char io_name[512];

    getModelNameromArtifactsDir((char *)artifacts_path.c_str(), net_name, io_name);

    sTIDLRT_Params_t prms;
    void *handle = NULL;

    status = TIDLRT_setParamsDefault(&prms);

    FILE * fp_network = fopen(net_name, "rb");
    if (fp_network == NULL)
    {
      printf("Invoke  : ERROR: Unable to open network file %s \n", net_name);
      // return -1;
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
      // return -1;
    }
    fseek(fp_config, 0, SEEK_END);
    prms.io_capacity = ftell(fp_config);
    fseek(fp_config, 0, SEEK_SET);
    fclose(fp_config);
    prms.ioBufDescPtr = malloc(prms.io_capacity);
    status = TIDLReadBinFromFile(io_name, prms.ioBufDescPtr, prms.io_capacity);

    prms.traceLogLevel = 0;
    prms.traceWriteLevel = 0;

    prms.targetPriority = model_input_args->priority;
    prms.maxPreEmptDelay = model_input_args->max_pre_empt_delay;
    prms.coreNum = 1;

    pthread_mutex_lock(&priority_lock); /*Remove this and test */
    status = TIDLRT_create(&prms, &handle);
    handles[i] = handle;
    pthread_mutex_unlock(&priority_lock);

    sTIDLRT_Tensor_t in_tensor;
    sTIDLRT_Tensor_t out_tensor;

    int32_t j = 0; /* Currently implemented only for models with 1 input and 1 output */
    in[i][j] = &in_tensor;
    status = TIDLRT_setTensorDefault(in[i][j]);
    in[i][j]->layout = TIDLRT_LT_NCHW;
    in[i][j]->elementType = TIDLRT_Uint8;
    int32_t in_tensor_size = model_input_args->in_width * model_input_args->in_height * model_input_args->in_numCh * model_input_args->in_element_size_in_bytes;

    in[i][j]->ptr =  TIDLRT_allocSharedMem(64, in_tensor_size);
    in[i][j]->memType = TIDLRT_MEM_SHARED;

    out[i][j] = &out_tensor;
    status = TIDLRT_setTensorDefault(out[i][j]);
    out[i][j]->layout = TIDLRT_LT_NCHW;
    out[i][j]->elementType = model_input_args->out_element_type;

    int32_t out_tensor_size = model_input_args->out_width * model_input_args->out_height * model_input_args->out_numCh * model_input_args->out_element_size_in_bytes;
    out_tensor_sizes[i][0] = out_tensor_size;
    out[i][j]->ptr =  TIDLRT_allocSharedMem(64, out_tensor_size);
    out[i][j]->memType = TIDLRT_MEM_SHARED;
    
    /* Use random number generator with a seed to create input */
    unsigned int seed = model_info->model_id;
    int min = 0;
    int max = 255;
    char * inPtr = (char *)(in[i][j]->ptr);
    for(int m = 0; m < in_tensor_size; m++)
    {
      inPtr[m] = rand_r(&seed) % (max - min + 1) + min;
    }
  }

  struct timeval start_time, stop_time;
  std::string output_filename;

  if(arg->is_reference_run == 1)
  {
    for(int i = 0; i < num_models; i++)
    {
      gettimeofday(&start_time, nullptr);
      for(int j = 0; j < s->loop_count; j++)
      {
        TIDLRT_invoke(handles[i], in[i], out[i]);
      }
      gettimeofday(&stop_time, nullptr);
      arg->model_info[i]->actual_times = (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000);

      LOG_INFO("Model %s :: Actual time  = %f ms \n", arg->model_info[i]->model_name.c_str(), (get_us(stop_time) - get_us(start_time)) / (arg->s->loop_count * 1000));

      output_filename = "output_reference_" + arg->model_info[i]->model_name + "_" + std::to_string(arg->model_info[i]->test_id) + "_" + std::to_string(arg->model_info[i]->thread_id) + "_" + std::to_string(arg->model_info[i]->thread_id) + ".bin";
    }
  }
  else if (arg->is_reference_run == 0)
  {
    for(int i = 0; i < num_models; i++)
    {
      arg->model_info[i]->num_iterations_run = 0;
      output_filename = "output_test_" + arg->model_info[i]->model_name + "_" + std::to_string(arg->model_info[i]->test_id) + "_" + std::to_string(arg->model_info[i]->thread_id) + "_" + std::to_string(arg->model_info[i]->thread_id) + ".bin";
    }

    gettimeofday(&start_time, nullptr);
    auto finish = system_clock::now() + minutes{1};
    do
    {
      for(int i = 0; i < num_models; i++)
      {
        TIDLRT_invoke(handles[i], in[i], out[i]);
        arg->model_info[i]->num_iterations_run++;
      }
    } while (system_clock::now() < finish);
    gettimeofday(&stop_time, nullptr);

    LOG_INFO("Model %s :: Average time with pre-emption = %f ms \n", arg->model_info[0]->model_name.c_str(), (get_us(stop_time) - get_us(start_time)) / (arg->model_info[0]->num_iterations_run * 1000));
    LOG_INFO("Model %s :: Total number of iterations run = %d \n", arg->model_info[0]->model_name.c_str(), arg->model_info[0]->num_iterations_run);
  }

  int j = 0;
  for(int i = 0; i < num_models; i++)
  {
    char * outPtr = (char *)out[i][j]->ptr;
    std::ofstream fs(output_filename, std::ios::out | std::ios::binary | std::ios::out);
    fs.write(outPtr, out_tensor_sizes[i][0]);
    fs.close();

    status = TIDLRT_deactivate(handles[i]);
    status = TIDLRT_delete(handles[i]);
  }

  for (uint32_t i = 0; i < num_models; i++)
  {
    if (in[i][0]->ptr)
    {
      // TIDLRT_freeSharedMem(in[i][0]->ptr);
    }
    if (out[i][0]->ptr)
    {
      // TIDLRT_freeSharedMem(out[i][0]->ptr);
    }
  }

  void * retPtr;
  return retPtr;
}

std::vector<std::vector<std::vector<model_input_info>>> gPriorityMapping = 
{
  /* Test 1 */
  {
    /* Threads*/
    {
      /* Models in each thread */
      {"model-artifacts/ss-ort-deeplabv3lite_mobilenetv2", 0, FLT_MAX, 512, 512, 3, 1, TIDLRT_Uint8, 512, 512, 1, 1, TIDLRT_Uint8}
    },
    {
      {"model-artifacts/cl-ort-resnet18-v1", 0, FLT_MAX, 224, 224, 3, 1, TIDLRT_Uint8, 1000, 1, 1, 4, TIDLRT_Float32}
    }
  },
  {
    /* Threads*/
    {
      /* Models in each thread */
      {"model-artifacts/ss-ort-deeplabv3lite_mobilenetv2", 1, 0, 512, 512, 3, 1, TIDLRT_Uint8, 512, 512, 1, 1, TIDLRT_Uint8}
    },
    {
      {"model-artifacts/cl-ort-resnet18-v1", 0, FLT_MAX, 224, 224, 3, 1, TIDLRT_Uint8, 1000, 1, 1, 4, TIDLRT_Float32}
    }
  }

};

int runInference(Settings * s)
{
  s->number_of_threads = 1;
  s->loop_count = 10;

  int ret;
  int num_tests = gPriorityMapping.size();
  LOG_INFO("Num tests = %d \n", num_tests);
  
  for(int i = 0; i < num_tests; i++) /* for each test */
  {
    auto& test = gPriorityMapping[i];
    int num_threads = test.size();
    thread_arguments thread_args[MAX_THREADS];
    model_generic_info modelInfo[MAX_THREADS][MAX_MODELS_PER_THREAD];
    std::vector<thread_arguments *> thread_arguments_all;
    for(int j = 0; j < num_threads; j++) /* For each thread in test */
    {
      auto& thread_info = test[j];
      for(int k = 0; k < thread_info.size(); k++) /* For each model in thread */
      {
        auto& model_inputs = thread_info[k];
        modelInfo[j][k].test_id = i;
        modelInfo[j][k].thread_id = j;
        modelInfo[j][k].model_id = k;
        /* Populate model name */
        std::string modelName = model_inputs.model_artifacts_path;
        size_t sep = modelName.find_last_of("\\/");
        if (sep != std::string::npos)
            modelName = modelName.substr(sep + 1, modelName.size() - sep - 1);
        modelInfo[j][k].model_name = modelName;
        /***************/

        thread_args[j].model_input_args[k] = &model_inputs;
        thread_args[j].model_info[k] = &modelInfo[j][0];
      }
      thread_args[j].num_models_in_thread = thread_info.size();
      thread_args[j].s = s;
      /* Run to get reference run results - base output and base inference time */
      thread_args[j].is_reference_run = 1;
      infer(&thread_args[j]);
      thread_args[j].is_reference_run = 0;

    }

    /* thread spawning and running inference in parallel */
    if (pthread_mutex_init(&priority_lock, NULL) != 0)
    {
        LOG_ERROR("\n mutex init has failed\n");
        // return RETURN_FAIL;
    }
    pthread_attr_t tattr;
    ret = pthread_attr_init(&tattr);
    if (ret != 0)
    {
        LOG_ERROR("barrier creation failed exiting\n");
        // return RETURN_FAIL;
    }

    pthread_t ptid[MAX_THREADS];
    LOG_INFO("************* Creating threads *************** \n");
    for (size_t i = 0; i < num_threads; i++)
    {
        /* Creating a new thread*/
        pthread_create(&ptid[i], &tattr, &infer, &thread_args[i]);
    }
    for (size_t i = 0; i < num_threads; i++)
    {
        // Waiting for the created thread to terminate
        pthread_join(ptid[i], NULL);
    }

    pthread_mutex_destroy(&priority_lock);
  }
  return RETURN_SUCCESS;
}
#if 0
/* Primitive code for just 2 models with different priorities and single test at a time */

typedef struct
{
    std::string model_artifacts_path;
    int priority;
    float max_pre_empt_delay;
    int model_id;
    Settings *s;
    int is_reference_run;
    std::string model_name;
    int in_width;
    int in_height;
    int in_numCh;
    int in_element_size_in_bytes;
    int in_element_type;
    int out_width;
    int out_height;
    int out_numCh;
    int out_element_size_in_bytes;
    int out_element_type;
    float actual_times;
} model_struct;

int getActualRunTime(model_struct *arg0, model_struct *arg1)
{
  LOG_INFO("Inferring model 1 \n");
  infer(arg0);
  LOG_INFO("Inferring model 2 \n");
  infer(arg1);
  LOG_INFO("Actual run times of models are : \n %s -- %f \n %s -- %f \n", arg0->model_name.c_str(), arg0->actual_times, arg1->model_name.c_str(),arg1->actual_times);
  return RETURN_SUCCESS;   
}

int runInference(Settings * s)
{
    model_struct args[NUM_PRIORITIES];
    int ret;
    for (size_t i = 0; i < NUM_PRIORITIES; i++)
    {
        LOG_INFO("Prep model %d\n", i);
        args[i].model_id = i;
        args[i].s = s;
        if(i == 0)
        {
          args[i].model_artifacts_path = "model-artifacts/ss-ort-deeplabv3lite_mobilenetv2";
          args[i].priority = 1;
          args[i].max_pre_empt_delay = 0; 
          /* Dims */
          args[i].in_width = 512;
          args[i].in_height = 512;
          args[i].in_numCh = 3;
          args[i].in_element_size_in_bytes = 1;
          args[i].in_element_type = TIDLRT_Uint8;
          args[i].out_width = 512;
          args[i].out_height = 512;
          args[i].out_numCh = 1;
          args[i].out_element_size_in_bytes = 1;
          args[i].out_element_type = TIDLRT_Uint8;
        }
        if(i == 1)
        {
          args[i].model_artifacts_path = "model-artifacts/cl-ort-resnet18-v1";
          args[i].priority = 0;
          args[i].max_pre_empt_delay = FLT_MAX;
          /* Dims */
          args[i].in_width = 224;
          args[i].in_height = 224;
          args[i].in_numCh = 3;
          args[i].in_element_size_in_bytes = 1;
          args[i].in_element_type = TIDLRT_Uint8;
          args[i].out_width = 1000;
          args[i].out_height = 1;
          args[i].out_numCh = 1;
          args[i].out_element_size_in_bytes = 4;
          args[i].out_element_type = TIDLRT_Float32;
        }
        std::string modelName = args[i].model_artifacts_path;
        size_t sep = modelName.find_last_of("\\/");
        if (sep != std::string::npos)
            modelName = modelName.substr(sep + 1, modelName.size() - sep - 1);
        args[i].model_name = modelName;
    }

    s->number_of_threads = 1;
    s->loop_count = 10;

    /* Reference run starts */
    
    args[0].is_reference_run = 1;
    args[1].is_reference_run = 1;

    if (RETURN_FAIL == getActualRunTime(&args[0], &args[1]))
            return RETURN_FAIL;

    if (pthread_mutex_init(&priority_lock, NULL) != 0)
    {
        LOG_ERROR("\n mutex init has failed\n");
        return RETURN_FAIL;
    }
    pthread_attr_t tattr;
    ret = pthread_attr_init(&tattr);
    pthread_barrierattr_t barr_attr;
    ret = pthread_barrier_init(&barrier, &barr_attr, (2 * s->number_of_threads));
    if (ret != 0)
    {
        LOG_ERROR("barrier creation failied exiting\n");
        return RETURN_FAIL;
    }

    /* Test run starts */
    args[0].is_reference_run = 0;
    args[1].is_reference_run = 0;

    pthread_t ptid[2 * NUM_PRIORITIES];
    LOG_INFO("************* Creating threads *************** \n");
    for (size_t i = 0; i < s->number_of_threads; i++)
    {
        /* Creating a new thread*/
        pthread_create(&ptid[2 * i], &tattr, &infer, &args[0]);
        pthread_create(&ptid[2 * i + 1], &tattr, &infer, &args[1]);
    }
    for (size_t i = 0; i < s->number_of_threads; i++)
    {
        // Waiting for the created thread to terminate
        pthread_join(ptid[2 * i], NULL);
        pthread_join(ptid[2 * i + 1], NULL);
    }

    pthread_barrierattr_destroy(&barr_attr);
    pthread_mutex_destroy(&priority_lock);
    return RETURN_SUCCESS;
}
#endif

/* Options are kept same as base application --- can be updated to take model_struct arguments */
void display_usage() {
  LOG(INFO)
      << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
      << "--threads, -t: number of threads\n"
      << "\n";
}

int main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {"threads", required_argument, nullptr, 't'},
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
        s.accel = strtol(optarg, nullptr, 10);
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
  
  return (runInference(&s));
}
