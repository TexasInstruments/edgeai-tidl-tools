/*
*
* Copyright (c) {2020 - 2024} Texas Instruments Incorporated
*
* All rights reserved not granted herein.
*
* Limited License.
*
* Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
* license under copyrights and patents it now or hereafter owns or controls to make,
* have made, use, import, offer to sell and sell ("Utilize") this software subject to the
* terms herein.  With respect to the foregoing patent license, such license is granted
* solely to the extent that any such patent is necessary to Utilize the software alone.
* The patent license shall not apply to any combinations which include this software,
* other than combinations with devices manufactured by or for TI ("TI Devices").
* No hardware patent is licensed hereunder.
*
* Redistributions must preserve existing copyright notices and reproduce this license
* (including the above copyright notice and the disclaimer and (if applicable) source
* code license limitations below) in the documentation and/or other materials provided
* with the distribution
*
* Redistribution and use in binary form, without modification, are permitted provided
* that the following conditions are met:
*
* *       No reverse engineering, decompilation, or disassembly of this software is
* permitted with respect to any software provided in binary form.
*
* *       any redistribution and use are licensed by TI for use only with TI Devices.
*
* *       Nothing shall obligate TI to provide you with source code for the software
* licensed and provided to you in object code.
*
* If software source code is provided to you, modification and redistribution of the
* source code are permitted provided that the following conditions are met:
*
* *       any redistribution and use of the source code, including any resulting derivative
* works, are licensed by TI for use only with TI Devices.
*
* *       any redistribution and use of any object code compiled from the source code
* and any resulting derivative works, are licensed by TI for use only with TI Devices.
*
* Neither the name of Texas Instruments Incorporated nor the names of its suppliers
*
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* DISCLAIMER.
*
* THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
* OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
* IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
* OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
* OF THE POSSIBILITY OF SUCH DAMAGE.
*
*/

#include "tidlrt_priority_scheduling_utils.h"

using namespace std::chrono;

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

pthread_mutex_t priority_lock;
pthread_barrier_t barrier;

#define MAX_THREADS 8
#define MAX_MODELS_PER_THREAD 8

/* This struct specifies the arguments expected to be provided by user as part of the gPriorityMapping */
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

/* Information specific to each individual model being run as part of tests */
typedef struct 
{
  std::string model_name;
  int test_id;
  int thread_id;
  int model_id;
  float baseline_time_without_preemption;
  int num_iterations_run;
  float avg_time;
} model_generic_info;

/* These are arguments passed to infer function call as part of the pthread call */
typedef struct
{
  int num_models_in_thread;
  model_input_info * model_input_args[MAX_MODELS_PER_THREAD];    /* Info provided for individual model */
  Priority_settings * s;               /* common argument across threads - pass pointer */
  int is_reference_run;      /* Reference run is used to get reference output and inference runtimes */
  model_generic_info * model_info[MAX_MODELS_PER_THREAD];
} thread_arguments;


/* Structure to store results for each test to be used for further analysis */
typedef struct
{
  std::string output_test_filename[MAX_THREADS][MAX_MODELS_PER_THREAD];
  std::string output_ref_filename[MAX_THREADS][MAX_MODELS_PER_THREAD];
  int num_iterations[MAX_THREADS][MAX_MODELS_PER_THREAD];
  int functional_result[MAX_THREADS][MAX_MODELS_PER_THREAD];
  float max_pre_empt_delay[MAX_THREADS][MAX_MODELS_PER_THREAD];
  int priority[MAX_THREADS][MAX_MODELS_PER_THREAD];
  float avg_time[MAX_THREADS][MAX_MODELS_PER_THREAD];
  float baseline_time_without_preemption[MAX_THREADS][MAX_MODELS_PER_THREAD];
} aggregate_results;

/* Specify tests to be run -- Vector of tests, each test has N threads, with each thread running M models */
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
  /* Test 2 */
  {
    /* Threads*/
    {
      /* Models in each thread */
      {"model-artifacts/ss-ort-deeplabv3lite_mobilenetv2", 1, FLT_MAX, 512, 512, 3, 1, TIDLRT_Uint8, 512, 512, 1, 1, TIDLRT_Uint8}
    },
    {
      {"model-artifacts/cl-ort-resnet18-v1", 0, FLT_MAX, 224, 224, 3, 1, TIDLRT_Uint8, 1000, 1, 1, 4, TIDLRT_Float32}
    }
  },
  /* Test 3 */
  {
    /* Threads*/
    {
      /* Models in each thread */
      {"model-artifacts/ss-ort-deeplabv3lite_mobilenetv2", 1, 7, 512, 512, 3, 1, TIDLRT_Uint8, 512, 512, 1, 1, TIDLRT_Uint8}
    },
    {
      {"model-artifacts/cl-ort-resnet18-v1", 0, FLT_MAX, 224, 224, 3, 1, TIDLRT_Uint8, 1000, 1, 1, 4, TIDLRT_Float32}
    }
  },
  /* Test 4 */
  {
    /* Threads*/
    {
      /* Models in each thread */
      {"model-artifacts/ss-ort-deeplabv3lite_mobilenetv2", 1, 3, 512, 512, 3, 1, TIDLRT_Uint8, 512, 512, 1, 1, TIDLRT_Uint8}
    },
    {
      {"model-artifacts/cl-ort-resnet18-v1", 0, FLT_MAX, 224, 224, 3, 1, TIDLRT_Uint8, 1000, 1, 1, 4, TIDLRT_Float32}
    }
  },
  /* Test 5 */
  {
    /* Threads*/
    {
      /* Models in each thread */
      {"model-artifacts/ss-ort-deeplabv3lite_mobilenetv2", 1, 0.3, 512, 512, 3, 1, TIDLRT_Uint8, 512, 512, 1, 1, TIDLRT_Uint8}
    },
    {
      {"model-artifacts/cl-ort-resnet18-v1", 0, FLT_MAX, 224, 224, 3, 1, TIDLRT_Uint8, 1000, 1, 1, 4, TIDLRT_Float32}
    }
  },
  /* Test 6 */
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


/* Core inference function which does TIDLRT_Create followed by TIDLRT_invoke */
void * infer(void * argument) {

  thread_arguments *arg = (thread_arguments *)argument;
  Priority_settings *s = arg->s;
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

    pthread_mutex_lock(&priority_lock);
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
  struct timeval start_invoke, end_invoke;
  double infer_time = 0;
  std::string output_filename;

  if(arg->is_reference_run == 1)
  {
    for(int i = 0; i < num_models; i++)
    {
      float baseline_time_without_preemption = 0;
      for(int j = 0; j < s->loop_count; j++)
      {
        gettimeofday(&start_time, nullptr);
        TIDLRT_invoke(handles[i], in[i], out[i]);
        gettimeofday(&stop_time, nullptr);
        baseline_time_without_preemption += get_us(stop_time) - get_us(start_time);
      }
      arg->model_info[i]->baseline_time_without_preemption = baseline_time_without_preemption / (s->loop_count * 1000);

      LOG_INFO("Model %s :: Actual time  = %f ms \n", arg->model_info[i]->model_name.c_str(), arg->model_info[i]->baseline_time_without_preemption);

      output_filename = "examples/tidlrt_cpp/advanced_examples/outputs/output_reference_" + arg->model_info[i]->model_name + "_" + std::to_string(arg->model_info[i]->test_id) + "_" + std::to_string(arg->model_info[i]->thread_id) + "_" + std::to_string(arg->model_info[i]->thread_id) + ".bin";
    }
  }
  else if (arg->is_reference_run == 0)
  {
    for(int i = 0; i < num_models; i++)
    {
      arg->model_info[i]->num_iterations_run = 0;
      arg->model_info[i]->avg_time = 0;
      output_filename = "examples/tidlrt_cpp/advanced_examples/outputs/output_test_" + arg->model_info[i]->model_name + "_" + std::to_string(arg->model_info[i]->test_id) + "_" + std::to_string(arg->model_info[i]->thread_id) + "_" + std::to_string(arg->model_info[i]->thread_id) + ".bin";
    }

    /* Wait for all threads to synchronize before Invoke runs start across threads */
    pthread_barrier_wait(&barrier);

    gettimeofday(&start_time, nullptr);
    auto finish = system_clock::now() + minutes{s->test_duration};
    do
    {
      for(int i = 0; i < num_models; i++)
      {
        gettimeofday(&start_invoke, nullptr);
        TIDLRT_invoke(handles[i], in[i], out[i]);
        gettimeofday(&end_invoke, nullptr);
        arg->model_info[i]->num_iterations_run++;
        arg->model_info[i]->avg_time += get_us(end_invoke) - get_us(start_invoke);
      }
    } while (system_clock::now() < finish);
    gettimeofday(&stop_time, nullptr);

    for(int i = 0; i < num_models; i++) /* Average over number of iterations */
    {
      arg->model_info[i]->avg_time = arg->model_info[i]->avg_time / (arg->model_info[i]->num_iterations_run * 1000);
    }

    LOG_INFO("Model %s :: Average time with pre-emption = %f ms \n", arg->model_info[0]->model_name.c_str(), arg->model_info[0]->avg_time);
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

  void * retPtr;
  return retPtr;
}


/* TI Internal testing function - Used to analyze test results and give a PASS/FAIL result for pre-emption test */
int analyzeResults(aggregate_results * results, int test_duration)
{
  printf("\n\n ################### Results summary #######################\n\n");
  int status;
  int num_tests = gPriorityMapping.size();
  /***** Set up results table header *****/
  std::stringstream tableStream;
  std::string tableString;
  std::vector<std::string> header = {"Test id",
                                     "N1 - Pri",
                                     "N1 - Delay",
                                     "N2 - Pri",
                                     "N2 - Delay",
                                     "N1 - Iterations",
                                     "N2 - Iterations",
                                     "N1 - Functionality",
                                     "N2 - Functionality",
                                     "Test status"
                                    };
  std::vector<std::vector<std::string>> data = {};
  std::vector<TIDL_table_align_t> columnAlignment = {ALIGN_LEFT,ALIGN_LEFT,ALIGN_LEFT,ALIGN_LEFT,ALIGN_LEFT,ALIGN_LEFT,ALIGN_LEFT,ALIGN_RIGHT,ALIGN_RIGHT,ALIGN_RIGHT};

  int overall_status = STATUS_PASS; /* Status across all tests */

  float percentage_overhead[num_tests];

  /* Get baseline data for preemption testing */
  int baseline_test = -1;
  int same_priority_test = -1;
  float baseline_time_with_preemption_high_pri;
  for(int i = 0; i < num_tests; i++)
  {
    /* Test with low priority model having max preempt delay = 0 is the baseline test - almost immediate preemption */
    if(results[i].priority[0][0] == 1 && results[i].max_pre_empt_delay[0][0] == 0)
    {
      baseline_test = i;
      baseline_time_with_preemption_high_pri = results[i].avg_time[1][0];
      LOG_INFO("Baseline time with pre-emption for high priority model : %f ms\n", baseline_time_with_preemption_high_pri);
    }
    if(results[i].priority[0][0] == results[i].priority[1][0])
    {
      same_priority_test = i;
    }
  }

  /**********************************/

  /*** Testing ***/
  for(int i = 0; i < num_tests; i++)
  {
    std::vector<std::string> test_results;
    int test_status = STATUS_PASS;
    test_results.push_back(std::to_string(i + 1));
    /* Functional testing */
    int num_threads = gPriorityMapping[i].size();

    test_results.push_back(std::to_string(results[i].priority[0][0]));
    if(results[i].max_pre_empt_delay[0][0] == FLT_MAX)
    {
      test_results.push_back("FLT_MAX");
    }
    else
    {
      test_results.push_back(std::to_string(results[i].max_pre_empt_delay[0][0]));
    }
    test_results.push_back(std::to_string(results[i].priority[1][0]));
    if(results[i].max_pre_empt_delay[1][0] == FLT_MAX)
    {
      test_results.push_back("FLT_MAX");
    }
    else
    {
      test_results.push_back(std::to_string(results[i].max_pre_empt_delay[1][0]));
    }
    test_results.push_back(std::to_string(results[i].num_iterations[0][0]));
    test_results.push_back(std::to_string(results[i].num_iterations[1][0]));

    /*************************************** Functionality testing ******************************/
    /* Model 1 */
    std::string sysCmd = "diff " + results[i].output_ref_filename[0][0] + " " + results[i].output_test_filename[0][0];
    status = system(sysCmd.c_str());
    if (WIFEXITED(status))
    {
      std::string function =  WEXITSTATUS(status) == 0 ? "PASS" : "FAIL";
      test_results.push_back(function);
      if(status != 0)
      {
        test_status &= STATUS_FAIL;
      }
    }
    else
    {
      LOG_INFO("Diff returned with incorrect status for model 1");
      test_status &= STATUS_FAIL;
    }
    /* Model 2 */
    sysCmd = "diff " + results[i].output_ref_filename[1][0] + " " + results[i].output_test_filename[1][0];
    status = system(sysCmd.c_str());
    if (WIFEXITED(status))
    {
      std::string function =  WEXITSTATUS(status) == 0 ? "PASS" : "FAIL";
      test_results.push_back(function);
      if(status != 0)
      {
        test_status &= STATUS_FAIL;
      }
    }
    else
    {
      LOG_INFO("Diff returned with incorrect status for model 2");
      test_status &= STATUS_FAIL;
    }
    /***********************************************************************************/

    /************************* Pre-emption tests **************************************/
    if(i == same_priority_test)
    {
      /* Same priority models -- Maximum difference of 1 iteration since round robin scheduling */
      int diff_iterations = results[i].num_iterations[0][0] - results[i].num_iterations[1][0];
      std::vector<int> valid_diff_iterations = {-1, 0, 1};
      if(std::find(valid_diff_iterations.begin(), valid_diff_iterations.end(), diff_iterations) == valid_diff_iterations.end())
      {
         test_status &= STATUS_FAIL;
         LOG_INFO("Same priority models show a difference in num iterations which is > 1 \n");
      }
    }
    else /* Different priority models tests */
    {
      /* Tests are in decreasing order of max preempt delay, so iterations of low priority model should decrease */
      if (i != 0) /* cannot test for (i - 1) in this case */
      {
        if(! (results[i].num_iterations[0][0] < results[i - 1].num_iterations[0][0]))
        {
          test_status &= STATUS_FAIL;
          LOG_INFO("Test %d Iterations for low priority model did not decrease with decreasing max pre empt delay \n", i);
        }
      }

      if(baseline_test != -1)
      {
        /* High priority Curr test time with preemption < baseline test time with preemption + MAX(max preempt delay of all lower priority models) */ 
        int high_priority_model_idx = (results[i].priority[0][0] > results[i].priority[1][0]) ? 1 : 0; 
        if(results[i].avg_time[1][0] > baseline_time_with_preemption_high_pri + results[i].max_pre_empt_delay[0][0])
        {
          test_status &= STATUS_FAIL;
          LOG_INFO("Inference time of high priority model %f > (baseline time %f + max preempt delay of low priority model %f) \n", 
          results[i].avg_time[1][0], baseline_time_with_preemption_high_pri, results[i].max_pre_empt_delay[0][0]);
        }
      }
      else
      {
        test_status &= STATUS_FAIL;
        LOG_INFO("Baseline test with max preempt delay = 0 for low priority model missing \n");
      }
    }
    /********************************************************************/

    /**** Calculate preemption overhead ******/
    float actual_time_for_model_inference = 0;
    for(int j = 0; j < num_threads; j++)
    {
      actual_time_for_model_inference += results[i].baseline_time_without_preemption[j][0] * results[i].num_iterations[j][0];
    }
    float overhead = (test_duration * 60.0 * 1000.0 - actual_time_for_model_inference); /* in ms */
    percentage_overhead[i] = overhead / (test_duration * 60.0 * 1000.0) * 100;

    if(test_status == STATUS_PASS)
    {
      test_results.push_back("PASS");
    }
    else
    {
      test_results.push_back("FAIL");
    }

    data.push_back(test_results);
    overall_status &= test_status;
  }

  if(!data.empty())
  {
    TIDL_createTable(tableStream, header, data, 1, columnAlignment, false);
    tableString = tableStream.str();
    printf("%s\n",tableString.c_str());
  }

  printf("\n\n********** Pre-emption overhead analysis ***************\n\n");

  header = {"Test id",
            "N1 - Pri",
            "N1 - Delay",
            "N2 - Pri",
            "N2 - Delay",
            "Pre-emption %% overhead"
          };
  columnAlignment = {ALIGN_LEFT,ALIGN_LEFT,ALIGN_LEFT,ALIGN_LEFT,ALIGN_LEFT,ALIGN_RIGHT};
  data = {};
  for(int i = 0; i < num_tests; i++)
  {
    std::vector<std::string> test_results;
    int test_status = STATUS_PASS;
    test_results.push_back(std::to_string(i + 1));
    /* Functional testing */
    int num_threads = gPriorityMapping[i].size();

    test_results.push_back(std::to_string(results[i].priority[0][0]));
    if(results[i].max_pre_empt_delay[0][0] == FLT_MAX)
    {
      test_results.push_back("FLT_MAX");
    }
    else
    {
      test_results.push_back(std::to_string(results[i].max_pre_empt_delay[0][0]));
    }
    test_results.push_back(std::to_string(results[i].priority[1][0]));
    if(results[i].max_pre_empt_delay[1][0] == FLT_MAX)
    {
      test_results.push_back("FLT_MAX");
    }
    else
    {
      test_results.push_back(std::to_string(results[i].max_pre_empt_delay[1][0]));
    }
    test_results.push_back(std::to_string(percentage_overhead[i]));

    data.push_back(test_results);
  }

  if(!data.empty())
  {
    std::stringstream tableStreamOverhead;
    TIDL_createTable(tableStreamOverhead, header, data, 1, columnAlignment, false);
    tableString = tableStreamOverhead.str();
    printf("%s\n",tableString.c_str());
  }

  if(overall_status == 1)
  {
    printf("\n\nFinal test status - PASS \n\n");
  }
  else
  {
    printf("\n\nFinal test status - FAIL \n\n");
  }

  return overall_status;
}


/* Base inference function which parses tests and creates threads to run the tests */
int runInference(Priority_settings * s)
{
  int ret;
  int num_tests = gPriorityMapping.size();
  LOG_INFO("Num tests = %d \n", num_tests);
  int final_status = 1;

  ret = system("mkdir -p examples/tidlrt_cpp/advanced_examples/outputs");
  if (ret != 0)
  {
    LOG_ERROR("\n Cannot create directory examples/tidlrt_cpp/advanced_examples/outputs\n");
  }
  if (doesDirectoryExist("examples/tidlrt_cpp/advanced_examples/outputs"))
  {
    /* Command deletes the outputs, add check if directory exists to avoid any untoward rm -f happening */
    ret = system("cd examples/tidlrt_cpp/advanced_examples/outputs; rm -f *; cd - > /dev/null");
  }

  aggregate_results results[num_tests];
  
  for(int i = 0; i < num_tests; i++) /* for each test */
  {
    auto& test = gPriorityMapping[i];
    int num_threads = test.size();
    thread_arguments thread_args[MAX_THREADS];
    model_generic_info modelInfo[MAX_THREADS][MAX_MODELS_PER_THREAD];
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

        results[i].priority[j][k] = model_inputs.priority;
        results[i].max_pre_empt_delay[j][k] = model_inputs.max_pre_empt_delay;
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
    }
    pthread_attr_t tattr;
    ret = pthread_attr_init(&tattr);
    pthread_barrierattr_t barr_attr;
    ret = pthread_barrier_init(&barrier, &barr_attr, num_threads);
    if (ret != 0)
    {
        LOG_ERROR("pthread_attr_init failed \n");
    }

    pthread_t ptid[MAX_THREADS];
    LOG_INFO("************* Creating threads -- Test %d *************** \n", i+1);
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

    pthread_barrierattr_destroy(&barr_attr);
    pthread_mutex_destroy(&priority_lock);

    /* Save test run data for further analysis */
    if(s->disable_result_analysis != 1)
    {
      for (size_t j = 0; j < num_threads; j++)
      {
        for(int k = 0; k < thread_args[j].num_models_in_thread; k++)
        {
          results[i].num_iterations[j][k] = thread_args[j].model_info[k]->num_iterations_run;
          results[i].avg_time[j][k] = thread_args[j].model_info[k]->avg_time;
          results[i].baseline_time_without_preemption[j][k] = thread_args[j].model_info[k]->baseline_time_without_preemption;
          results[i].output_ref_filename[j][k] = "examples/tidlrt_cpp/advanced_examples/outputs/output_reference_" + thread_args[j].model_info[k]->model_name + "_" + std::to_string(thread_args[j].model_info[k]->test_id) 
                      + "_" + std::to_string(thread_args[j].model_info[k]->thread_id) + "_" + std::to_string(thread_args[j].model_info[k]->thread_id) + ".bin";
          results[i].output_test_filename[j][k] = "examples/tidlrt_cpp/advanced_examples/outputs/output_test_" + thread_args[j].model_info[k]->model_name + "_" + std::to_string(thread_args[j].model_info[k]->test_id) 
                      + "_" + std::to_string(thread_args[j].model_info[k]->thread_id) + "_" + std::to_string(thread_args[j].model_info[k]->thread_id) + ".bin"; 
        }
      }
    }
  }
  
  if(s->disable_result_analysis != 1)
  {
    final_status = analyzeResults(&results[0], s->test_duration);
  }
  
  return final_status;
}


void display_usage() {
  LOG(INFO)
      << "--test_duration, -t: Duration of each individual test in minutes \n"
      << "--disable_result_analysis, -r: [1/0] : Result analysis is meant for internal testing, disable for external applications \n"
      << "\n";
}

int main(int argc, char** argv) {
  Priority_settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"test_duration", required_argument, nullptr, 't'},
        {"disable_result_analysis", required_argument, nullptr, 'r'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "t:r:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 't':
        s.test_duration = strtol(optarg, nullptr, 10);
        break;
      case 'r':
        s.disable_result_analysis = strtol(optarg, nullptr, 10);
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
