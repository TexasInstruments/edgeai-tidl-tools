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

#include "tidlrt_priority_scheduling_utils.h"
using namespace std::chrono;


std::string removeAnsi(std::string str)
{
  std::regex re("\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])");
  std::stringstream result;
  std::regex_replace(std::ostream_iterator<char>(result), str.begin(), str.end(), re, "");
  return result.str();
}

void TIDL_createTable(std::ostream &stream,
                      std::vector<std::string> header,
                      std::vector<std::vector<std::string>> data,
                      int32_t padding,
                      std::vector<TIDL_table_align_t> columnAlignment,
                      bool printSeperator)
{
  int32_t numColumn = header.size();
  if(numColumn <= 0)
  {
    return;
  }
  for (auto &d: data)
  {
    if(d.size() != numColumn)
    {
      printf("No of headers columns - %d does not match number of data column - %d\n", numColumn, d.size());
      return;
    }
  }

  bool useDynamicColumnResizing=false;
  int32_t i;
  std::vector<size_t> columnSizes;
  columnSizes.resize(numColumn);


  // Initialize each column width with header text sizes.
  for (i = 0; i < numColumn; i++)
  {
    std::string text = removeAnsi(header[i]);
    columnSizes[i] = text.length();
  }

  // Get width for each column according to largest string present
  for (auto &d: data)
  {
    for(i = 0; i < d.size(); i++)
    {
      std::string text = removeAnsi(d[i]);
      if(text.length() > columnSizes[i])
      {
        columnSizes[i] = text.length();
      }
    }
  }

  for (i = 0; i < columnSizes.size(); i++)
  {
    if(columnSizes[i] > (((80-(numColumn + 1))/numColumn) - (2*padding)))
    {
      useDynamicColumnResizing = true;
      break;
    }
  }

  if(!useDynamicColumnResizing)
  {
    for (i = 0; i < columnSizes.size(); i++)
    {
      columnSizes[i] = ((80-(numColumn + 1))/numColumn) - (2*padding);
    }
  }

  std::string paddingString(padding, ' ');

  int32_t totalWidth = 0;
  for (i = 0; i < columnSizes.size(); i++)
  {
    totalWidth += columnSizes[i] + (2 * padding);
  }
  totalWidth += numColumn + 1;
  std::string dashLine(totalWidth, '-');

  // Print the top of the table
  stream << dashLine << "\n";

  // Print center aligned header
  stream << "|";
  for (i = 0; i < numColumn; i++)
  {
    std::string text = removeAnsi(header[i]);
    int32_t ansiExtra = header[i].length() - text.length();
    int32_t half = (columnSizes[i] / 2) - (text.length() / 2);
    if(half < 0)
    {
      half = 0;
    }
    stream << paddingString << std::setw(columnSizes[i]+ansiExtra) << std::left
              << std::string(half, ' ') + header[i] << paddingString << "|";
  }

  stream << "\n";

  // Print dash line at bottom of header
  stream << dashLine << "\n";

  if(data.size() <= 0)
  {
    return;
  }

  int32_t cnt = 0;
  // Print data
  for (auto &d : data)
  {
    stream << "|";

    // Set precision in case of float values;
    stream << std::fixed << std::setprecision(2);

    for (i = 0; i < d.size(); i++)
    {
        auto alignment = std::left;
        int32_t half = 0;
        std::string text = removeAnsi(d[i]);
        int32_t ansiExtra = d[i].length() - text.length();
        if(i < columnAlignment.size())
        {
            alignment = columnAlignment[i] == ALIGN_RIGHT ? std::right : std::left;
            if(columnAlignment[i] == ALIGN_CENTER)
            {
                half = (columnSizes[i] / 2) - (text.length() / 2);
                if(half < 0)
                {
                  half = 0;
                }
            }
        }
        stream << paddingString << std::setw(columnSizes[i]+ansiExtra) << alignment
                  << std::string(half, ' ') + d[i] << paddingString << "|";
    }
    stream << "\n";
    if(printSeperator && cnt++ < data.size() - 1)
    {
      stream << dashLine << "\n";
    }
  }

  stream << dashLine;
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

