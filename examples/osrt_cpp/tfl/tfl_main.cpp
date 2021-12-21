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

#include "tfl_main.h"

namespace tflite
{
  namespace main
  {

    void *in_ptrs[16] = {NULL};
    void *out_ptrs[16] = {NULL};

    /**
     *  \brief  prepare the segemntation result inplace
     *  \param  img cv image to do inplace transform
     *  \param  wanted_width
     *  \param  wanted_height
     *  \param  alpha
     *  \param  interpreter pointer of tflite
     *  \param  outputs pointer of output vector
     * @returns int status
     */
    int prepSegResult(cv::Mat *img, int wanted_width, int wanted_height, float alpha,
                      std::unique_ptr<tflite::Interpreter> *interpreter, const std::vector<int> *outputs)
    {
      LOG_INFO("preparing segmentation result \n");
      TfLiteType type = (*interpreter)->tensor((*outputs)[0])->type;
      if (type == TfLiteType::kTfLiteInt32)
      {
        int32_t *outputTensor = (*interpreter)->tensor((*outputs)[0])->data.i32;
        (*img).data = blendSegMask<int32_t>((*img).data, outputTensor, (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
      }
      else if (type == TfLiteType::kTfLiteInt64)
      {
        int64_t *outputTensor = (*interpreter)->tensor((*outputs)[0])->data.i64;
        (*img).data = blendSegMask<int64_t>((*img).data, outputTensor, (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
      }
      else if (type == TfLiteType::kTfLiteFloat32)
      {
        float *outputTensor = (*interpreter)->tensor((*outputs)[0])->data.f;
        (*img).data = blendSegMask<float>((*img).data, outputTensor, (*img).cols, (*img).rows, wanted_width, wanted_height, alpha);
      }else{
        LOG_ERROR("op tensor tyrp not supprted\n");
        return RETURN_FAIL;
      }
      return RETURN_SUCCESS;
    }

    /**
     *  \brief  prepare the classification result inplace
     *  \param  img cv image to do inplace transform
     *  \param  interpreter pointer of tflite
     *  \param  outputs pointer of output vector
     *  \param  s settings
     * @returns int status
     */
    int prepClassificationResult(cv::Mat *img, std::unique_ptr<tflite::Interpreter> *interpreter,
                                 const std::vector<int> *outputs, Settings *s)
    {
      LOG_INFO("preparing clasification result \n");
      const float threshold = 0.001f;
      std::vector<std::pair<float, int>> top_results;

      TfLiteIntArray *output_dims = (*interpreter)->tensor((*outputs)[0])->dims;
      /* assume output dims to be something like (1, 1, ... ,size) */
      auto output_size = output_dims->data[output_dims->size - 1];
      int outputoffset;
      if (output_size == 1001)
        outputoffset = 0;
      else
        outputoffset = 1;
      switch ((*interpreter)->tensor((*outputs)[0])->type)
      {
      case kTfLiteFloat32:
        getTopN<float>((*interpreter)->typed_output_tensor<float>(0), output_size,
                       s->number_of_results, threshold, &top_results, true);
        break;
      case kTfLiteUInt8:
        getTopN<uint8_t>((*interpreter)->typed_output_tensor<uint8_t>(0),
                         output_size, s->number_of_results, threshold,
                         &top_results, false);
        break;
      default:
        LOG_ERROR("cannot handle output type %d yet", (*interpreter)->tensor((*outputs)[0])->type);
        return RETURN_FAIL;
      }

      std::vector<string> labels;
      size_t label_count;

      if (readLabelsFile(s->labels_file_path, &labels, &label_count) != 0)
      {
        LOG_ERROR("label file not found!!! \n");
        return RETURN_FAIL;
      }

      for (const auto &result : top_results)
      {
        const float confidence = result.first;
        const int index = result.second;
        LOG_INFO("%f: %d :%s\n", confidence, index, labels[index + outputoffset].c_str());
      }
      int num_results = 5;
      (*img).data = overlayTopNClasses((*img).data, top_results, &labels, (*img).cols, (*img).rows, num_results);
      return RETURN_SUCCESS;
    }

    /**
     *  \brief  Actual infernce happening
     *  \param  ModelInfo YAML parsed model info
     *  \param  Settings user input options  and default values of setting if any
     * @returns int
     */
    int runInference(ModelInfo *modelInfo, Settings *s)
    {
      /* checking model path present or not*/
      if (!modelInfo->m_infConfig.modelFile.c_str())
      {
        LOG_ERROR("no model file name\n");
        return RETURN_FAIL;
      }
      /* preparing tflite model  from file*/
      std::unique_ptr<tflite::FlatBufferModel> model;
      std::unique_ptr<tflite::Interpreter> interpreter;
      model = tflite::FlatBufferModel::BuildFromFile(modelInfo->m_infConfig.modelFile.c_str());
      if (!model)
      {
        LOG_ERROR("\nFailed to mmap model %s\n", modelInfo->m_infConfig.modelFile);
        return RETURN_FAIL;
      }
      LOG_INFO("Loaded model %s \n", modelInfo->m_infConfig.modelFile.c_str());
      model->error_reporter();
      LOG_INFO("resolved reporter\n");

      tflite::ops::builtin::BuiltinOpResolver resolver;
      tflite::InterpreterBuilder(*model, resolver)(&interpreter);
      if (!interpreter)
      {
        LOG_ERROR("Failed to construct interpreter\n");
        return RETURN_FAIL;
      }
      const std::vector<int> inputs = interpreter->inputs();
      const std::vector<int> outputs = interpreter->outputs();

      LOG_INFO("tensors size: %d \n", interpreter->tensors_size());
      LOG_INFO("nodes size: %d\n", interpreter->nodes_size());
      LOG_INFO("number of inputs: %d\n", inputs.size());
      LOG_INFO("number of outputs: %d\n", outputs.size());
      LOG_INFO("input(0) name: %s\n", interpreter->GetInputName(0));

      if (inputs.size() != 1)
      {
        LOG_ERROR("Supports only single input models \n");
        return RETURN_FAIL;
      }

      if (s->log_level <= DEBUG)
      {
        int t_size = interpreter->tensors_size();
        for (int i = 0; i < t_size; i++)
        {
          if (interpreter->tensor(i)->name)
            LOG_INFO("%d: %s,%d,%d,%d,%d\n", i, interpreter->tensor(i)->name,
                     interpreter->tensor(i)->bytes,
                     interpreter->tensor(i)->type,
                     interpreter->tensor(i)->params.scale,
                     interpreter->tensor(i)->params.zero_point);
        }
      }

      if (s->number_of_threads != -1)
      {
        interpreter->SetNumThreads(s->number_of_threads);
      }

      int input = inputs[0];
      if (s->log_level <= INFO)
        LOG_INFO("input: %d\n", input);

      if (s->accel == 1)
      {
        /* This part creates the dlg_ptr */
        LOG_INFO("accelerated mode\n");
        typedef TfLiteDelegate *(*tflite_plugin_create_delegate)(char **, char **, size_t, void (*report_error)(const char *));
        tflite_plugin_create_delegate tflite_plugin_dlg_create;
        char *keys[] = {"artifacts_folder", "num_tidl_subgraphs", "debug_level"};
        char *values[] = {(char *)modelInfo->m_infConfig.artifactsPath.c_str(), "16", "0"};
        void *lib = dlopen("libtidl_tfl_delegate.so", RTLD_NOW);
        assert(lib);
        tflite_plugin_dlg_create = (tflite_plugin_create_delegate)dlsym(lib, "tflite_plugin_create_delegate");
        TfLiteDelegate *dlg_ptr = tflite_plugin_dlg_create(keys, values, 3, NULL);
        interpreter->ModifyGraphWithDelegate(dlg_ptr);
        LOG_INFO("ModifyGraphWithDelegate - Done \n");
      }
      if (interpreter->AllocateTensors() != kTfLiteOk)
      {
        LOG_ERROR("Failed to allocate tensors!");
        return RETURN_FAIL;
      }

      if (s->device_mem)
      {
        LOG_INFO("device mem enabled\n");
        for (uint32_t i = 0; i < inputs.size(); i++)
        {
          const TfLiteTensor *tensor = interpreter->input_tensor(i);
          in_ptrs[i] = TIDLRT_allocSharedMem(tflite::kDefaultTensorAlignment, tensor->bytes);
          if (in_ptrs[i] == NULL)
          {
            LOG_INFO("Could not allocate Memory for input: %s\n", tensor->name);
          }
          interpreter->SetCustomAllocationForTensor(inputs[i], {in_ptrs[i], tensor->bytes});
        }
        for (uint32_t i = 0; i < outputs.size(); i++)
        {
          const TfLiteTensor *tensor = interpreter->output_tensor(i);
          out_ptrs[i] = TIDLRT_allocSharedMem(tflite::kDefaultTensorAlignment, tensor->bytes);
          if (out_ptrs[i] == NULL)
          {
            LOG_INFO("Could not allocate Memory for ouput: %s\n", tensor->name);
          }
          interpreter->SetCustomAllocationForTensor(outputs[i], {out_ptrs[i], tensor->bytes});
        }
      }

      if (s->log_level <= DEBUG)
        PrintInterpreterState(interpreter.get());
      /* get input dimension from the YAML parsed  and batch
      from input tensor assuming one tensor*/
      TfLiteIntArray *dims = interpreter->tensor(input)->dims;
      int wanted_batch = dims->data[0];
      int wanted_height = modelInfo->m_preProcCfg.outDataHeight;
      int wanted_width = modelInfo->m_preProcCfg.outDataWidth;
      int wanted_channels = modelInfo->m_preProcCfg.numChans;
      /* assuming NHWC*/
      if (wanted_channels != dims->data[3])
      {
        LOG_INFO("missmatch in YAML parsed wanted channels:%d and model channels:%d\n", wanted_channels, dims->data[3]);
      }
      if (wanted_height != dims->data[1])
      {
        LOG_INFO("missmatch in YAML parsed wanted height:%d and model height:%d\n", wanted_height, dims->data[1]);
      }
      if (wanted_width != dims->data[2])
      {
        LOG_INFO("missmatch in YAML parsed wanted width:%d and model width:%d\n", wanted_width, dims->data[2]);
      }
      cv::Mat img;
      switch (interpreter->tensor(input)->type)
      {
      case kTfLiteFloat32:
      {
        img = preprocImage<float>(s->input_bmp_path, &interpreter->typed_tensor<float>(input)[0], modelInfo->m_preProcCfg);
        break;
      }
      case kTfLiteUInt8:
      {
        /* if model is already quantized update the scale and mean for
        preperocess computation */
        std::vector<float> temp_scale = modelInfo->m_preProcCfg.scale,
                           temp_mean = modelInfo->m_preProcCfg.mean;
        modelInfo->m_preProcCfg.scale = {1, 1, 1};
        modelInfo->m_preProcCfg.mean = {0, 0, 0};
        img = preprocImage<uint8_t>(s->input_bmp_path, &interpreter->typed_tensor<uint8_t>(input)[0], modelInfo->m_preProcCfg);
        /*restoring mean and scale to preserve the data */
        modelInfo->m_preProcCfg.scale = temp_scale;
        modelInfo->m_preProcCfg.mean = temp_mean;
        break;
      }
      default:
        LOG_ERROR("cannot handle input type %d yet\n", interpreter->tensor(input)->type);
        return RETURN_FAIL;
      }

      LOG_INFO("interpreter->Invoke - Started \n");
      if (s->loop_count > 1)
        for (int i = 0; i < s->number_of_warmup_runs; i++)
        {
          if (interpreter->Invoke() != kTfLiteOk)
          {
            LOG_ERROR("Failed to invoke tflite!\n");
          }
        }

      struct timeval start_time, stop_time;
      gettimeofday(&start_time, nullptr);
      for (int i = 0; i < s->loop_count; i++)
      {
        if (interpreter->Invoke() != kTfLiteOk)
        {
          LOG_ERROR("Failed to invoke tflite!\n");
        }
      }
      gettimeofday(&stop_time, nullptr);
      LOG_INFO("interpreter->Invoke - Done \n");
      float avg_time = (getUs(stop_time) - getUs(start_time)) / (s->loop_count * 1000);
      LOG_INFO("average time:%f ms\n",avg_time);

      if (modelInfo->m_preProcCfg.taskType == "classification")
      {
        if (RETURN_FAIL == prepClassificationResult(&img, &interpreter, &outputs, s))
          return RETURN_FAIL;
      }
      else if (modelInfo->m_preProcCfg.taskType == "detection")
      {
        /*store tensor_shape info of op tensors in arr
               to avaoid recalculation*/
        int num_ops = outputs.size();
        vector<vector<float>> f_tensor_unformatted;
        /*num of detection in op tensor is assumed to be given by last tensor*/
        int nboxes;
        if(interpreter->tensor(outputs[num_ops-1])->type == kTfLiteFloat32)
          nboxes = (int)*interpreter->tensor(outputs[num_ops-1])->data.f;
        else if(interpreter->tensor(outputs[num_ops-1])->type == kTfLiteInt64)
          nboxes = (int)*interpreter->tensor(outputs[num_ops-1])->data.i64;
        else{
          LOG_ERROR("unknown type for op tensor:%d\n",num_ops-1);
          return RETURN_FAIL;
        }
        LOG_INFO("detected objects:%d \n",nboxes);
        /* TODO verify this holds true for every tfl model*/
        vector<vector<int64_t>> tensor_shapes_vec = {{nboxes,4},{nboxes,1},{nboxes,1},{nboxes,1}};
        /* TODO Incase of only single tensor op od-2110 above tensor shape is 
        invalid*/

        /* run through all tensors excpet last one which contain
        num_of detected boxes */
        for (size_t i = 0; i < num_ops-1; i++)
        {
          /* temp vector to store converted ith tensor */
          vector<float> f_tensor;
          /* shape of the ith tensor*/
          vector<int64_t> tensor_shape = tensor_shapes_vec[i];
          
          /* type of the ith tensor*/
          TfLiteType tensor_type = interpreter->tensor(outputs[i])->type;
          /* num of values in ith tensor is assumed to be the tensor's 
          shape in tflite*/
          int num_val_tensor = tensor_shape[tensor_shape.size()-1];
          /*convert tensor to float vector*/
          if (tensor_type == kTfLiteFloat32)
          {
            float *inDdata = interpreter->tensor(outputs[i])->data.f;
            createFloatVec<float>(inDdata, &f_tensor, tensor_shape);
          }
          else if (tensor_type == kTfLiteInt64)
          {
            int64_t *inDdata = interpreter->tensor(outputs[i])->data.i64;
            createFloatVec<int64_t>(inDdata, &f_tensor, tensor_shape);
          }
          else if (tensor_type == kTfLiteInt32)
          {
            int32_t *inDdata = (int32_t*)interpreter->tensor(outputs[i])->data.data;
            createFloatVec<int32_t>(inDdata, &f_tensor, tensor_shape);
          }
          else
          {
            LOG_ERROR("out tensor data type not supported %d\n", tensor_type);
            return RETURN_FAIL;
          }
          /*append all output tensors in to single vector<vector<float>*/
          for (size_t j = 0; j < nboxes; j++)
          {
            vector<float> temp;
            for (size_t k = 0; k < num_val_tensor; k++)
            {
              temp.push_back(f_tensor[j * num_val_tensor + k]);
            }
            f_tensor_unformatted.push_back(temp);
          }
        }
        if (RETURN_FAIL == prepDetectionResult(&img, &f_tensor_unformatted, tensor_shapes_vec, modelInfo, num_ops-1,nboxes))
          return RETURN_FAIL;
      }

      else if (modelInfo->m_preProcCfg.taskType == "segmentation")
      {
        float alpha = modelInfo->m_postProcCfg.alpha;
        if (RETURN_FAIL == prepSegResult(&img, wanted_width, wanted_height, alpha, &interpreter, &outputs))
          return RETURN_FAIL;
      }

      LOG_INFO("saving image result file \n");
      cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
      char filename[200];
      char foldername[600];
      strcpy(foldername, "model-artifacts/tfl/");
      strcat(foldername,modelInfo->m_postProcCfg.modelName.c_str());
      strcat(foldername, "/");
      struct stat buffer;
      if (stat(foldername, &buffer) != 0) {
        if (mkdir(foldername, 0777) == -1){
          LOG_ERROR("failed to create folder %s:%s\n", foldername,strerror(errno));
          return RETURN_FAIL;
        }
      } 
      strcpy(filename, "post_proc_out_");
      strcat(filename, modelInfo->m_preProcCfg.modelName.c_str());
      strcat(filename, ".jpg");
      strcat(foldername,filename);
      if (false == cv::imwrite(foldername, img))
      {
        LOG_INFO("Saving the image, FAILED\n");
        return RETURN_FAIL;
      }

      if (s->device_mem)
      {
        for (uint32_t i = 0; i < inputs.size(); i++)
        {
          if (in_ptrs[i])
          {
            TIDLRT_freeSharedMem(in_ptrs[i]);
          }
        }
        for (uint32_t i = 0; i < outputs.size(); i++)
        {
          if (out_ptrs[i])
          {
            TIDLRT_freeSharedMem(out_ptrs[i]);
          }
        }
      }
      LOG_INFO("\nCompleted_Model : 0, Name : %s, Total time : %f, Offload Time : 0 , DDR RW MBs : 0, Output File : %s \n \n",\
       modelInfo->m_postProcCfg.modelName.c_str(), avg_time,filename);
      return RETURN_SUCCESS;
    }

  } // namespace main
} // namespace tflite

int main(int argc, char **argv)
{
  Settings s;
  if (parseArgs(argc, argv, &s) == RETURN_FAIL)
  {
    LOG_ERROR("Failed to parse the args\n");
    return RETURN_FAIL;
  }
  dumpArgs(&s);
  logSetLevel((LogLevel)s.log_level);
  /* Parse the input configuration file */
  ModelInfo model(s.model_zoo_path);
  if (model.initialize() == RETURN_FAIL)
  {
    LOG_ERROR("Failed to initialize model\n");
    return RETURN_FAIL;
  }
  if (tflite::main::runInference(&model, &s) == RETURN_FAIL)
  {
    LOG_ERROR("Failed to run runInference\n");
    return RETURN_FAIL;
  }
  return RETURN_SUCCESS;
}
