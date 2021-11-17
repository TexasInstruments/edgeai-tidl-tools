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

#include "tfl_main.h"

namespace tflite
{
  namespace main
  {

    void *in_ptrs[16] = {NULL};
    void *out_ptrs[16] = {NULL};

    /**
  
  *  \brief  Actual infernce happening 
  *  \param  ModelInfo YAML parsed model info
  *  \param  Settings user input options  and default values of setting if any
  * @returns void
  */
    void RunInference(tidl::modelInfo::ModelInfo *modelInfo, tidl::arg_parsing::Settings *s)
    {
      /* checking model path present or not*/
      if (!modelInfo->m_infConfig.modelFile.c_str())
      {
        std::cout << "no model file name\n";
        exit(-1);
      }
      /* preparing tflite model  from file*/
      std::unique_ptr<tflite::FlatBufferModel> model;
      std::unique_ptr<tflite::Interpreter> interpreter;
      model = tflite::FlatBufferModel::BuildFromFile(modelInfo->m_infConfig.modelFile.c_str());
      if (!model)
      {
        std::cout << "\nFailed to mmap model " << modelInfo->m_infConfig.modelFile << "\n";
        exit(-1);
      }
      std::cout << "Loaded model " << modelInfo->m_infConfig.modelFile << "\n";
      model->error_reporter();
      std::cout << "resolved reporter\n";

      tflite::ops::builtin::BuiltinOpResolver resolver;
      tflite::InterpreterBuilder(*model, resolver)(&interpreter);
      if (!interpreter)
      {
        std::cout << "Failed to construct interpreter\n";
        exit(-1);
      }
      const std::vector<int> inputs = interpreter->inputs();
      const std::vector<int> outputs = interpreter->outputs();

      std::cout << "tensors size: " << interpreter->tensors_size() << "\n";
      std::cout << "nodes size: " << interpreter->nodes_size() << "\n";
      std::cout << "number of inputs: " << inputs.size() << "\n";
      std::cout << "number of outputs: " << outputs.size() << "\n";
      std::cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";

      if (inputs.size() != 1)
      {
        std::cout << "Supports only single input models \n";
        exit(-1);
      }

      if (s->log_level <= tidl::utils::DEBUG)
      {
        int t_size = interpreter->tensors_size();
        for (int i = 0; i < t_size; i++)
        {
          if (interpreter->tensor(i)->name)
            std::cout << i << ": " << interpreter->tensor(i)->name << ", "
                      << interpreter->tensor(i)->bytes << ", "
                      << interpreter->tensor(i)->type << ", "
                      << interpreter->tensor(i)->params.scale << ", "
                      << interpreter->tensor(i)->params.zero_point << "\n";
        }
      }

      if (s->number_of_threads != -1)
      {
        interpreter->SetNumThreads(s->number_of_threads);
      }

      int input = inputs[0];
      if (s->log_level <= tidl::utils::INFO)
        std::cout << "input: " << input << "\n";

      if (s->accel == 1)
      {
        /* This part creates the dlg_ptr */
        std::cout << "accelerated mode\n";
        typedef TfLiteDelegate *(*tflite_plugin_create_delegate)(char **, char **, size_t, void (*report_error)(const char *));
        tflite_plugin_create_delegate tflite_plugin_dlg_create;
        char *keys[] = {"artifacts_folder", "num_tidl_subgraphs", "debug_level"};
        char *values[] = {(char *)modelInfo->m_infConfig.artifactsPath.c_str(), "16", "0"};
        void *lib = dlopen("libtidl_tfl_delegate.so", RTLD_NOW);
        assert(lib);
        tflite_plugin_dlg_create = (tflite_plugin_create_delegate)dlsym(lib, "tflite_plugin_create_delegate");
        TfLiteDelegate *dlg_ptr = tflite_plugin_dlg_create(keys, values, 3, NULL);
        interpreter->ModifyGraphWithDelegate(dlg_ptr);
        printf("ModifyGraphWithDelegate - Done \n");
      }
      if (interpreter->AllocateTensors() != kTfLiteOk)
      {
        std::cout << "Failed to allocate tensors!";
      }

      if (s->device_mem)
      {
        std::cout << "device mem enabled\n";
        for (uint32_t i = 0; i < inputs.size(); i++)
        {
          const TfLiteTensor *tensor = interpreter->input_tensor(i);
          in_ptrs[i] = TIDLRT_allocSharedMem(tflite::kDefaultTensorAlignment, tensor->bytes);
          if (in_ptrs[i] == NULL)
          {
            std::cout << "Could not allocate Memory for input: " << tensor->name << "\n";
          }
          interpreter->SetCustomAllocationForTensor(inputs[i], {in_ptrs[i], tensor->bytes});
        }
        for (uint32_t i = 0; i < outputs.size(); i++)
        {
          const TfLiteTensor *tensor = interpreter->output_tensor(i);
          out_ptrs[i] = TIDLRT_allocSharedMem(tflite::kDefaultTensorAlignment, tensor->bytes);
          if (out_ptrs[i] == NULL)
          {
            std::cout << "Could not allocate Memory for ouput: " << tensor->name << "\n";
          }
          interpreter->SetCustomAllocationForTensor(outputs[i], {out_ptrs[i], tensor->bytes});
        }
      }

      if (s->log_level <= tidl::utils::DEBUG)
        PrintInterpreterState(interpreter.get());
      /* get input dimension from the YAML parsed  and batch 
      from input tensor assuming one tensor*/
      TfLiteIntArray *dims = interpreter->tensor(input)->dims;
      int wanted_batch = dims->data[0];
      int wanted_height = modelInfo->m_preProcCfg.outDataHeight;
      int wanted_width = modelInfo->m_preProcCfg.outDataWidth;
      int wanted_channels = modelInfo->m_preProcCfg.numChans;
      cv::Mat img;
      switch (interpreter->tensor(input)->type)
      {
      case kTfLiteFloat32:
      {
        img = tidl::preprocess::preprocImage<float>(s->input_bmp_path, &interpreter->typed_tensor<float>(input)[0], modelInfo->m_preProcCfg);
        break;
      }
      case kTfLiteUInt8:
      {
        img = tidl::preprocess::preprocImage<uint8_t>(s->input_bmp_path, &interpreter->typed_tensor<uint8_t>(input)[0], modelInfo->m_preProcCfg);
        break;
      }
      default:
        std::cout << "cannot handle input type " << interpreter->tensor(input)->type << " yet";
        exit(-1);
      }

      printf("interpreter->Invoke - Started \n");
      if (s->loop_count > 1)
        for (int i = 0; i < s->number_of_warmup_runs; i++)
        {
          if (interpreter->Invoke() != kTfLiteOk)
          {
            LOG(FATAL) << "Failed to invoke tflite!\n";
          }
        }

      struct timeval start_time, stop_time;
      gettimeofday(&start_time, nullptr);
      for (int i = 0; i < s->loop_count; i++)
      {
        if (interpreter->Invoke() != kTfLiteOk)
        {
          LOG(FATAL) << "Failed to invoke tflite!\n";
        }
      }
      gettimeofday(&stop_time, nullptr);
      printf("interpreter->Invoke - Done \n");

      std::cout << "invoked \n";
      std::cout << "average time: "
                << (tidl::utility_functs::get_us(stop_time) - tidl::utility_functs::get_us(start_time)) / (s->loop_count * 1000)
                << " ms \n";

      if (!strcmp(modelInfo->m_preProcCfg.taskType.c_str(), "segmentation"))
      {
        std::cout << "preparing segmentation result \n";
        void *outputTensor = interpreter->tensor(outputs[0])->data.data;
        TfLiteType type = interpreter->tensor(outputs[0])->type;
        float alpha = 0.4f;
        img.data = tidl::postprocess::blendSegMask(img.data, outputTensor, type, img.cols, img.rows, wanted_width, wanted_height, alpha);
      }
      else if (!strcmp(modelInfo->m_preProcCfg.taskType.c_str(), "detection"))
      {
        std::cout << "preparing detection result \n";
        const float *detectection_location = interpreter->tensor(outputs[0])->data.f;
        const float *detectection_classes = interpreter->tensor(outputs[1])->data.f;
        const float *detectection_scores = interpreter->tensor(outputs[2])->data.f;
        const int num_detections = (int)*interpreter->tensor(outputs[3])->data.f;
        std::cout << "results " << num_detections << "\n";
        tidl::postprocess::overlayBoundingBox(img, num_detections, detectection_location);
        for (int i = 0; i < num_detections; i++)
        {
          std::cout << "class " << detectection_classes[i] << "\n";
          std::cout << "cordinates " << detectection_location[i * 4] << detectection_location[i * 4 + 1] << detectection_location[i * 4 + 2] << detectection_location[i * 4 + 3] << "\n";
          std::cout << "score " << detectection_scores[i] << "\n";
        }
      }
      else if (!strcmp(modelInfo->m_preProcCfg.taskType.c_str(), "classification"))
      {
        std::cout << "preparing clasification result \n";
        const float threshold = 0.001f;
        std::vector<std::pair<float, int>> top_results;

        TfLiteIntArray *output_dims = interpreter->tensor(outputs[0])->dims;
        // assume output dims to be something like (1, 1, ... ,size)
        auto output_size = output_dims->data[output_dims->size - 1];
        switch (interpreter->tensor(outputs[0])->type)
        {
        case kTfLiteFloat32:
          tidl::postprocess::get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                                              s->number_of_results, threshold, &top_results, true);
          break;
        case kTfLiteUInt8:
          tidl::postprocess::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                                                output_size, s->number_of_results, threshold,
                                                &top_results, false);
          break;
        default:
          LOG(FATAL) << "cannot handle output type "
                     << interpreter->tensor(outputs[0])->type << " yet";
          exit(-1);
        }

        std::vector<string> labels;
        size_t label_count;

        if (tidl::postprocess::ReadLabelsFile(s->labels_file_path, &labels, &label_count) != 0)
        {
          LOG(FATAL) << "label file not found!!! \n";
          exit(-1);
        }

        for (const auto &result : top_results)
        {
          const float confidence = result.first;
          const int index = result.second;
          LOG(INFO) << confidence << ": " << index << " " << labels[index] << "\n";
        }
        int num_results = 5;
        img.data = tidl::postprocess::overlayTopNClasses(img.data, top_results, &labels, img.cols, img.rows, num_results);
      }
      LOG(INFO) << "saving image result file \n";
      cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
      char filename[100];
      // strcpy(filename, s->artifact_path.c_str());
      strcpy(filename, "/home/a0496663/edgeai-tidl-tools/test_data/");
      strcat(filename, "cpp_inference_out.jpg");
      bool check = cv::imwrite(filename, img);
      if (check == false)
      {
        std::cout << "Saving the image, FAILED" << std::endl;
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
    }

  } // namespace main
} // namespace tflite

int main(int argc, char **argv)
{
  tidl::arg_parsing::Settings s;
  tidl::arg_parsing::parse_args(argc, argv, &s);
  tidl::arg_parsing::dump_args(&s);
  tidl::utils::logSetLevel((tidl::utils::LogLevel)s.log_level);
  // Parse the input configuration file
  tidl::modelInfo::ModelInfo model(s.model_zoo_path);

  int status = model.initialize();
  tflite::main::RunInference(&model, &s);

  return 0;
}
