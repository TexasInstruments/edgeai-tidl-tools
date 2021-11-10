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
  *  \param  Settings user input options  and default values of setting if any
  * @returns void
  */
    void RunInference(tidl::arg_parsing::Settings *s)
    {
      if (!s->model_path.c_str())
      {
        LOG(ERROR) << "no model file name\n";
        exit(-1);
      }

      std::unique_ptr<tflite::FlatBufferModel> model;
      std::unique_ptr<tflite::Interpreter> interpreter;
      model = tflite::FlatBufferModel::BuildFromFile(s->model_path.c_str());
      if (!model)
      {
        LOG(FATAL) << "\nFailed to mmap model " << s->model_path << "\n";
        exit(-1);
      }

      LOG(INFO) << "Loaded model " << s->model_path << "\n";
      model->error_reporter();
      LOG(INFO) << "resolved reporter\n";

      tflite::ops::builtin::BuiltinOpResolver resolver;
      tflite::InterpreterBuilder(*model, resolver)(&interpreter);
      if (!interpreter)
      {
        LOG(FATAL) << "Failed to construct interpreter\n";
        exit(-1);
      }

      if (s->verbose)
      {
        LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
        LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
        LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
        LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

        int t_size = interpreter->tensors_size();
        for (int i = 0; i < t_size; i++)
        {
          if (interpreter->tensor(i)->name)
            LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
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

      int input = interpreter->inputs()[0];
      if (s->verbose)
        LOG(INFO) << "input: " << input << "\n";

      const std::vector<int> inputs = interpreter->inputs();
      const std::vector<int> outputs = interpreter->outputs();

      if (s->verbose)
      {
        LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
        LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
      }

      if (s->accel == 1)
      {
        char artifact_path[512];
        /* This part creates the dlg_ptr */
        typedef TfLiteDelegate *(*tflite_plugin_create_delegate)(char **, char **, size_t, void (*report_error)(const char *));
        tflite_plugin_create_delegate tflite_plugin_dlg_create;
        char *keys[] = {"artifacts_folder", "num_tidl_subgraphs", "debug_level"};
        char *values[] = {(char *)s->artifact_path.c_str(), "16", "0"};
        void *lib = dlopen("libtidl_tfl_delegate.so", RTLD_NOW);
        assert(lib);
        tflite_plugin_dlg_create = (tflite_plugin_create_delegate)dlsym(lib, "tflite_plugin_create_delegate");
        TfLiteDelegate *dlg_ptr = tflite_plugin_dlg_create(keys, values, 3, NULL);
        interpreter->ModifyGraphWithDelegate(dlg_ptr);
        printf("ModifyGraphWithDelegate - Done \n");
      }

      if (interpreter->AllocateTensors() != kTfLiteOk)
      {
        LOG(FATAL) << "Failed to allocate tensors!";
      }

      if (s->device_mem)
      {
        for (uint32_t i = 0; i < inputs.size(); i++)
        {
          const TfLiteTensor *tensor = interpreter->input_tensor(i);
          in_ptrs[i] = TIDLRT_allocSharedMem(tflite::kDefaultTensorAlignment, tensor->bytes);
          if (in_ptrs[i] == NULL)
          {
            LOG(FATAL) << "Could not allocate Memory for input: " << tensor->name << "\n";
          }
          interpreter->SetCustomAllocationForTensor(inputs[i], {in_ptrs[i], tensor->bytes});
        }
        for (uint32_t i = 0; i < outputs.size(); i++)
        {
          const TfLiteTensor *tensor = interpreter->output_tensor(i);
          out_ptrs[i] = TIDLRT_allocSharedMem(tflite::kDefaultTensorAlignment, tensor->bytes);
          if (out_ptrs[i] == NULL)
          {
            LOG(FATAL) << "Could not allocate Memory for ouput: " << tensor->name << "\n";
          }
          interpreter->SetCustomAllocationForTensor(outputs[i], {out_ptrs[i], tensor->bytes});
        }
      }

      if (s->verbose)
        PrintInterpreterState(interpreter.get());

      /* get input dimension from the input tensor metadata
      assuming one input only */
      TfLiteIntArray *dims = interpreter->tensor(input)->dims;
      int wanted_height = dims->data[1];
      int wanted_width = dims->data[2];
      int wanted_channels = dims->data[3];
      cv::Mat img;
      switch (interpreter->tensor(input)->type)
      {
      case kTfLiteFloat32:
      {
        std::vector<float> image_data(wanted_height * wanted_width * wanted_channels);
        img = tidl::preprocess::preprocImage<float>(s->input_bmp_path, image_data, wanted_height, wanted_width, wanted_channels, s->input_mean, s->input_std);
        for (int i = 0; i < wanted_width * wanted_height; i++)
        {
          for (int j = 0; j < wanted_channels; j++)
          {
            interpreter->typed_tensor<float>(input)[i * 3 + j] = image_data[j * wanted_height * wanted_width + i];
          }
        }
        break;
      }
      case kTfLiteUInt8:
      {
        std::vector<uint8_t> image_data(wanted_height * wanted_width * wanted_channels);
        img = tidl::preprocess::preprocImage<uint8_t>(s->input_bmp_path, image_data, wanted_height, wanted_width, wanted_channels, s->input_mean, s->input_std);
        for (int i = 0; i < wanted_width * wanted_height; i++)
        {
          for (int j = 0; j < wanted_channels; j++)
          {
            interpreter->typed_tensor<uint8_t>(input)[i * 3 + j] = image_data[j * wanted_height * wanted_width + i];
          }
        }
        break;
      }
      default:
        LOG(FATAL) << "cannot handle input type " << interpreter->tensor(input)->type << " yet";
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

      LOG(INFO) << "invoked \n";
      LOG(INFO) << "average time: "
                << (tidl::utility_functs::get_us(stop_time) - tidl::utility_functs::get_us(start_time)) / (s->loop_count * 1000)
                << " ms \n";

      if (strcmp(s->task_type.c_str(), "segmentation"))
      {
        int32_t *outputTensor = interpreter->tensor(outputs[0])->data.i32;
        float alpha = 0.4f;
        img.data = tidl::postprocess::blendSegMask(img.data, outputTensor, img.cols, img.rows, wanted_width, wanted_height, alpha);
      }
      else if (strcmp(s->task_type.c_str(), "detection"))
      {
        const float *detectection_location = interpreter->tensor(outputs[0])->data.f;
        const float *detectection_classes = interpreter->tensor(outputs[1])->data.f;
        const float *detectection_scores = interpreter->tensor(outputs[2])->data.f;
        const int num_detections = (int)*interpreter->tensor(outputs[3])->data.f;
        LOG(INFO) << "results " << num_detections << "\n";
        tidl::postprocess::overlayBoundingBox(img, num_detections, detectection_location);
        for (int i = 0; i < num_detections; i++)
        {
          LOG(INFO) << "class " << detectection_classes[i] << "\n";
          LOG(INFO) << "cordinates " << detectection_location[i * 4] << detectection_location[i * 4 + 1] << detectection_location[i * 4 + 2] << detectection_location[i * 4 + 3] << "\n";
          LOG(INFO) << "score " << detectection_scores[i] << "\n";
        }
      }
      else if (strcmp(s->task_type.c_str(), "classification"))
      {
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
          exit(-1);

        for (const auto &result : top_results)
        {
          const float confidence = result.first;
          const int index = result.second;
          LOG(INFO) << confidence << ": " << index << " " << labels[index] << "\n";
        }
        int num_results = 5;
        img.data = tidl::postprocess::overlayTopNClasses(img.data, top_results, &labels, img.cols, img.rows, num_results);
      }
      cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
      char filename[100];
      strcpy(filename, s->artifact_path.c_str());
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
    /**
  *  \brief  options parsing and infernce calling
  * @returns int
  */
    int TFLite_Main(int argc, char **argv)
    {

      //YAML parsing

      // for (int i = 0; i < NUM_CONFIGS; i++)
      // {
      //   bool isTflModel = endsWith(tidl::config::model_configs[i].model_path, "tflite");
      //   if (isTflModel)
      //   {
      //     s.artifact_path = tidl::config::model_configs[i].artifact_path;
      //     s.model_name = tidl::config::model_configs[i].model_path;
      //     s.labels_file_name = tidl::config::model_configs[i].labels_path;
      //     s.input_bmp_name = tidl::config::model_configs[i].image_path;
      //     s.input_mean = tidl::config::model_configs[i].mean;
      //     s.input_std = tidl::config::model_configs[i].std;
      //     s.model_type = tidl::config::model_configs[i].model_type;
      //     RunInference(&s);
      //   }
      // }

      return 0;
    }

  } // namespace main
} // namespace tflite

int main(int argc, char **argv)
{
  tidl::arg_parsing::Settings s;
  tidl::arg_parsing::parse_args(argc, argv, &s);
  tidl::arg_parsing::dump_args(&s);
  // Parse the input configuration file
  std::string artifacts_yaml_file_path(s.model_zoo_path);
  artifacts_yaml_file_path += "/artifacts.yaml";
  YAML::Node yaml = YAML::LoadFile(artifacts_yaml_file_path);
  YAML::Node cl0010tflitert = yaml["cl-0010_tflitert"];
  std::cout << yaml.size() << "\n";
  for (YAML::const_iterator it = yaml.begin(); it != yaml.end(); ++it)
  {
    std::string key = it->first.as<std::string>();       // <- key
    YAML::Node model_node = it->second; // <- value
    tidl::utility_functs::Model model;
    model.model_name = model_node["model_name"].as<std::string>();
    model.recommended = model_node["recommended"].as<bool>();
    model.run_dir = model_node["run_dir"].as<std::string>();
    model.session_name = model_node["session_name"].as<std::string>();
    model.shortlisted = model_node["shortlisted"].as<bool>();
    model.size = model_node["size"].as<uint64_t>();
    model.task_type = model_node["task_type"].as<std::string>();
    
  }
  return 0;
}
