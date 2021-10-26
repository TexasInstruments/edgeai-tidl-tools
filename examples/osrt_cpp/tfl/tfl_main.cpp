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


    void RunInference(Settings *s)
    {
      if (!s->model_name.c_str())
      {
        LOG(ERROR) << "no model file name\n";
        exit(-1);
      }

      std::unique_ptr<tflite::FlatBufferModel> model;
      std::unique_ptr<tflite::Interpreter> interpreter;
      model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
      if (!model)
      {
        LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
        exit(-1);
      }
      s->model = model.get();
      LOG(INFO) << "Loaded model " << s->model_name << "\n";
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
        img = tflite::preprocess::preprocImage<float>(s->input_bmp_name, interpreter->typed_tensor<float>(input), wanted_height, wanted_width, wanted_channels, s->input_mean, s->input_std);
        break;
      case kTfLiteUInt8:
        img = tflite::preprocess::preprocImage<uint8_t>(s->input_bmp_name, interpreter->typed_tensor<uint8_t>(input), wanted_height, wanted_width, wanted_channels, s->input_mean, s->input_std);
        break;
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
                << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000)
                << " ms \n";

      
      if (s->model_type == tflite::config::SEG)
      {
        int32_t *outputTensor = interpreter->tensor(outputs[0])->data.i32;
        float alpha = 0.4f;
        img.data = tflite::postprocess::blendSegMask(img.data, outputTensor, img.cols, img.rows, wanted_width, wanted_height, alpha);
      }
      else if (s->model_type == tflite::config::OD)
      {
        const float *detectection_location = interpreter->tensor(outputs[0])->data.f;
        const float *detectection_classes = interpreter->tensor(outputs[1])->data.f;
        const float *detectection_scores = interpreter->tensor(outputs[2])->data.f;
        const int num_detections = (int)*interpreter->tensor(outputs[3])->data.f;
        LOG(INFO) << "results " << num_detections << "\n";
        tflite::postprocess::overlayBoundingBox(img, num_detections, detectection_location);
        for (int i = 0; i < num_detections; i++)
        {
          LOG(INFO) << "class " << detectection_classes[i] << "\n";
          LOG(INFO) << "cordinates " << detectection_location[i * 4] << detectection_location[i * 4 + 1] << detectection_location[i * 4 + 2] << detectection_location[i * 4 + 3] << "\n";
          LOG(INFO) << "score " << detectection_scores[i] << "\n";
        }
      }
      else if (s->model_type == tflite::config::CLF)
      {
        const float threshold = 0.001f;
        std::vector<std::pair<float, int>> top_results;

        
        TfLiteIntArray *output_dims = interpreter->tensor(outputs[0])->dims;
        // assume output dims to be something like (1, 1, ... ,size)
        auto output_size = output_dims->data[output_dims->size - 1];
        switch (interpreter->tensor(outputs[0])->type)
        {
        case kTfLiteFloat32:
          tflite::postprocess::get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                           s->number_of_results, threshold, &top_results, true);
          break;
        case kTfLiteUInt8:
          tflite::postprocess::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
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

        if (tflite::postprocess::ReadLabelsFile(s->labels_file_name, &labels, &label_count) != kTfLiteOk)
          exit(-1);

        for (const auto &result : top_results)
        {
          const float confidence = result.first;
          const int index = result.second;
          LOG(INFO) << confidence << ": " << index << " " << labels[index] << "\n";
        }
        // std::string str = "std::string to const char*";
        // const char *c = str.c_str();
        img.data = tflite::postprocess::overlayTopNClasses(img.data,top_results,&labels, img.cols,img.rows,10,5,10);
      }
      /*fix the location and name of the saving image file */
      cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
      char filename[100];
      strcpy(filename, s->artifact_path.c_str());
      strcat(filename,"cpp_inference_out.jpg");
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

    int TFLite_Main(int argc, char **argv)
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
          tflite::main::display_usage();
          exit(-1);
        default:
          exit(-1);
        }
      }
      for (int i = 0; i < NUM_CONFIGS; i++)
      {
        s.artifact_path = tflite::config::model_configs[i].artifact_path;
        s.model_name = tflite::config::model_configs[i].tflite_model_path;
        s.labels_file_name = tflite::config::model_configs[i].tflite_labels_path;
        s.input_bmp_name = tflite::config::model_configs[i].image_path;
        s.input_mean = tflite::config::model_configs[i].mean;
        s.input_std = tflite::config::model_configs[i].std;
        s.model_type = tflite::config::model_configs[i].model_type;
        RunInference(&s);
      }

      return 0;
    }

  } // namespace main
} // namespace tflite

int main(int argc, char **argv)
{
  return tflite::main::TFLite_Main(argc, argv);
}
