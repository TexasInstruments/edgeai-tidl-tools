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

#ifndef TENSORFLOW_LITE_EXAMPLES_MAIN_H_
#define TENSORFLOW_LITE_EXAMPLES_MAIN_H_

#include <fcntl.h>     // NOLINT(build/include_order)
#include <getopt.h>    // NOLINT(build/include_order)
#include <sys/time.h>  // NOLINT(build/include_order)
#include <sys/types.h> // NOLINT(build/include_order)
#include <sys/uio.h>   // NOLINT(build/include_order)
#include <unistd.h>    // NOLINT(build/include_order)

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

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "dlfcn.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <functional>
#include <queue>

#include "tensorflow/lite/tools/evaluation/utils.h"

#include "itidl_rt.h"
#include "tfl_config.h"
#include "post_process/tfl_post_process.h"
#include "pre_process/tfl_pre_process.h"

#define LOG(x) std::cerr

namespace tflite
{
    namespace main
    {
        struct Settings
        {
            bool verbose = false;
            bool accel = false;
            bool device_mem = false;
            bool input_floating = false;
            bool profiling = false;
            bool allow_fp16 = false;
            bool gl_backend = false;
            bool hexagon_delegate = false;
            int loop_count = 1;
            float input_mean = 127.5f;
            float input_std = 127.5f;
            string artifact_path = "";
            string model_name = "./mobilenet_quant_v1_224.tflite";
            tflite::FlatBufferModel *model;
            string input_bmp_name = "./grace_hopper.bmp";
            string labels_file_name = "./labels.txt";
            string input_layer_type = "uint8_t";
            int number_of_threads = 4;
            int number_of_results = 5;
            int max_profiling_buffer_entries = 1024;
            int number_of_warmup_runs = 2;
            tflite::config::TFliteModeltype model_type;
        };

        void display_usage()
        {
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

        template <class T>
        cv::Mat preprocImage(const std::string &input_bmp_name,
                             T *out, int wanted_height, int wanted_width,
                             int wanted_channels, float mean, float scale)
        {
            int i;
            uint8_t *pSrc;
            cv::Mat image = cv::imread(input_bmp_name, cv::IMREAD_COLOR);
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            cv::resize(image, image, cv::Size(wanted_width, wanted_height), 0, 0, cv::INTER_AREA);
            if (image.channels() != wanted_channels)
            {
                // LOG(FATAL) << "Warning : Number of channels wanted differs from number of channels in the actual image \n";
                exit(-1);
            }
            pSrc = (uint8_t *)image.data;
            for (i = 0; i < wanted_height * wanted_width * wanted_channels; i++)
                out[i] = ((T)pSrc[i] - mean) / scale;
            return image;
        }
        /**
     * Use OpenCV to do in-place update of a buffer with post processing content like
     * alpha blending a specific color for each classified pixel. Typically used for
     * semantic segmentation models.
     * Although OpenCV expects BGR data, this function adjusts the color values so that
     * the post processing can be done on a RGB buffer without extra performance impact.
     * For every pixel in input frame, this will find the scaled co-ordinates for a
     * downscaled result and use the color associated with detected class ID.
     *
     * @param frame Original RGB data buffer, where the in-place updates will happen
     * @param classes Reference to a vector of vector of floats representing the output
     *          from an inference API. It should contain 1 vector describing the class ID
     *          detected for that pixel.
     * @returns original frame with some in-place post processing done
     */
        template <typename T1, typename T2>
        static T1 *blendSegMask(T1 *frame, T2 *classes, int32_t inDataWidth, int32_t inDataHeight,
                                int32_t outDataWidth, int32_t outDataHeight, float alpha)
        {
            uint8_t *ptr;
            uint8_t a;
            uint8_t sa;
            uint8_t r;
            uint8_t g;
            uint8_t b;
            uint8_t r_m;
            uint8_t g_m;
            uint8_t b_m;
            int32_t w;
            int32_t h;
            int32_t sw;
            int32_t sh;
            int32_t class_id;

            a = alpha * 255;
            sa = (1 - alpha) * 255;

            // Here, (w, h) iterate over frame and (sw, sh) iterate over classes
            for (h = 0; h < outDataHeight; h++)
            {
                sh = (int32_t)(h * inDataHeight / outDataHeight);
                ptr = frame + h * (outDataWidth * 3);

                for (w = 0; w < outDataWidth; w++)
                {
                    int32_t index;
                    sw = (int32_t)(w * inDataWidth / outDataWidth);

                    // Get the RGB values from original image
                    r = *(ptr + 0);
                    g = *(ptr + 1);
                    b = *(ptr + 2);

                    // sw and sh are scaled co-ordiates over the results[0] vector
                    // Get the color corresponding to class detected at this co-ordinate
                    index = (int32_t)(sh * inDataHeight + sw);
                    class_id = classes[index];

                    // random color assignment based on class-id's
                    r_m = 10 * class_id;
                    g_m = 20 * class_id;
                    b_m = 30 * class_id;

                    // Blend the original image with mask value
                    *(ptr + 0) = ((r * a) + (r_m * sa)) / 255;
                    *(ptr + 1) = ((g * a) + (g_m * sa)) / 255;
                    *(ptr + 2) = ((b * a) + (b_m * sa)) / 255;

                    ptr += 3;
                }
            }

            return frame;
        }

        // Takes a file name, and loads a list of labels from it, one per line, and
        // returns a vector of the strings. It pads with empty strings so the length
        // of the result is a multiple of 16, because our model expects that.
        TfLiteStatus ReadLabelsFile(const string &file_name,
                                    std::vector<string> *result,
                                    size_t *found_label_count)
        {
            std::ifstream file(file_name);
            if (!file)
            {
                LOG(FATAL) << "Labels file " << file_name << " not found\n";
                return kTfLiteError;
            }
            result->clear();
            string line;
            while (std::getline(file, line))
            {
                result->push_back(line);
            }
            *found_label_count = result->size();
            const int padding = 16;
            while (result->size() % padding)
            {
                result->emplace_back();
            }
            return kTfLiteOk;
        }

        // Returns the top N confidence values over threshold in the provided vector,
        // sorted by confidence in descending order.
        template <class T>
        void get_top_n(T *prediction, int prediction_size, size_t num_results,
                       float threshold, std::vector<std::pair<float, int>> *top_results,
                       bool input_floating)
        {
            // Will contain top N results in ascending order.
            std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                                std::greater<std::pair<float, int>>>
                top_result_pq;

            const long count = prediction_size; // NOLINT(runtime/int)
            for (int i = 0; i < count; ++i)
            {
                float value;
                if (input_floating)
                    value = prediction[i];
                else
                    value = prediction[i] / 255.0;
                // Only add it if it beats the threshold and has a chance at being in
                // the top N.
                if (value < threshold)
                {
                    continue;
                }

                top_result_pq.push(std::pair<float, int>(value, i));

                // If at capacity, kick the smallest value out.
                if (top_result_pq.size() > num_results)
                {
                    top_result_pq.pop();
                }
            }

            // Copy to output vector and reverse into descending order.
            while (!top_result_pq.empty())
            {
                top_results->push_back(top_result_pq.top());
                top_result_pq.pop();
            }

            std::reverse(top_results->begin(), top_results->end());
        }

    } //namespace tflite::main
} //namespace tflite

#endif // TENSORFLOW_LITE_EXAMPLES_MAIN_H_