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

#ifndef _POST_PROCESS_H_
#define _POST_PROCESS_H_

#include <stdint.h>
#include <string>
#include <iostream>
#include <queue>
#include <fstream>

/* Third-party headers. */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/profiling/profiler.h>
#include <tensorflow/lite/string_util.h>

#define TI_POSTPROC_DEFAULT_WIDTH 1280

namespace tflite
{
    namespace postprocess
    {
        /** Post-processing for image based object detection. */

        cv::Mat overlayBoundingBox(cv::Mat img, int num_of_detection, const float *cordinates);

        uchar *blendSegMask(uchar *frame,
                            int32_t *classes,
                            int32_t inDataWidth,
                            int32_t inDataHeight,
                            int32_t outDataWidth,
                            int32_t outDataHeight,
                            float alpha);

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
        TfLiteStatus ReadLabelsFile(const string &file_name,
                                    std::vector<string> *result,
                                    size_t *found_label_count);

        uchar *overlayTopNClasses(uchar *frame,
                                  std::vector<std::pair<float, int>> &top_results ,
                                  std::vector<string> *labels,
                                  int32_t outDataWidth,
                                  int32_t outDataHeight,
                                  int32_t labelOffset,
                                  int32_t N,
                                  int32_t size);
    } // namespace tflite::postprocess

#endif /* _POST_PROCESS_H_ */
}