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

#include "../utils/include/model_info.h"

#define TI_POSTPROC_DEFAULT_WIDTH 1280

namespace tidl
{
    namespace postprocess
    {
        using namespace std;

        /**
 * Use OpenCV to do in-place update of a buffer with post processing content like
 * drawing bounding box around a detected object in the frame. Typically used for
 * object classification models.
 * Although OpenCV expects BGR data, this function adjusts the color values so that
 * the post processing can be done on a RGB buffer without extra performance impact.
 *
 * @param frame Original RGB data buffer, where the in-place updates will happen
 * @param num_of_detections 
 * @param box bounding box co-ordinates.
 * @param score scores of detection for comparing with threshold.
 * @param threshold threshold.
 * @returns original frame with some in-place post processing done
 */
        cv::Mat overlayBoundingBox(cv::Mat img, int num_of_detection,  float *cordinates, float *scores, float threshold);

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
        uchar *blendSegMask(uchar *frame,
                            void *classes,
                            tidl::modelInfo::DlInferType type,
                            int32_t inDataWidth,
                            int32_t inDataHeight,
                            int32_t outDataWidth,
                            int32_t outDataHeight,
                            float alpha);
        /**
 *  Returns the top N confidence values over threshold in the provided vector,
 * sorted by confidence in descending order.
 * @returns top resultls
 */
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

        /**
 *  \brief Takes a file name, and loads a list of labels from it, one per line, and
 * returns a vector of the strings. It pads with empty strings so the length
 * of the result is a multiple of 16, because our model expects that.
 *
 *  \param  file_name : previous interrupt state
 *  \param  result : previous interrupt state
 *  \param  found_label_count : previous interrupt state
 * 
 *  \return int :0?Success:Failure
 */

        int ReadLabelsFile(const string &file_name,
                           std::vector<string> *result,
                           size_t *found_label_count);
        /**
  *  \brief Use OpenCV to do in-place update of a buffer with post processing content like
  * a black rectangle at the top-left corner and text lines describing the
  * detected class names. Typically used for image classification models
  * Although OpenCV expects BGR data, this function adjusts the color values so that
  * the post processing can be done on a RGB buffer without extra performance impact.
  *
  * @param frame Original RGB data buffer, where the in-place updates will happen
  * @param top_results Reference to a vector of pair of float and int representing the output
  *          from an inference API. It should vectors representing the
  *          probability with which that class is detected and class index in this image.
  * @param labels labels in indexed form to print 
  * @param outDataWidth 
  * @param outDataHeight
  * @param N Number of results to be displayed
  * @returns original frame with some in-place post processing done
  */
        uchar *overlayTopNClasses(uchar *frame,
                                  std::vector<std::pair<float, int>> &top_results,
                                  std::vector<string> *labels,
                                  int32_t outDataWidth,
                                  int32_t outDataHeight,
                                  int32_t N);
    } // namespace tidl::postprocess

#endif /* _POST_PROCESS_H_ */
}