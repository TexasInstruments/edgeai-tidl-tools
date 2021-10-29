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

/* Module headers. */
#include "post_process.h"

/**
 * \brief Class implementing the image based object detection post-processing
 *        logic.
 */

namespace tidl
{
    namespace postprocess
    {
        using namespace cv;
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
 *
 * @returns original frame with some in-place post processing done
 */
        cv::Mat overlayBoundingBox(cv::Mat img, int num_of_detection, const float *cordinates)
        {
            cv::Scalar box_color = (20, 220, 20);
            for (int i = 0; i < num_of_detection; i++)
            {
                /* hard coded colour of box */
                cv::Scalar boxColor = cv::Scalar(20, 20, 20);
                int boxThickness = 3;
                float ymin = cordinates[i * 4 + 0];
                float xmin = cordinates[i * 4 + 1];
                float ymax = cordinates[i * 4 + 2];
                float xmax = cordinates[i * 4 + 3];
                cv::Point topleft = cv::Point(xmin * img.cols, ymax * img.rows);
                cv::Point bottomright = cv::Point(xmax * img.cols, ymin * img.rows);
                cv::rectangle(img, topleft, bottomright, box_color, boxThickness, cv::LINE_8);
            }
            return img;
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
        uchar *blendSegMask(uchar *frame,
                            int32_t *classes,
                            int32_t inDataWidth,
                            int32_t inDataHeight,
                            int32_t outDataWidth,
                            int32_t outDataHeight,
                            float alpha)
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
                           size_t *found_label_count)
        {
            std::ifstream file(file_name);
            if (!file)
            {
                // LOG(FATAL) << "Labels file " << file_name << " not found\n";
                return -1;
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
            return 0;
        }

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
                                  int32_t N)
        {

            float txtSize = static_cast<float>(outDataWidth) / TI_POSTPROC_DEFAULT_WIDTH;
            int rowSize = 40 * outDataWidth / 500;
            Scalar text_color(200, 0, 0);

            Mat img = Mat(outDataHeight, outDataWidth, CV_8UC3, frame);

            std::string title = "Top " + std::to_string(N) + " detected classes:";
            putText(img, title.c_str(), Point(5, 2 * rowSize),
                    FONT_HERSHEY_SIMPLEX, txtSize, text_color, 1);
            int i = 0;
            for (auto result : top_results)
            {
                const float confidence = result.first;
                int index = result.second;
                int32_t row = i + 3;
                string str = std::to_string(confidence);
                str.append((*labels)[index]);
                putText(img, str, Point(5, row * rowSize),
                        FONT_HERSHEY_SIMPLEX, txtSize, text_color, 1);
                i++;
            }
            return frame;
        }

    } // namespace tidl::postprocess
}
