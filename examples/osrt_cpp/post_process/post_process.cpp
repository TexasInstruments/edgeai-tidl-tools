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

/* Module headers. */
#include "post_process.h"

namespace tidl
{
    namespace postprocess
    {
        using namespace cv;
        using namespace std;

        /**
         * Use OpenCV to do in-place update of a buffer with post processing content
         * like drawing bounding box around a detected object in the frame. Typically
         * used for object classification models.
         * Although OpenCV expects BGR data, this function adjusts the color values so
         * that the post processing can be done on a RGB buffer without extra
         * performance impact.
         *
         * @param frame Original RGB data buffer, where the in-place updates will happen
         * @param num_of_detections
         * @param box bounding box co-ordinates.
         * @param score scores of detection for comparing with threshold.
         * @param threshold threshold.
         * @returns original frame with some in-place post processing done
         */
        cv::Mat overlayBoundingBox(cv::Mat img, int num_of_detection, float *cordinates, float *scores, float threshold)
        {
            cv::Scalar box_color = (20, 120, 20);
            for (int i = 0; i < num_of_detection; i++)
            {
                if (scores[i] > threshold)
                {
                    /* hard coded colour of box */
                    int boxThickness = 2;
                    float ymin = cordinates[i * 4 + 0];
                    float xmin = cordinates[i * 4 + 1];
                    float ymax = cordinates[i * 4 + 2];
                    float xmax = cordinates[i * 4 + 3];
                    cv::Point topleft = cv::Point(xmin * img.cols, ymax * img.rows);
                    cv::Point bottomright = cv::Point(xmax * img.cols, ymin * img.rows);
                    cv::rectangle(img, topleft, bottomright, box_color, boxThickness, cv::LINE_8);
                }
            }
            return img;
        }

        /**
         * Use OpenCV to do in-place update of a buffer with post processing content
         * like alpha blending a specific color for each classified pixel. Typically
         * use for semantic segmentation models.
         * Although OpenCV expects BGR data, this function adjusts the color values so
         * that the post processing can be done on a RGB buffer without extra
         * performance impact.
         * For every pixel in input frame, this will find the scaled co-ordinates for a
         * downscaled result and use the color associated with detected class ID.
         *
         * @param frame Original RGB data buffer, where the in-place updates will happen
         * @param classes Reference to a vector of vector of floats representing the
         *          output from an inference API. It should contain 1 vector describing
         *           the class ID detected for that pixel.
         * @returns original frame with some in-place post processing done
         */
        template <class T>
        uchar *blendSegMask(uchar *frame,
                            T *classes,
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
            /* Here, (w, h) iterate over frame and (sw, sh) iterate over
            classes*/
            for (h = 0; h < outDataHeight; h++)
            {
                sh = (int32_t)(h * inDataHeight / outDataHeight);
                ptr = frame + h * (outDataWidth * 3);

                for (w = 0; w < outDataWidth; w++)
                {
                    int32_t index;
                    sw = (int32_t)(w * inDataWidth / outDataWidth);

                    /* Get the RGB values from original image*/
                    r = *(ptr + 0);
                    g = *(ptr + 1);
                    b = *(ptr + 2);

                    /* sw and sh are scaled co-ordiates over the results[0]
                    vector Get the color corresponding to class detected at
                    this co-ordinate*/
                    index = (int32_t)(sh * inDataHeight + sw);
                    class_id = classes[index];

                    /* random color assignment based on class-id's */
                    r_m = 10 * class_id;
                    g_m = 20 * class_id;
                    b_m = 30 * class_id;

                    /* Blend the original image with mask value*/
                    *(ptr + 0) = ((r * a) + (r_m * sa)) / 255;
                    *(ptr + 1) = ((g * a) + (g_m * sa)) / 255;
                    *(ptr + 2) = ((b * a) + (b_m * sa)) / 255;

                    ptr += 3;
                }
            }

            return frame;
        }

        template uchar *blendSegMask<int64_t>(uchar *frame,
                                              int64_t *classes,
                                              int32_t inDataWidth,
                                              int32_t inDataHeight,
                                              int32_t outDataWidth,
                                              int32_t outDataHeight,
                                              float alpha);

        template uchar *blendSegMask<int32_t>(uchar *frame,
                                              int32_t *classes,
                                              int32_t inDataWidth,
                                              int32_t inDataHeight,
                                              int32_t outDataWidth,
                                              int32_t outDataHeight,
                                              float alpha);

        template uchar *blendSegMask<float>(uchar *frame,
                                            float *classes,
                                            int32_t inDataWidth,
                                            int32_t inDataHeight,
                                            int32_t outDataWidth,
                                            int32_t outDataHeight,
                                            float alpha);

        /**
         *  \brief Takes a file name, and loads a list of labels from it, one per line,
         *  and returns a vector of the strings. It pads with empty strings so the
         * length of the result is a multiple of 16, because our model expects that.
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
                LOG_ERROR("Labels file  %s not found\n", file_name.c_str());
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
         *  \brief Use OpenCV to do in-place update of a buffer with post processing
         *  content like a black rectangle at the top-left corner and text lines
         * describing the detected class names. Typically used for image
         * classification models Although OpenCV expects BGR data, this function
         * adjusts the color values so that the post processing can be done on a RGB
         * buffer without extra performance impact.
         *
         * @param frame Original RGB data buffer, where the in-place updates will happen
         * @param top_results Reference to a vector of pair of float and int
         *          representing the output from an inference API. It should vectors
         *          representing the probability with which that class is detected and
         *          class index in this image.
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
