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
         * Use OpenCV to do in-place update of a buffer with post processing
         * content like drawing bounding box around a detected object in the
         * frame. Typically used for object classification models.
         * Although OpenCV expects BGR data, this function adjusts the color
         * values so that the post processing can be done on a RGB buffer
         * without extra performance impact.
         * od_formatted_vec wil have detected obj data, process accroding to
         * info from postprocess info
         *
         * @param img Original RGB data buffer, where the in-place updates will
         *  happen
         * @param od_format_vector od processed vector of vector having only
         * detected object cordiantes and 6 float va curresponding to
         * x1y1x2y2 score label in order of output tensor.
         * @param modelInfo
         * @returns status
         */
        int overlayBoundingBox(cv::Mat *img, std::vector<std::vector<float>> *od_formatted_vec, ModelInfo *modelInfo)
        {
            cv::Scalar box_color = (20, 120, 20);
            int boxThickness = 2;
            /* extracting index of x1x2y1y1format  [x1y1 x2y2 label score] */
            std::vector<int32_t> format = modelInfo->m_postProcCfg.formatter;
            int x1Index = format[0];
            int y1Index = format[1];
            int x2Index = format[2];
            int y2Index = format[3];
            /* hard coded colour of box */
            for (auto it = (*od_formatted_vec).begin(); it != (*od_formatted_vec).end(); ++it)
            {
                float xmin = (*it)[x1Index];
                float ymin = (*it)[y1Index];
                float xmax = (*it)[x2Index];
                float ymax = (*it)[y2Index];

                cv::Point topleft = cv::Point(xmin * (*img).cols, ymax * (*img).rows);
                cv::Point bottomright = cv::Point(xmax * (*img).cols, ymin * (*img).rows);
                cv::rectangle((*img), topleft, bottomright, box_color, boxThickness, cv::LINE_8);
            }
            return RETURN_SUCCESS;
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
        template uchar *blendSegMask<uint8_t>(uchar *frame,
                                              uint8_t *classes,
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
        int readLabelsFile(const string &file_name,
                           std::vector<string> *result,
                           size_t *found_label_count)
        {
            std::ifstream file(file_name);
            if (!file)
            {
                LOG_ERROR("Labels file  %s not found\n", file_name.c_str());
                return RETURN_FAIL;
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
            return RETURN_SUCCESS;
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
         * @param outputoffset offset in top_result vector coz of bg class
         * @returns original frame with some in-place post processing done
         */
        uchar *overlayTopNClasses(uchar *frame,
                                  std::vector<std::pair<float, int>> &top_results,
                                  std::vector<string> *labels,
                                  int32_t outDataWidth,
                                  int32_t outDataHeight,
                                  int32_t N,
                                  int outputoffset)
        {

            float txtSize = .4f;
            int rowSize = 40 * outDataWidth / 500;
            Scalar text_color(0, 0, 255);

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
                string str;
                str.append((*labels)[index+ outputoffset]);
                putText(img, str, Point(5, row * rowSize),
                        FONT_HERSHEY_SIMPLEX, txtSize, text_color, 1);
                i++;
            }
            return frame;
        }

        /**
         *  Returns the top N confidence values over threshold in the provided vector,
         * sorted by confidence in descending order.
         * @returns top resultls
         */
        template <class T>
        void getTopN(T *prediction, int prediction_size, size_t num_results,
                     float threshold, std::vector<std::pair<float, int>> *top_results,
                     bool input_floating)
        {
            /* Will contain top N results in ascending order.*/
            std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                                std::greater<std::pair<float, int>>>
                top_result_pq;
            /* NOLINT(runtime/int) */
            const long count = prediction_size;
            for (int i = 0; i < count; ++i)
            {
                float value;
                if (input_floating)
                    value = prediction[i];
                else
                    value = prediction[i] / 255.0;
                /* Only add it if it beats the threshold and has a chance at
                being in the top N. */
                if (value < threshold)
                {
                    continue;
                }

                top_result_pq.push(std::pair<float, int>(value, i));

                /* If at capacity, kick the smallest value out. */
                if (top_result_pq.size() > num_results)
                {
                    top_result_pq.pop();
                }
            }

            /* Copy to output vector and reverse into descending order. */
            while (!top_result_pq.empty())
            {
                top_results->push_back(top_result_pq.top());
                top_result_pq.pop();
            }

            std::reverse(top_results->begin(), top_results->end());
        }

        template void getTopN<float>(float *prediction, int prediction_size, size_t num_results,
                                     float threshold, std::vector<std::pair<float, int>> *top_results,
                                     bool input_floating);

        template void getTopN<int64_t>(int64_t *prediction, int prediction_size, size_t num_results,
                                       float threshold, std::vector<std::pair<float, int>> *top_results,
                                       bool input_floating);
        template void getTopN<uint8_t>(uint8_t *prediction, int prediction_size, size_t num_results,
                                       float threshold, std::vector<std::pair<float, int>> *top_results,
                                       bool input_floating);

        /**
         *  \brief Argmax computation for seg model
         *
         *  \param  arr : output array of size nheight*nwidth
         *  \param  tensor_op_array : tensor op of model
         *  \param  nwidth
         *  \param  nheight
         *  \param  nclasses
         *  \return null
         */
        template <class T>
        void argMax(T *arr, T *tensor_op_array, int nwidth, int nheight, int nclasses)
        {
            /* iterate through all classes of nclasses */
            for (int i = 0; i < nwidth * nheight; i++)
            {
                float max_val = 0;
                int max_class = 0;
                for (int j = 0; j < nclasses; j++)
                {
                    if (tensor_op_array[i + j * nwidth * nheight] >= max_val)
                    {
                        max_val = tensor_op_array[i + j * nwidth * nheight];
                        max_class = j;
                    }
                }
                arr[i] = max_class;
            }
        }

        template void argMax<float>(float *arr, float *tensor_op_array, int nwidth, int nheight, int nclasses);

        /**
         *  \brief create a float vec from array of type data
         *
         *  \param  inData : poimter to input array of data
         *  \param  outData : pointer to output vector of float
         *  \param  tensor_shape
         *  \return null
         */
        template <class T>
        void createFloatVec(T *inData, vector<float> *outData, vector<int64_t> tensor_shape)
        {
            int size = 1;
            for (size_t i = 0; i < tensor_shape.size(); i++)
            {
                size = size * tensor_shape[i];
            }
            for (int i; i < size; i++)
            {
                (*outData).push_back(inData[i]);
            }
        }
        template void createFloatVec<float>(float *inData, vector<float> *outData, vector<int64_t> tensor_shape);
        template void createFloatVec<int64_t>(int64_t *inData, vector<float> *outData, vector<int64_t> tensor_shape);
        template void createFloatVec<int32_t>(int32_t *inData, vector<float> *outData, vector<int64_t> tensor_shape);

        /**
         *  \brief  prepare the od result inplace
         *  \param  img cv image to do inplace transform
         *  \param  f_tensor_unformatted unformatted tensor outputs
         *  \param tensor_shapes_vec vector containign shpe of all tensors
         *  \param  modelInfo
         *  \param nboxes num of detections
         *  \param output_count num of output tensors
         * @returns int status
         */
        int prepDetectionResult(cv::Mat *img, vector<vector<float>> *f_tensor_unformatted, vector<vector<int64_t>> tensor_shapes_vec,
                                ModelInfo *modelInfo, size_t output_count, int nboxes)
        {
            LOG_INFO("preparing detection result \n");
            vector<vector<float>> od_formatted_vec;
            float threshold = modelInfo->m_postProcCfg.vizThreshold;
            int cols = (*img).cols;
            /* copy the op tensors into single vector of for od post-process
            This loop will extract the data from f_tensor_unformatted
            in format od post process is expecting */
            for (size_t i = 0; i < nboxes; i++)
            {
                vector<float> temp;
                for (size_t j = 0; j < output_count; j++)
                {
                    /* shape of the ith tensor*/
                    vector<int64_t> tensor_shape = tensor_shapes_vec[j];
                    /* num of values in ith tensor*/
                    int num_val_tensor = 1;
                    /*Extract the last dimension from each of the output
                    tensors.last dimension will give the number of values present in
                    given tensor. Need to ignore all dimensions with value 1 since it
                    does not actually add a dimension */
                    auto temp_shape = tensor_shape;
                    for (auto it = temp_shape.begin(); it < temp_shape.end(); it++)
                    {
                        if ((*it) == 1)
                        {
                            temp_shape.erase(it);
                            it--;
                        }
                    }
                    if (temp_shape.size() <= 1)
                        num_val_tensor = 1;
                    else
                        num_val_tensor = temp_shape[temp_shape.size() - 1];

                    /*TODO this condition is given for tflite issue*/
                    if(temp_shape.size() == 1 && temp_shape[0] == 4)
                        num_val_tensor = 4;

                    for (size_t k = 0; k < num_val_tensor; k++)
                    {
                        temp.push_back((*f_tensor_unformatted)[nboxes * j + i][k]);
                    }
                }
                od_formatted_vec.push_back(temp);
            }
    
            vector<int32_t> format = modelInfo->m_postProcCfg.formatter;
            string formatter_name = modelInfo->m_postProcCfg.formatterName;
            /*format [x1y1 x2y2 label score]*/
            int score_index = format[5];
            /*remove all the vectors which does'nt have socre more than
             threshold */
            int x1Index = format[0];
            int y1Index = format[1];
            int x2Index = format[2];
            int y2Index = format[3];
            for (auto it = od_formatted_vec.begin(); it != od_formatted_vec.end(); ++it)
            {
                if ((*it)[score_index] < threshold)
                {
                    od_formatted_vec.erase(it);
                    it--;
                }
                else
                {
                    LOG_INFO("box with score:%f, threshold set to:%f\n",(*it)[score_index], threshold);
                    if (formatter_name == "DetectionBoxSL2BoxLS" && (*it)[x1Index] > 1)
                    {
                        (*it)[x1Index] = ((*it)[x1Index]) / cols;
                        (*it)[x2Index] = (*it)[x2Index] / cols;
                        (*it)[y1Index] = (*it)[y1Index] / cols;
                        (*it)[y2Index] = (*it)[y2Index] / cols;
                    }
                }
            }
            overlayBoundingBox(img, &od_formatted_vec, modelInfo);
            return RETURN_SUCCESS;
        }

    } // namespace tidl::postprocess
}
