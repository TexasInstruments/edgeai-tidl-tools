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
#include "../utils/include/utility_functs.h"

#define TI_POSTPROC_DEFAULT_WIDTH 1280

namespace tidl
{
    namespace postprocess
    {
        using namespace std;
        using namespace tidl::modelInfo;

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
        int overlayBoundingBox(cv::Mat *img, std::vector<std::vector<float>> *od_formatted_vec, ModelInfo *modelInfo);

        /**
         * Use OpenCV to do in-place update of a buffer with post processing content
         * like alpha blending a specific color for each classified pixel. Typically
         * used for semantic segmentation models.
         * Although OpenCV expects BGR data, this function adjusts the color values
         * so that the post processing can be done on a RGB buffer without extra
         * performance impact.
         * For every pixel in input frame, this will find the scaled co-ordinates for a
         * downscaled result and use the color associated with detected class ID.
         *
         * @param frame Original RGB data buffer, where the in-place updates will happen
         * @param classes Reference to a vector of vector of floats representing the
         *          output from an inference API. It should contain 1 vector describing
         *          the class ID detected for that pixel.
         * @returns original frame with some in-place post processing done
         */
        template <class T>
        uchar *blendSegMask(uchar *frame,
                            T *classes,
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
        void getTopN(T *prediction, int prediction_size, size_t num_results,
                     float threshold, std::vector<std::pair<float, int>> *top_results,
                     bool input_floating);

        /**
         *  \brief Takes a file name, and loads a list of labels from it, one per line,
         * and returns a vector of the strings. It pads with empty strings so the
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
                           size_t *found_label_count);
        /**
         *  \brief Use OpenCV to do in-place update of a buffer with post processing
         * content like a black rectangle at the top-left corner and text lines
         * describing the detected class names. Typically used for image
         * classification models
         * Although OpenCV expects BGR data, this function adjusts the color values
         * so that the post processing can be done on a RGB buffer without extra
         * performance impact.
         *
         * @param frame Original RGB data buffer, where the in-place updates will
         *              happen
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
                                  int32_t N);

        /**  \brief Argmax computation for seg model
         *
         *  \param  arr : output array of size nheight*nwidth
         *  \param  tensor_op_array : tensor op of model
         *  \param  nwidth
         *  \param  nheight
         *  \param  nclasses
         *  \return null
         */
        template <class T>
        void argMax(T *arr, T *tensor_op_array, int nwidth, int nheight, int nclasses);

        /**
         *  \brief create a float vec from array of type data
         *
         *  \param  inData : poimter to input array of data
         *  \param  outData : pointer to output vector of float
         *  \param  tensor_shape
         *  \return null
         */
        template <class T>
        void createFloatVec(T *inData, vector<float> *outData, vector<int64_t> tensor_shape);

        /**
         *  \brief  create a vector which is in format for od post process
         * eg: [{x1,y1,x2,y2,score,label},{x1,y1,x2,y2,score,label}...]
         * from  unformatted  vec<vec<float>>
         *  \param f_tensor_unformatted onnx model output tensor
         *  \param tensor_shapes_vec vec comntainig shape of all op tensors
         *  \param  od_formatted_vec pointer to final formatted vec
         *  \param  num_op_tensors
         * @returns int status
         */
        int createOdFormatVec(vector<vector<float>> *f_tensor_unformatted, vector<vector<int64_t>> tensor_shapes_vec,
                              vector<vector<float>> *od_formatted_vec, int num_op_tensors);

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
                                ModelInfo *modelInfo, size_t output_count, int nboxes);
    } // namespace tidl::postprocess

#endif // _POST_PROCESS_H_
}