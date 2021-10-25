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

/* Third-party headers. */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>

namespace tflite
{
    namespace postprocess
    {
        /** Post-processing for image based object detection. */

        static void *overlayBoundingBox(void *frame,
                                        int *box,
                                        int32_t outDataWidth,
                                        int32_t outDataHeight,
                                        const std::string objectname);

        /** Post-processing for image based semantic segmentation.*/
        class PostprocessImageSeg
        {
        public:
            /** Constructor.
             *
             * @param config Configuration information not present in YAML
         */
            PostprocessImageSeg();

            /** Destructor. */
            ~PostprocessImageSeg(){

            };
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
            static T1 *blendSegMask(T1 *frame,
                                    T2 *classes,
                                    int32_t inDataWidth,
                                    int32_t inDataHeight,
                                    int32_t outDataWidth,
                                    int32_t outDataHeight,
                                    float alpha);
        };
        int test();

    } // namespace tflite::postprocess

#endif /* _POST_PROCESS_H_ */
}