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
#include "tfl_post_process.h"

/**
 * \brief Class implementing the image based object detection post-processing
 *        logic.
 */

namespace tflite {
    namespace postprocess{
    using namespace cv;
    
    /**
 * Use OpenCV to do in-place update of a buffer with post processing content like
 * drawing bounding box around a detected object in the frame. Typically used for
 * object classification models.
 * Although OpenCV expects BGR data, this function adjusts the color values so that
 * the post processing can be done on a RGB buffer without extra performance impact.
 *
 * @param frame Original RGB data buffer, where the in-place updates will happen
 * @param box bounding box co-ordinates.
 * @param outDataWidth width of the output buffer.
 * @param outDataHeight Height of the output buffer.
 *
 * @returns original frame with some in-place post processing done
 */
    void * overlayBoundingBox(void *frame,
                                                        int *box,
                                                        int32_t outDataWidth,
                                                        int32_t outDataHeight,
                                                        const std::string objectname)
    {
        Mat img = Mat(outDataHeight, outDataWidth, CV_8UC3, frame);
        Scalar box_color = (20, 220, 20);
        Scalar text_color = (240, 240, 240);

        Point topleft = Point(box[0], box[1]);
        Point bottomright = Point(box[2], box[3]);

        // Draw bounding box for the detected object
        rectangle(img, topleft, bottomright, box_color, 3);

        Point t_topleft = Point((box[0] + box[2]) / 2 - 5, (box[1] + box[3]) / 2 + 5);
        Point t_bottomright = Point((box[0] + box[2]) / 2 + 120, (box[1] + box[3]) / 2 - 15);

        // Draw text with detected class with a background box
        rectangle(img, t_topleft, t_bottomright, box_color, -1);
        putText(img, objectname, t_topleft,
                FONT_HERSHEY_SIMPLEX, 0.5, text_color);

        return frame;
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
    T1 *PostprocessImageSeg::blendSegMask(T1 *frame,
                                          T2 *classes,
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



} // namespace tflite::postprocess
}
