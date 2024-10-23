# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import copy


class DetectionFormatting():
    def __init__(self, dst_indices, src_indices):
        self.src_indices = src_indices
        self.dst_indices = dst_indices

    def __call__(self, bbox, info_dict):
        bbox_copy = copy.deepcopy(bbox)
        bbox_copy[..., self.dst_indices] = bbox[..., self.src_indices]
        return bbox_copy, info_dict


class DetectionXYXY2YXYX(DetectionFormatting):
    def __init__(self, dst_indices=(0, 1, 2, 3), src_indices=(1, 0, 3, 2)):
        super().__init__(dst_indices, src_indices)


class DetectionYXYX2XYXY(DetectionFormatting):
    def __init__(self, dst_indices=(0, 1, 2, 3), src_indices=(1, 0, 3, 2)):
        super().__init__(dst_indices, src_indices)


class DetectionYXHW2XYWH(DetectionFormatting):
    def __init__(self, dst_indices=(0, 1, 2, 3), src_indices=(1, 0, 3, 2)):
        super().__init__(dst_indices, src_indices)


class DetectionXYXY2XYWH():
    def __call__(self, bbox, info_dict):
        w = bbox[..., 2] - bbox[..., 0]
        h = bbox[..., 3] - bbox[..., 1]
        bbox[..., 2] = w
        bbox[..., 3] = h
        return bbox, info_dict


class DetectionXYWH2XYXY():
    def __call__(self, bbox, info_dict):
        x2 = bbox[..., 0] + bbox[..., 2]
        y2 = bbox[..., 1] + bbox[..., 3]
        bbox[..., 2] = x2
        bbox[..., 3] = y2
        return bbox, info_dict
    
class DetectionXYWH2XYXYCenterXY():
    def __call__(self, bbox, info_dict):
        x1 = bbox[..., 0] - 0.5 * bbox[..., 2]
        y1 = bbox[..., 1] - 0.5 * bbox[..., 3]
        x2 = bbox[..., 0] + 0.5 * bbox[..., 2]
        y2 = bbox[..., 1] + 0.5 * bbox[..., 3]
        img_shape =  info_dict['data_shape']
        resize_shape =  info_dict['resize_shape']
        bbox[..., 0] = x1 * resize_shape[1]
        bbox[..., 1] = y1 * resize_shape[0]
        bbox[..., 2] = x2 * resize_shape[1]
        bbox[..., 3] = y2 * resize_shape[0]
        return bbox, info_dict
    

class DetectionBoxSL2BoxLS(DetectionFormatting):
    def __init__(self, dst_indices=(4, 5), src_indices=(5, 4)):
        super().__init__(dst_indices, src_indices)

class Yolov4DetectionBoxSL2BoxLS(DetectionFormatting):
    def __init__(self, dst_indices=(0,1,2,3,4), src_indices=(1,2,3,4,0)):
        super().__init__(dst_indices, src_indices)
