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

import numpy as np


# Author: Manu Mathew
# Date: 2021 March
def get_color_palette_generic(num_classes):
    num_classes_3 = np.power(num_classes, 1.0/3)
    delta_color = int(256/num_classes_3)
    colors = [(r, g, b) for r in range(0,256,delta_color)
                        for g in range(0,256,delta_color)
                        for b in range(0,256,delta_color)]
    # spread the colors list to num_classes
    color_step = len(colors) / num_classes
    colors_list = []
    to_idx = 0
    while len(colors_list) < num_classes:
        from_idx = round(color_step * to_idx)
        if from_idx < len(colors):
            colors_list.append(colors[from_idx])
        else:
            break
        #
        to_idx = to_idx + 1
    #
    shortage = num_classes-len(colors_list)
    if shortage > 0:
        colors_list += colors[-shortage:]
    #
    if len(colors_list) < 256:
        colors_list += [(255,255,255)] * (256-len(colors_list))
    #
    assert len(colors_list) == 256, f'incorrect length for color palette {len(colors_list)}'
    return colors_list

    
def get_color_palette(num_classes):
    if num_classes < 8:
        color_step = 255
    elif num_classes < 27:
        color_step = 127
    elif num_classes < 64:
        color_step = 63        
    else:
        color_step  = 31
    #
    color_map = [(r, g, b) for r in range(0, 256, color_step) for g in range(0, 256, color_step) for b in range(0, 256, color_step)]
    return color_map

