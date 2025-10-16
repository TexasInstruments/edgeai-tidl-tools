import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
curr_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curr_dir)
import post_process_utils


class PostProcessSegmentation():
    """
    Post Process for Semantic Segmentation
    """
    def __init__(self):
        """
        Initialize the Semantic Segmentation Post Process
        """
        pass

    def process(self, input: Image, outputs: np.ndarray, batch: int) -> (str, Image):
        """
        Execute post processing on the image
        
        Args:
            input (Image): Input image to do post-processing on
            outputs (np.ndarray): Inference outputs
            batch (int): Output batch to pick from
            
        Returns:
            str: Mask of detected class per pixel
            Image: Post Processed output image with classes overlayed
        """
        mask = ""

        if len(outputs) == 1:
            outputs = outputs[0][batch]
        else:
            outputs = outputs[batch]

        img = input.resize((outputs.shape[-1], outputs.shape[-2]), PIL.Image.LANCZOS).convert("RGBA")

        outputs = np.squeeze(outputs)

        draw = ImageDraw.Draw(img)

        if outputs.ndim > 2:
            outputs = outputs.argmax(axis=2)

        outputs = np.squeeze(outputs)
        mask, mask_image = self._mask_transform(outputs)
        input = post_process_utils.RGB2YUV(input)
        mask_image = post_process_utils.RGB2YUV(mask_image)
        input[:, :, 1] = mask_image[:, :, 1]
        input[:, :, 2] = mask_image[:, :, 2]
        blend_image = post_process_utils.YUV2RGB(input)
        blend_image = blend_image.astype(np.uint8)
        blend_image = Image.fromarray(blend_image).convert("RGB")

        return mask, blend_image

    def _mask_transform(self, input):
        colors = np.asarray(post_process_utils.COLORS_LIST)
        mask = ""
        input = np.squeeze(input)
        color_img = np.zeros((input.shape[0], input.shape[1], 3), dtype=np.float32)
        height, width = input.shape
        input = np.rint(input)
        input = input.astype(np.uint8)
        for y in range(height):
            for x in range(width):
                if input[y][x] < len(colors):
                    color_img[y][x] = colors[input[y][x]]
                    mask += f"{input[y][x]} "
                else:
                    mask += f"-1 "
            mask += "\n"

        input = color_img.astype(np.uint8)
        return mask, input