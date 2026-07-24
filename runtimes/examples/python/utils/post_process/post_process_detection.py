import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
curr_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curr_dir)
import post_process_utils

class PostProcessDetection():
    """
    Post Process for Object Detection
    """
    def __init__(self, params: Dict):
        """
        Initialize the Object Detection Post Process
        
        Args:
            params (Dict): Post process parameters
        """
        self.framework = params.get('framework', None)
        self.od_type = params.get('od_type', None)
        if self.framework is None and self.od_type is None:
            raise ValueError(f"[ERROR] params is missing 'framework' or 'od_type'")
        
        supported_framework = ['MMDetection']
        supported_od_type = ['SSD', 'YoloV5', 'HasDetectionPostProcLayer', 'EfficientDetLite']

        if self.framework:
            self.framework = self.framework.strip()
            if self.framework != '' and self.framework not in supported_framework:
                raise ValueError(f"[ERROR] {self.framework} self.framework is not currently supported for post processing. Supported frameworks: {', '.join(supported_framework)}")

        elif self.od_type:
            self.od_type = self.od_type.strip()
            if self.od_type != '' and self.od_type not in supported_od_type:
                raise ValueError(f"[ERROR] {self.od_type} self.od_type is not currently supported for post processing. Supported od_type: {', '.join(supported_od_type)}")
    
    def process(self, input: Image, outputs: np.ndarray, batch: int) -> (str, Image):
        """
        Execute post processing on the image
        
        Args:
            input (Image): Input image to do post-processing on
            outputs (np.ndarray): Inference outputs
            batch (int): Output batch to pick from
            
        Returns:
            str: Detected bound boxes and confidence in "conf - [xmin, ymin, xmax, ymax]" format"
            Image: Post Processed output image with detection box overlayed
        """
        
        img = input.convert("RGBA")
        draw = ImageDraw.Draw(img)
        detections = ""

        # MMDetection
        if self.framework == "MMDetection":
            outputs = [np.squeeze(output_i) for output_i in outputs]
            if len(outputs[0].shape) == 2:
                num_boxes = int(outputs[0].shape[0])
                for i in range(num_boxes):
                    conf = outputs[0][i][4]
                    xmin = int(outputs[0][i][0])
                    ymin = int(outputs[0][i][1])
                    xmax = int(outputs[0][i][2])
                    ymax = int(outputs[0][i][3])
                    label = int(outputs[1][i])
                    if conf > 0.3:
                        color = post_process_utils.COLORS_LIST[label % len(post_process_utils.COLORS_LIST)]
                        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=2)
                        detections += f"{conf:.3f} - {xmin, ymin, xmax, ymax} - {label}\n"

            elif len(outputs[0].shape) == 1:
                num_boxes = 1
                for i in range(num_boxes):
                    conf = outputs[i][4]
                    xmin = int(outputs[i][0])
                    ymin = int(outputs[i][1])
                    xmax = int(outputs[i][2])
                    ymax = int(outputs[i][3])
                    label = int(outputs[i])
                    if conf > 0.3:
                        color = post_process_utils.COLORS_LIST[label % len(post_process_utils.COLORS_LIST)]
                        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=2)
                        detections += f"{conf:.3f} - {xmin, ymin, xmax, ymax} - {label}\n"
        
        # SSD
        elif self.od_type == "SSD":
            outputs = [np.squeeze(output_i) for output_i in outputs]
            num_boxes = int(outputs[0].shape[0])
            for i in range(num_boxes):
                conf = outputs[2][i]
                xmin = int(outputs[0][i][0] * img.width)
                ymin = int(outputs[0][i][1] * img.height)
                xmax = int(outputs[0][i][2] * img.width)
                ymax = int(outputs[0][i][3] * img.height)
                label = int(outputs[1][i])
                if conf > 0.3:
                    color = post_process_utils.COLORS_LIST[label % len(post_process_utils.COLORS_LIST)]
                    draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=2)
                    detections += f"{conf:.3f} - {xmin, ymin, xmax, ymax} - {label}\n"
        # YoloV5
        elif self.od_type == "YoloV5":
            outputs = [np.squeeze(output_i) for output_i in outputs]
            num_boxes = int(outputs[0].shape[0])
            for i in range(num_boxes):
                conf = outputs[0][i][4]
                xmin = int(outputs[0][i][0])
                ymin = int(outputs[0][i][1])
                xmax = int(outputs[0][i][2])
                ymax = int(outputs[0][i][3])
                label = int(outputs[0][i][5])
                if conf > 0.3:
                    color = post_process_utils.COLORS_LIST[label % len(post_process_utils.COLORS_LIST)]
                    draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=2)
                    detections += f"{conf:.3f} - {xmin, ymin, xmax, ymax} - {label}\n"

        # Model has detection post processing layer
        elif self.od_type == "HasDetectionPostProcLayer":
            for i in range(int(outputs[3][0])):
                conf = outputs[2][0][i]
                ymin = int(outputs[0][0][i][0] * img.height)
                xmin = int(outputs[0][0][i][1] * img.width)
                ymax = int(outputs[0][0][i][2] * img.height)
                xmax = int(outputs[0][0][i][3] * img.width)
                label = int(outputs[1][0][i])
                if conf > 0.3:
                    color = post_process_utils.COLORS_LIST[label % len(post_process_utils.COLORS_LIST)]
                    draw.rectangle(((xmin,ymin), (xmax,ymax)), outline=color, width=2)
                    detections += f"{conf:.3f} - {xmin, ymin, xmax, ymax} - {label}\n"

        # Model does not have detection post processing layer
        elif self.od_type == "EfficientDetLite":
            for i in range(int(outputs[0].shape[1])):
                conf = outputs[0][0][i][5]
                ymin = int(outputs[0][0][i][1])
                xmin = int(outputs[0][0][i][2])
                ymax = int(outputs[0][0][i][3])
                xmax = int(outputs[0][0][i][4])
                label = int(outputs[0][0][i][6])
                if conf > 0.3:
                    color = post_process_utils.COLORS_LIST[label % len(post_process_utils.COLORS_LIST)]
                    draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=2)
                    detections += f"{conf:.3f} - {xmin, ymin, xmax, ymax} - {label}\n"

        img = img.convert("RGB")
        return detections, img