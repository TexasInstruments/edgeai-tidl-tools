import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

class PostProcessClassification():
    """
    Post Process for Image Classification
    """
    def __init__(self, params: Dict = None):
        """
        Initialize the Image Classification Post Process
        
        Args:
            params (Dict)(Optional): Post process parameters
        """
        self.labels_file = None
        self.labels = None
        self.label_offset = 0
        if params:
            self.label_offset = params.get('label_offset', 0)
            self.labels_file = params.get('labels', None)
            if self.labels_file:
                try:
                    with open(self.labels_file, "r") as f:
                        self.labels = [line.strip() for line in f.readlines()]
                except:
                    print(f"[WARN] Could not parse {self.labels_file}. Using boilerplate labels for post-processing.")

    def process(self, input: Image, outputs: np.ndarray, batch: int) -> (str, Image):
        """
        Execute post processing on the image
        
        Args:
            input (Image): Input image to do post-processing on
            outputs (np.ndarray): Inference outputs
            batch (int): Output batch to pick from
            
        Returns:
            str: Detected labels and confidence in "conf - label" format"
            Image: Post Processed output image with classes overlayed
        """

        img = input.convert("RGBA")
        draw = ImageDraw.Draw(img)

        if len(outputs) == 1:
            outputs = outputs[0][batch]
        else:
            outputs = outputs[batch]

        outputs = np.squeeze(np.float32(outputs))

        top_k = outputs.argsort()[-5:][::-1]
        classes = ""
        for i, j in enumerate(top_k):
            if self.labels:
                label = self.labels[j + self.label_offset]
            else:
                label = f"Class {j + self.label_offset}"
            curr_class = f"{outputs[j]:.3f} - {label}\n"
            classes = classes + curr_class
        draw.text((0, 0), classes, fill="red")

        img = img.convert("RGB")
        return classes, img