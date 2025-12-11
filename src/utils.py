"""
utils.py

Utility functions for the Survue YOLOv8 project.
All comments are in English for grading consistency.
"""

import os
from typing import Tuple, List


def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not already exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def coco_bbox_to_yolo(
    bbox: List[float], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert a COCO-format bounding box to YOLO format.

    COCO bbox: [x_min, y_min, width, height] in absolute pixels.
    YOLO bbox: [x_center, y_center, width, height] normalized to [0, 1].

    Args:
        bbox: COCO bounding box [x, y, w, h] in pixels.
        img_width: Width of the image in pixels.
        img_height: Height of the image in pixels.

    Returns:
        (x_center, y_center, width, height) in YOLO normalized format.
    """
    x, y, w, h = bbox
    # Convert top-left + width/height -> center coordinates
    x_center = x + w / 2.0
    y_center = y + h / 2.0

    # Normalize
    x_center /= float(img_width)
    y_center /= float(img_height)
    w /= float(img_width)
    h /= float(img_height)

    return x_center, y_center, w, h


def clamp01(value: float) -> float:
    """
    Clamp a float value to the [0, 1] range.

    Args:
        value: Input value.

    Returns:
        Clamped value in [0, 1].
    """
    return max(0.0, min(1.0, value))
