from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from camera_viewer.models import PipelineStep
from camera_viewer.pipeline_ops import OPERATIONS


class VisionPipeline:
    def __init__(self, steps: Iterable[PipelineStep]) -> None:
        self.steps = list(steps)

    def apply(self, image: np.ndarray) -> np.ndarray:
        output = image.copy()
        for step in self.steps:
            if not step.enabled:
                continue
            operation = OPERATIONS.get(step.operation)
            if operation is None:
                raise ValueError(f"Unsupported operation: {step.operation}")
            output = operation(output, step.params)
        return output


def normalize_for_stream(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image
