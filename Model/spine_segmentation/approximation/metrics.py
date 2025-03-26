"""
    This file contains metrics for approximation functions
"""

import numpy as np
import numpy.typing as npt

import cv2

from typing import Literal

def lr_q_metric(pixel_array: npt.NDArray[np.float32], borders: npt.NDArray[np.int32], y: npt.NDArray[np.float32], mode: Literal["left", "right"], threshold = 0.5) -> float:
    """
        Computes `(1 - IOU metric)` for quantile regression over spine

        Parameters:
        -----------

        pixel_array:
            2d array of pixels - spine image
        \n
        borders:
            Bounding rectangle of the spine - `[up-left, down-right]` coords
        \n
        y:
            Approximation to be evaluated
        \n
        mode:
            For which side compute metric `[left, right]
        \n
        threshold:
            Greater than -> spine-pixel
    """

    inter = np.zeros_like(pixel_array, dtype=np.float32)

    inter = cv2.fillConvexPoly(
        inter,
        np.array([
            *[[int(y[i]), i + borders[0, 0]] for i in range(y.shape[0])],
            *([[borders[0, 1], borders[1, 0]], [borders[0, 1], borders[0, 0]]] if mode == "right" else [[borders[1, 1], borders[1, 0]], [borders[1, 1], borders[0, 0]]])
        ]),
        1.0
    )

    inter_white = np.multiply(inter, pixel_array)

    return 1 - (2 * np.sum(inter_white) / np.sum(pixel_array + inter))