"""
    This file contains functions for spine-image segmentation and neural-network model
"""

import numpy as np
import numpy.typing as npt

import cv2

from typing import Literal

def __search_for_border_row(
            pixel_array: npt.NDArray[np.float32],
            borders: npt.NDArray[np.float32], 
            threshold: float, 
            x: Literal[0, 1]
        ) -> None:
    """
        Searching for row in borders `[up, down]`

        Parameters:
        -----------

        pixel_array:
            2d array of pixels - spine image
        \n
        borders:
            Array of shape(2, 2) to store results
        \n
        threshold:
            Greater than -> spine-pixel
        \n
        x:
            Mode (`0` - from top to bottom, `1` - from bottom to top)
    """
    IMG_SHAPE = pixel_array.shape
    break_flag = False
    for i in (range(IMG_SHAPE[0]) if x == 0 else range(IMG_SHAPE[0] - 1, -1, -1)):
        for j in range(IMG_SHAPE[1]):
            if pixel_array[i, j] > threshold:
                borders[x, 0] = i
                break_flag = True
                break
        if break_flag: break

def __search_for_border_col(
            pixel_array: npt.NDArray[np.float32],
            borders: npt.NDArray[np.float32], 
            threshold: float, 
            x: Literal[0, 1]
        ) -> None:
    """
        Searching for col in borders `[left, right]`

        Parameters:
        -----------

        pixel_array:
            2d array of pixels - spine image
        \n
        borders:
            Array of shape(2, 2) to store results
        \n
        threshold:
            Greater than -> spine-pixel
        \n
        x:
            Mode (`0` - from left to right, `1` - from right to left)
    """
    IMG_SHAPE = pixel_array.shape
    break_flag = False
    for i in (range(IMG_SHAPE[1]) if x == 0 else range(IMG_SHAPE[1] - 1, 0, -1)):
        for j in range(IMG_SHAPE[0]):
            if pixel_array[j, i] > threshold:
                borders[x, 1] = i
                break_flag = True
                break
        if break_flag: break

from multiprocessing import Process

def get_borders(pixel_array: npt.NDArray[np.float32], threshold = 0.5) -> npt.NDArray[np.int32]: 
    """
        Searching for bounding rectagle coordinates `[up-left, down-right]` of the spine. `All edges included!`
        
        Parameters:
        -----------

        pixel_array:
            2d array of pixels - spine image
        \n
        threshold:
            Greater than -> spine-pixel
    """
    borders = np.zeros((2, 2), dtype=np.int32) # [[up-left] and [down-right] coords]

    processes = []

    for params in [(pixel_array, borders, threshold, 0), (pixel_array, borders, threshold, 1)]:
        processes.append(Process(target=__search_for_border_row, args=params))
        processes.append(Process(target=__search_for_border_col, args=params))

        processes[-2].start()
        processes[-1].start()
    
    for p in processes:
        p.join()
    
    return borders

def height_corr_frontal(pixel_array_frontal: npt.NDArray[np.float32], pixel_array_side_hip: npt.NDArray[np.float32], pixel_array_frontal_hip: npt.NDArray[np.float32], threshold = 0.5) -> npt.NDArray[np.float32]:
    """
        Height correction for frontal image via hip position

        Parameters:
        -----------

        pixel_array_frontal:
            2d array of pixels - spine image (frontal)
        \n
        pixel_array_side_hip:
            2d array of pixels - hip image (side)
        \n
        pixel_array_frontal_hip:
            2d array of pixels - hip image (frontal)
        \n
        `They are all the same size!`
        \n
        threshold:
            Greater than -> spine-pixel
    """
    x_side, x_front = 0, 0
    sum_side, sum_front = 0, 0
    for i in pixel_array_side_hip.shape[0]:
        for j in range(pixel_array_side_hip.shape[1]):
            if pixel_array_side_hip[i, j] >= threshold:
                x_side += i
                sum_side += 1
            if pixel_array_frontal_hip[i, j] >= threshold:
                x_front += i
                sum_front += 1
    height_corr = int(x_side / sum_side - x_front / sum_front)
    
    if height_corr > 0:
        return cv2.copyMakeBorder(pixel_array_frontal, 0, height_corr, 0, 0, cv2.BORDER_CONSTANT, 0)[:-height_corr, :]
    else:
        return cv2.copyMakeBorder(pixel_array_frontal, -height_corr, 0, 0, 0, cv2.BORDER_CONSTANT, 0)[-height_corr: , :]