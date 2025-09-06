"""
    This file contains functions for spine-image segmentation and neural-network model
"""

import numpy as np
import numpy.typing as npt

import cv2
import pydicom

from pathlib import Path
from typing import Literal

from multiprocessing import Process, Manager
from multiprocessing.managers import DictProxy

from ultralytics.models import YOLO

def __search_for_border_row(
            pixel_array: npt.NDArray[np.float32],
            manager_dict: DictProxy,
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
        manager_dict:
            Sync dictionary to store results
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
                manager_dict["bottom" if x == 1 else "top"] = i
                break_flag = True
                break
        if break_flag: break

def __search_for_border_col(
            pixel_array: npt.NDArray[np.float32],
            manager_dict: DictProxy,
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
        manager_dict:
            Sync dictionary to store results
        \n
        threshold:
            Greater than -> spine-pixel
        \n
        x:
            Mode (`0` - from left to right, `1` - from right to left)
    """
    IMG_SHAPE = pixel_array.shape
    break_flag = False
    for i in (range(IMG_SHAPE[1]) if x == 0 else range(IMG_SHAPE[1] - 1, -1, -1)):
        for j in range(IMG_SHAPE[0]):
            if pixel_array[j, i] > threshold:
                manager_dict["right" if x == 1 else "left"] = i
                break_flag = True
                break
        if break_flag: break

def get_borders(pixel_array: npt.NDArray[np.float32], threshold = 0.5) -> npt.NDArray[np.int32]: 
    """
        Searching for bounding rectagle coordinates `[up-left, down-right]` of the spine. `All edges included!`
        
        Parameters
        -----------
        pixel_array:
            2d array of pixels - spine image
        \n
        threshold:
            Greater than -> spine-pixel
    """
    processes = []
    manager_dict = Manager().dict()

    for params in [(pixel_array, manager_dict, threshold, 0), (pixel_array, manager_dict, threshold, 1)]:
        processes.append(Process(target=__search_for_border_row, args=params))
        processes.append(Process(target=__search_for_border_col, args=params))

        processes[-2].start()
        processes[-1].start()
    
    for p in processes:
        p.join()
    
    return np.array(
        [[manager_dict["top"], manager_dict["left"]], [manager_dict["bottom"], manager_dict["right"]]], dtype=np.int32
    )

def to_same_size(img_side: np.ndarray[np.float32], img_frontal: np.ndarray[np.float32]) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
    """
        Pads images to the size of the biggest and reshapes if `new_image_size` is provided

        Parameters:
        -----------
            img_side:
                Pixel array of the side projection
            \n
            img_frontal:
                Pixel array of the frontal projection
    """
    d = int((img_frontal.shape[0] - img_side.shape[0]) / 2)
    cv2.copyMakeBorder((img_side if d > 0 else img_frontal), abs(d), abs(d), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))

    d = int((img_frontal.shape[1] - img_side.shape[1]) / 2)
    cv2.copyMakeBorder((img_side if d > 0 else img_frontal), 0, 0, abs(d), abs(d), cv2.BORDER_CONSTANT, (0, 0, 0))

    return (img_side, img_frontal)

def open_png_prj(file_path: str, w_part: Literal["spine", "hip"] = "spine", new_image_size: tuple[int, int] | None = None) -> np.ndarray[np.float32]:
    """
        Opens png-files of ONE projection to numpy binary array of float32\n
        Provided `spine - red region` and `hip - blue region` modes

        Parameters
        -----------
            file_path_side:
                Path to png file with side projection
            \n
            w_part:
                Watched part - `hip | spine`
            \n
            new_image_size:
                Reshaping size 
    """
    img = cv2.imread(file_path)
    lf, hf = [(0, 0, 210), (40, 40, 256)] if w_part == "spine" else [(210, 0, 0), (256, 40, 40)]
    img = cv2.inRange(img, lf, hf)

    if new_image_size != None:
        img = cv2.resize(img, new_image_size[::-1], interpolation=cv2.INTER_CUBIC)

    return img.astype(np.float32) / np.max(img)

def open_png_prjs(file_path_side: str, file_path_frontal: str, w_part: Literal["spine", "hip"] = "spine", new_image_size: tuple[int, int] | None = None) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
    """
        Opens png-files of projections to numpy binary array of float32\n
        Provided `spine - red region` and `hip - blue region` modes

        Parameters
        -----------
            file_path_side:
                Path to png file with side projection
            \n
            file_path_frontal:
                Path to png file with frontal projection
            \n
            w_part:
                Watched part - `hip | spine`
            \n
            new_image_size:
                Reshaping size 
    """
    img_side = open_png_prj(file_path_side, w_part, new_image_size)
    img_frontal = open_png_prj(file_path_frontal, w_part, new_image_size)

    img_side, img_frontal = to_same_size(img_side, img_frontal)

    return (img_side, img_frontal)

def open_dcm_prj(file_path: str, new_image_size: tuple[int, int] | None = None) -> np.ndarray[np.float32]:
    """
        Opens dicom-file of ONE projection to numpy grayscale `[from 0 to 1]` array of float32\n

        Parameters
        -----------
            file_path_side:
                Path to dicom file with side projection
            \n
            new_image_size:
                Reshaping size 
    """
    img = pydicom.dcmread(file_path).pixel_array

    if new_image_size != None:
        img = cv2.resize(img, new_image_size[::-1], interpolation=cv2.INTER_CUBIC)

    return img.astype(np.float32) / np.max(img)

def open_dcm_prjs(file_path_side: str, file_path_frontal: str, new_image_size: tuple[int, int] | None = None) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
    """
        Opens dicom-files of projections to numpy grayscale `[from 0 to 1]` array of float32\n

        Parameters
        -----------
            file_path_side:
                Path to dicom file with side projection
            \n
            file_path_frontal:
                Path to dicom file with frontal projection
            \n
            new_image_size:
                Reshaping size 
    """
    img_side = open_dcm_prj(file_path_side, new_image_size)
    img_frontal = open_dcm_prj(file_path_frontal, new_image_size)
    
    img_side, img_frontal = to_same_size(img_side, img_frontal)

    return (img_side, img_frontal)

def correct_heights(img_side: np.ndarray[np.float32], img_frontal: np.ndarray[np.float32], img_side_hip: np.ndarray[np.float32], img_frontal_hip: np.ndarray[np.float32]) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
    """
        Returns corrected spine side and frontal projections, based on height of hip

        Parameters
        -----------
            img_side:
                Pixel array of the spine side projection
            \n
            img_frontal:
                Pixel array of the spine frontal projection
            \n
            img_side_hip:
                Pixel array of the hip side projection
            \n
            img_frontal_hip:
                Pixel array of the hip frontal projection
    """
    x_side = np.median([i for i in range(img_side_hip.shape[0]) for j in range(img_side_hip.shape[1]) if img_side_hip[i, j] > 0])
    x_frontal = np.median([i for i in range(img_frontal_hip.shape[0]) for j in range(img_frontal_hip.shape[1]) if img_frontal_hip[i, j] > 0])

    diff = int(x_frontal - x_side)
    if diff > 0:
        img_frontal_ranged = cv2.copyMakeBorder(img_frontal, 0, diff, 0, 0, cv2.BORDER_CONSTANT, 0.0)[diff:, :]
    else:
        img_frontal_ranged = cv2.copyMakeBorder(img_frontal, -diff, 0, 0, 0, cv2.BORDER_CONSTANT, 0.0)[:diff, :]
    
    return (np.array(img_side, dtype=np.float32), np.array(img_frontal_ranged, dtype=np.float32))

def yolo_segment_vertebraes_from_img(img_path: Path | str, weights_path: Path | str, imgsz: list[int] = [4608, 1920]) -> np.ndarray[np.float32]:
    """
        Returns segmented vertebraes on image as a numpy mask

        Parameters
        ----------
            img_path
                Path to image to be segmented
            \n
            weights_path
                Path to YOLO weights
    """
    model = YOLO(weights_path)

    results = model.predict(img_path, imgsz=imgsz, conf=0.5, show=False, show_labels=False, show_boxes=False)

    res = np.zeros(imgsz, dtype=np.int32)
    for v in results[0].masks.xy:
        cv2.fillPoly(res, np.int32([v]), True, 255, 1)
    
    return res