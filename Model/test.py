from pathlib import Path

import numpy as np
import cv2
import pydicom

from ultralytics import YOLO

from segmentation.spine import Spine

DATA_DIR = Path(__file__).resolve().parent.parent / "Data/spine-segmentation/"
test_file_path: Path = DATA_DIR / "t_side.png"
print(test_file_path)
if test_file_path.suffix == ".dcm":
    pixel_array: np.ndarray = pydicom.dcmread(test_file_path).pixel_array
else:
    pixel_array: np.ndarray = cv2.cvtColor(cv2.imread(test_file_path), cv2.COLOR_BGR2GRAY)

model: YOLO = YOLO(Path(__file__).resolve().parent / "weights/best.pt")

Spine(pixel_array, None, model)