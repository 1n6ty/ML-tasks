# Testing

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import os
import pydicom
from typing import TypedDict, Callable, Literal

f_name = "001_SD"
DATA_DIR = os.path.abspath('../Data/spine-segmentation')

def __open_png(file_path_side, new_image_size = None):
    img_side = cv2.imread(file_path_side)
    
    img_side = cv2.inRange(img_side, (10, 10, 10), (256, 256, 256))

    if new_image_size != None:
        img_side = cv2.resize(img_side, new_image_size[::-1], interpolation=cv2.INTER_CUBIC)

    return np.array(img_side, dtype="float32") / np.max(img_side)

pixel_array_real = __open_png(f'../Data/spine-segmentation/niito/{f_name}.png')

from spine_segmentation.segmentation.funcs import get_borders
from spine_segmentation.approximation.approximation import get_lr_matrixes, get_lr_Vandermonde_matrix, get_lr_q_coefs
from spine_segmentation.approximation.typings import Q_REGRESSION_PARAMS, VANDERMONDE_MATRIXES

print("starting testing")

from time import time

def regression_func(x, t, n):
    return (x ** t) * ((1 - x) ** (n - t))

regression_params = Q_REGRESSION_PARAMS(n=10, regression_func=regression_func, q_part_e=0.1, q_iter=16)

start = time()

borders = get_borders(pixel_array_real)

end = time()
print("borders computed", end - start, "seconds")

start = time()

lr = get_lr_matrixes(pixel_array_real, borders)

end = time()

print("lr computed", end - start, "seconds")

start = time()

A = get_lr_Vandermonde_matrix(0, 1, lr, regression_params.regression_func, regression_params.n)

end = time()

print("Vandermonde computed", end - start, "seconds")

start = time()

c = get_lr_q_coefs(borders, pixel_array_real, lr, regression_params, A)

end = time()
print("coef computed", end - start, "seconds")

