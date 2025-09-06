from __future__ import print_function
from pathlib import Path

from spine_segmentation.segmentation.utils import get_borders, open_png_prjs
from spine_segmentation.approximation.approximation import get_lr_matrixes, get_lr_Vandermonde_matrix, get_lr_q_coefs, Q_REGRESSION_PARAMS

import cv2 as cv
import numpy as np
import random as rng

f_name = "side_1"

pixel_array_real_side, _ = open_png_prjs(
    Path(__file__).resolve().parent / f'testing/{f_name}.png', 
    Path(__file__).resolve().parent / f'testing/{f_name}.png', 
    w_part="spine",
    new_image_size=(576, 240)
)

from time import time

borders_side: np.ndarray = get_borders(pixel_array_real_side)

pixel_array_real_side = pixel_array_real_side[borders_side[0][0]: borders_side[1][0], borders_side[0][1]: borders_side[1][1]]


rng.seed(12345)
def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Find the convex hull object for each contour
    
    c = np.concatenate(contours)
    hull = cv.convexHull(c)
    # Draw contours + hull results
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color)
    cv.drawContours(drawing, [hull], 0, color)
    # Show in a window
    cv.imshow('Contours', drawing)
# Load source image


# Convert image to gray and blur it
src = (pixel_array_real_side * 255).astype(dtype=np.uint8)
src_gray = src
src_gray = cv.blur(src_gray, (3,3))
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()