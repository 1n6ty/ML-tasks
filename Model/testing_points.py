# Testing

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import pandas as pd

from spine_segmentation.segmentation.utils import get_borders, open_png_prjs, correct_heights
from spine_segmentation.approximation.approximation import get_lr_matrixes, get_lr_Vandermonde_matrix, get_lr_q_coefs
from spine_segmentation.approximation.typings import Q_REGRESSION_PARAMS
from spine_segmentation.vertebraes.rpoints import get_mean_vertebraes_dividing_lines, compute_vertebraes_points, formating_prj_vertebraes_points, fullfill_vertebraes, link_projections
from spine_segmentation.vertebraes.medical_parameters import Medical_Parameters

f_name = "side_1"

pixel_array_real_side, pixel_array_real_frontal = open_png_prjs(
        f'./testing/{f_name}.png', 
        f'./testing/{f_name}.png', 
        w_part="spine",
        new_image_size=(1152, 480)
    )

from time import time

def regression_func(x, n, t):
    return (x ** t) * ((1 - x) ** (n - t))

regression_params_side = Q_REGRESSION_PARAMS(n=10, regression_func=regression_func, q_part_e=0.001, q_iter=16)

print("starting testing side")

start = time()

borders_side = get_borders(pixel_array_real_side)

end = time()
print("borders computed", end - start, "seconds")

start = time()

lr_side = get_lr_matrixes(pixel_array_real_side, borders_side)

end = time()

print("lr computed", end - start, "seconds")

start = time()

A_side = get_lr_Vandermonde_matrix(0, 1, lr_side, regression_params_side.regression_func, regression_params_side.n)

end = time()

print("Vandermonde computed", end - start, "seconds")

start = time()

c_side = get_lr_q_coefs(borders_side, pixel_array_real_side, lr_side, regression_params_side, A_side)

end = time()
print("coef computed", end - start, "seconds")

y_side = [
    np.dot(A_side.left, c_side.c_left.T), 
    np.dot(A_side.right, c_side.c_right.T)
]
y_side = [(y_side[0] + y_side[1]) / 2, *y_side]

start = time()

vertebraes_div_lines_side = get_mean_vertebraes_dividing_lines(pixel_array_real_side, borders_side, y_side, 0.1)

end = time()
print("div lines computed", end - start, "seconds")

start = time()

points_side = compute_vertebraes_points(pixel_array_real_side, vertebraes_div_lines_side)
points_side = formating_prj_vertebraes_points(borders_side, points_side, y_side[0])

end = time()
print("points computed", end - start, "seconds")

start = time()

points_side = fullfill_vertebraes(borders_side, points_side, y_side[0])

end = time()
print("Fulfilled", end - start)

for p in points_side:
    for i in p:
        pixel_array_real_side = cv2.circle(pixel_array_real_side, i[::-1].astype(np.int32), 2, 2, 3)
sns.heatmap(pixel_array_real_side, cmap="Blues")
plt.show()
exit()
def __open_png_points(file_path_side, new_shape=None):
    global pixel_array_real_side

    points = []

    img_side = cv2.imread(file_path_side)
    
    img_side = cv2.inRange(img_side, (0, 0, 210), (40, 40, 256))

    for i in range(img_side.shape[0]):
        for j in range(img_side.shape[1]):
            if img_side[i, j] >= 127:
                if new_shape == None:
                    points.append([i, j])
                else:
                    points.append([i * new_shape[0] / img_side.shape[0], j * new_shape[1] / img_side.shape[1]])
                    pixel_array_real_side = cv2.circle(pixel_array_real_side, np.array([i * new_shape[0] / img_side.shape[0], j * new_shape[1] / img_side.shape[1]][::-1]).astype(np.int32), 2, 2.3, 3)

    return np.array(points)

for p in points_side:
    for i in p:
        pixel_array_real_side = cv2.circle(pixel_array_real_side, i[::-1].astype(np.int32), 2, 2, 3)

res = []

points_true = __open_png_points(f'./testing/{f_name}_true.png')

for p in points_side:
    for i in p:
        res.append(np.min(np.sqrt(np.sum((i - points_true) ** 2, axis=1))) * 0.19 * 2)

res = res[2:-2]

pd.DataFrame(res).to_csv("./testing/test.csv")

print(np.median(res), np.mean(res), res)
sns.heatmap(pixel_array_real_side, cmap="Blues")
plt.show()
