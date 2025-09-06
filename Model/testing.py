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

pixel_array_real_side, pixel_array_real_frontal = correct_heights(
    *open_png_prjs(
        '../Data/spine-segmentation/side.png', 
        '../Data/spine-segmentation/frontal.png', 
        w_part="spine", 
        new_image_size=(2304, 960)
    ),
    *open_png_prjs(
        '../Data/spine-segmentation/side.png', 
        '../Data/spine-segmentation/frontal.png', 
        w_part="hip", 
        new_image_size=(2304, 960)
    )
)
pixel_spacing = [0.18 / 0.3, 0.18 / 0.3]

from time import time

def regression_func(x, n, t):
    return (x ** t) * ((1 - x) ** (n - t))

regression_params_side = Q_REGRESSION_PARAMS(n=10, regression_func=regression_func, q_part_e=0.001, q_iter=16)
regression_params_frontal = Q_REGRESSION_PARAMS(n=10, regression_func=regression_func, q_part_e=0.001, q_iter=16)

print("starting testing side")

start = time()

borders_side = get_borders(pixel_array_real_side)
borders_frontal = get_borders(pixel_array_real_frontal)

end = time()
print("borders computed", end - start, "seconds")

start = time()

lr_side = get_lr_matrixes(pixel_array_real_side, borders_side)
lr_frontal = get_lr_matrixes(pixel_array_real_frontal, borders_frontal)

end = time()

print("lr computed", end - start, "seconds")

start = time()

A_side = get_lr_Vandermonde_matrix(0, 1, lr_side, regression_params_side.regression_func, regression_params_side.n)
A_frontal = get_lr_Vandermonde_matrix(0, 1, lr_frontal, regression_params_frontal.regression_func, regression_params_frontal.n)

end = time()

print("Vandermonde computed", end - start, "seconds")

start = time()

c_side = get_lr_q_coefs(borders_side, pixel_array_real_side, lr_side, regression_params_side, A_side)
c_frontal = get_lr_q_coefs(borders_frontal, pixel_array_real_frontal, lr_frontal, regression_params_frontal, A_frontal)

end = time()
print("coef computed", end - start, "seconds")

y_side = [
    np.dot(A_side.left, c_side.c_left.T), 
    np.dot(A_side.right, c_side.c_right.T)
]
y_side = [(y_side[0] + y_side[1]) / 2, *y_side]
y_frontal = [
    np.dot(A_frontal.left, c_frontal.c_left.T), 
    np.dot(A_frontal.right, c_frontal.c_right.T)
]
y_frontal = [(y_frontal[0] + y_frontal[1]) / 2, *y_frontal]

start = time()

vertebraes_div_lines_side = get_mean_vertebraes_dividing_lines(pixel_array_real_side, borders_side, y_side, 0.1)
vertebraes_div_lines_frontal = get_mean_vertebraes_dividing_lines(pixel_array_real_frontal, borders_frontal, y_frontal, 0.1)

end = time()
print("div lines computed", end - start, "seconds")

start = time()

points_side = compute_vertebraes_points(pixel_array_real_side, vertebraes_div_lines_side)
points_side = formating_prj_vertebraes_points(borders_side, points_side, y_side[0])
points_side = fullfill_vertebraes(borders_side, points_side, y_side[0])

points_frontal = compute_vertebraes_points(pixel_array_real_frontal, vertebraes_div_lines_frontal)
points_frontal = formating_prj_vertebraes_points(borders_frontal, points_frontal, y_frontal[0])
points_frontal = fullfill_vertebraes(borders_frontal, points_frontal, y_frontal[0])

points_frontal = link_projections(points_side, points_frontal, borders_frontal, y_frontal[0])

end = time()
print("points computed", end - start, "seconds")

# start = time()

# med_class = Medical_Parameters(reference_points)

# print(med_class._params["vertebrae"])

# end = time()
# print("medical parameters computed", end - start, "seconds")

# import json
# with open("../Results/sngl_params.json", "w") as f:
#     json.dump(med_class._single_params, f)

# writer = pd.ExcelWriter("../Results/params.xlsx", engine="xlsxwriter")
# med_class._params["vertebrae"].to_excel(writer, sheet_name="Тела позвонков")
# med_class._params["gap"].to_excel(writer, sheet_name="Межпозвоночные диски")
# med_class._params["segment"].to_excel(writer, sheet_name="Отделы")
# writer.close()

for vertebrae in points_side:
    cv2.polylines(pixel_array_real_side, [vertebrae[:, ::-1].astype(np.int32)], True, 2, 2)
    for p in vertebrae:
        pixel_array_real_side = cv2.circle(pixel_array_real_side, p[::-1].astype(np.int32), 2, 2, 3)

for vertebrae in points_frontal:
    cv2.polylines(pixel_array_real_frontal, [vertebrae[:, ::-1].astype(np.int32)], True, 2, 2)
    for p in vertebrae:
        pixel_array_real_frontal = cv2.circle(pixel_array_real_frontal, p[::-1].astype(np.int32), 2, 2, 3)

fig, ax = plt.subplots(nrows=1, ncols=2)

sns.heatmap(pixel_array_real_side, cmap="Blues", ax=ax[0])
sns.heatmap(pixel_array_real_frontal, cmap="Blues", ax=ax[1])
plt.show()
