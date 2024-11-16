import cv2
import numpy as np
import os
from func import *

import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.abspath('../Data/spine-segmentation/')

def regression_func(x, n, t):
    return (x ** t) * ((1 - x) ** (n - t))

regression_params = {
    "n": 20,
    "regression_func": regression_func,
    "quantile_part_e": 0.0001,
    "quantile_iterations": 16
}

threshold_e = 0.00001
f_name = "001_SD"
ind = 0

def __open_png(file_path_side, new_image_size = None):
    img_side = cv2.imread(file_path_side)
    
    img_side = cv2.inRange(img_side, (10, 10, 10), (256, 256, 256))

    if new_image_size != None:
        img_side = cv2.resize(img_side, new_image_size[::-1], interpolation=cv2.INTER_CUBIC)

    return np.array(img_side, dtype="float32") / np.max(img_side)

pixel_array_real = __open_png(f'../Data/spine-segmentation/niito/{f_name}.png')

#pixel_array_side = np.squeeze(np.load(os.path.join("../Results/", "model_tresult.npy"))[3][ind])
pixel_array_side = __open_png(f'../Data/spine-segmentation/filled/{f_name}.png')
pixel_array_side = np.where(pixel_array_side >= 0.5, np.ones_like(pixel_array_side, dtype=np.float32), np.zeros_like(pixel_array_side, dtype=np.float32))

borders_side = get_borders(pixel_array_side)
A_side = get_Vandermond_matrix(0, 1, borders_side[1, 0] + 1 - borders_side[0, 0], regression_params["n"], regression_params["regression_func"])

matrixes_side = get_lr_matrixes(pixel_array_side, borders_side)

from multiprocessing import Pool
import pandas as pd

ps = pd.read_csv(os.path.join(DATA_DIR, "pixel_spacings.csv"), delimiter=';')
pixel_spacing = np.array(list(map(float, ps[ps["dicom filename"] == (f_name + '.dcm')].iloc[0]["dicom file field:(0028,0030) ImagerPixelSpacing = [Row Spacing, Column Spacing]"][1:-1].split(', '))), dtype=np.float32)

pixel_spacing_real = np.copy(pixel_spacing)
pixel_spacing = np.multiply(pixel_spacing, np.divide(np.array(pixel_array_real.shape), np.array(pixel_array_side.shape), dtype=np.float32))

print(pixel_spacing)
def compute_regression_metric(borders, pixel_array, y, mode):
    inter = np.copy(pixel_array)
    for i in range(y.shape[0]):
        for j in (range(max(int(y[i]), borders[0, 1]), borders[1, 1] + 1) if mode == "left" else range(borders[0, 1], min(int(y[i]) + 1, borders[1, 1] + 1))):
            if inter[i + borders[0, 0], j] > 0.5:
                inter[i + borders[0, 0], j] = 2
            else:
                inter[i + borders[0, 0], j] = -2

    IOU_white = np.sum(np.where(inter > 1.5, 1, 0)) / np.sum(pixel_array)
    #IOU_black = 1 - np.sum(np.where(inter < -1.5, 1, 0)) / (y.shape[0] * (borders[1, 1] - borders[0, 1] + 1))
                
    #return 2 - (IOU_white + IOU_black)
    return 1 - IOU_white

def get_lr_quantilepart(matrixes: tuple[np.ndarray, np.ndarray], iter: int, q_part: float, A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes quantile regression through iter iterations over q_part SquareError.\n
    A - Vandermont matrix
    """
    y_l, W_l = matrixes

    coef_l= np.dot(np.linalg.pinv(np.dot(np.dot(A.T, W_l), A)), np.dot(np.dot(A.T, W_l), y_l.T))
    y_l_q = np.dot(A, coef_l.T)

    for _ in range(iter):
        se_l = (y_l_q - y_l) ** 2

        i = min(int(se_l.shape[0] * (1 - q_part)), se_l.shape[0] - 1)
        se_q = np.sort(se_l)[i]

        W_tmp = np.where(se_l < se_q, 1 - q_part, q_part)
        
        W_l = np.multiply(np.diag(W_tmp), W_l)

        coef_l = np.dot(np.linalg.pinv(np.dot(np.dot(A.T, W_l), A)), np.dot(np.dot(A.T, W_l), y_l.T))
        y_l_q = np.dot(A, coef_l.T)

    return (y_l_q, W_l)

def get_quantile_metric(borders, pixel_array, regression_params, matrixes, A, t, mode):
    y, W = get_lr_quantilepart(matrixes, regression_params["quantile_iterations"], t, A)
    return compute_regression_metric(borders, pixel_array, y, mode)

def get_quantile_part_brute(borders, pixel_array, regression_params, matrixes, A, mode):
    e_log = int(np.log10(regression_params["quantile_part_e"])) * -1
    init = 0
    for t in range(1, e_log + 1):
        power = 10 ** -t
        gen = range(1, 10) if power == 0.1 else range(-9, 10)
        metrics = []
        with Pool() as p:
            metrics = p.starmap(get_quantile_metric, [(borders, pixel_array, regression_params, matrixes, A, init + t * power, mode) for t in gen])
        min_metric = [metrics[0], 0]
        for m in range(1, len(metrics)):
            if min_metric[0] > metrics[m]: min_metric = [metrics[m], m]

        init += gen[min_metric[1]] * power

    return init

def get_radius(y, pivot, delta):
    y_m, y_l, y_r = [i.astype(dtype=np.float32) for i in y]
    left_coords, right_coords = np.copy(pivot), np.copy(pivot)

    reach_edge = [False, False] # indicator of reaching the edge [left, right] by line
    while not all(reach_edge):
        if not reach_edge[0]:
            if (0 <= left_coords[0] < y_m.shape[0]) and (y_l[int(left_coords[0])] < left_coords[1]):
                left_coords += (-1 if delta[1] > 0 else 1) * delta
            else:
                reach_edge[0] = True

        if not reach_edge[1]:
            if (0 <= right_coords[0] < y_m.shape[0]) and (right_coords[1] < y_r[int(right_coords[0])]):
                right_coords += (1 if delta[1] > 0 else -1) * delta
            else:
                reach_edge[1] = True

    return np.max([
        np.sqrt(np.sum((pivot - left_coords) ** 2)),
        np.sqrt(np.sum((pivot - right_coords) ** 2))
    ])

def get_rad_list(y):
    gamma = 1e-6 # used as gradient of a straight horizontal line
    y_m, y_l, y_r = [i.astype(dtype=np.float32) for i in y]
    
    dy_m = np.concatenate([y_m[1:], y_m[-1:]]) - y_m; dy_m = np.where(dy_m != 0, dy_m, np.full_like(dy_m, gamma))
    grad = np.divide(-np.ones_like(y_m, dtype=np.float32), dy_m) # tan of normal

    rad = []

    for x in range(grad.shape[0]):
        delta = np.array([1.0, grad[x]], dtype=np.float32); delta = delta / np.max(np.abs(delta))
        pivot = np.array([x, y_m[x]], dtype=np.float32)

        rad.append(get_radius(y, pivot, delta))

    return np.array(rad)


def compute_regression_vertebra_metric(vertebra, side_x, borders, pixel_array, y, mode):
    if not (0 <= y[0] < pixel_array.shape[1] and 0 <= y[-1] < pixel_array.shape[1] and y.shape[0] > 1):
        return float("inf")

    inter = np.zeros_like(pixel_array)
    vertebra_c = np.copy(vertebra)

    inter_v = np.zeros_like(pixel_array)
    cv2.fillConvexPoly(inter_v, np.array([*vertebra_c[0: 2][::, ::-1].astype(np.int32), *vertebra_c[2: 4][::-1, ::-1].astype(np.int32)]), 1)
    vertebra_c = np.copy(vertebra)
    
    if mode == "left":
        pivot_a = np.copy(vertebra[0])
        delta_a = vertebra[1] - vertebra[0]

        pivot_b = np.array([side_x[0] + borders[0, 0], y[0]])
        delta_b = np.array([side_x[2] + borders[0, 0], y[-1]]) - pivot_b
        if pivot_a[0] == pivot_b[0] and (delta_a[0] == 0 or pivot_a[1] == pivot_b[1]):
            vertebra_c[0] = pivot_b
        else:
            t = (pivot_b[0] + (delta_b[0] / (delta_b[1] + 1e-6)) * (pivot_a[1] - pivot_b[1]) - pivot_a[0]) / (delta_a[0] * (1 - (delta_b[0] * delta_a[1]) / (delta_b[1] * delta_a[0] + 1e-6)) + 1e-6)
            vertebra_c[0] = pivot_a + t * delta_a

        pivot_a = np.copy(vertebra[2])
        delta_a = vertebra[3] - vertebra[2]

        pivot_b = np.array([side_x[0] + borders[0, 0], y[0]])
        delta_b = np.array([side_x[2] + borders[0, 0], y[-1]]) - pivot_b
        if pivot_a[0] == (pivot_b + delta_b)[0] and (delta_a[0] == 0 or pivot_a[1] == (pivot_b + delta_b)[1]):
            vertebra_c[2] = pivot_b + delta_b
        else:
            t = (pivot_b[0] + (delta_b[0] / (delta_b[1] + 1e-6)) * (pivot_a[1] - pivot_b[1]) - pivot_a[0]) / (delta_a[0] * (1 - (delta_b[0] * delta_a[1]) / (delta_b[1] * delta_a[0] + 1e-6)) + 1e-6)
            vertebra_c[2] = pivot_a + t * delta_a
    else:
        pivot_a = np.copy(vertebra[0])
        delta_a = vertebra[1] - vertebra[0]

        pivot_b = np.array([side_x[1] + borders[0, 0], y[0]])
        delta_b = np.array([side_x[3] + borders[0, 0], y[-1]]) - pivot_b
        if pivot_a[0] == pivot_b[0] and (delta_a[0] == 0 or pivot_a[1] == pivot_b[1]):
            vertebra_c[1] = pivot_b
        else:
            t = (pivot_b[0] + (delta_b[0] / (delta_b[1] + 1e-6)) * (pivot_a[1] - pivot_b[1]) - pivot_a[0]) / (delta_a[0] * (1 - (delta_b[0] * delta_a[1]) / (delta_b[1] * delta_a[0] + 1e-6)) + 1e-6)
            vertebra_c[1] = pivot_a + t * delta_a

        pivot_a = np.copy(vertebra[2])
        delta_a = vertebra[3] - vertebra[2]

        pivot_b = np.array([side_x[1] + borders[0, 0], y[0]])
        delta_b = np.array([side_x[3] + borders[0, 0], y[-1]]) - pivot_b
        if pivot_a[0] == (pivot_b + delta_b)[0] and (delta_a[0] == 0 or pivot_a[1] == (pivot_b + delta_b)[1]):
            vertebra_c[3] = pivot_b + delta_b
        else:
            t = (pivot_b[0] + (delta_b[0] / (delta_b[1] + 1e-6)) * (pivot_a[1] - pivot_b[1]) - pivot_a[0]) / (delta_a[0] * (1 - (delta_b[0] * delta_a[1]) / (delta_b[1] * delta_a[0] + 1e-6)) + 1e-6)
            vertebra_c[3] = pivot_a + t * delta_a
    
    cv2.fillConvexPoly(inter, np.array([*vertebra_c[0: 2][::, ::-1].astype(np.int32), *vertebra_c[2: 4][::-1, ::-1].astype(np.int32)]), 1)

    IOU_white = np.sum(np.where(inter + pixel_array > 1.5, 1, 0)) / np.sum(np.where(inter_v + pixel_array > 1.5, 1, 0))
    IOU_black = (np.sum(inter) - np.sum(np.where(inter + pixel_array > 1.5, 1, 0))) / (np.sum(inter_v) - np.sum(np.where(inter_v + pixel_array > 1.5, 1, 0)))

    return 1 - IOU_white + IOU_black

def get_quantile_metric_vertebra(vertebra, side_x, borders, pixel_array, regression_params, matrixes, A, t, mode):
    y_new = get_lr_quantilepart(matrixes, regression_params["quantile_iterations"], t, A)[0]

    return compute_regression_vertebra_metric(vertebra, side_x, borders, pixel_array, y_new, mode)

def get_quantile_vertebra_side_part_brute(vertebra, side_x, borders, pixel_array, regression_params, matrixes, A, mode):
    e_log = int(np.log10(regression_params["quantile_part_e"])) * -1
    init = 0
    for t in range(1, e_log + 1):
        power = 10 ** -t
        gen = range(1, 10) if power == 0.1 else range(-9, 10)
        metrics = []
        with Pool() as p:
            metrics = p.starmap(get_quantile_metric_vertebra, [(vertebra, side_x, borders, pixel_array, regression_params, matrixes, A, init + t * power, mode) for t in gen])
        min_metric = [metrics[0], 0]
        for m in range(1, len(metrics)):
            if min_metric[0] >= metrics[m]: min_metric = [metrics[m], m]

        init += gen[min_metric[1]] * power

    return init

def adjust_corners(matrixes, quantile_matrixes, vertebras_corners, borders, pixel_array, regression_params, start, finish, strip=True):
    y_l, y_r = np.dot(matrixes[0], matrixes[1]) + np.dot(quantile_matrixes[0], np.where(matrixes[1] < 0.5, np.diag(np.ones_like(matrixes[0])), 0)), np.dot(matrixes[2], matrixes[3]) + np.dot(quantile_matrixes[2], np.where(matrixes[3] < 0.5, np.diag(np.ones_like(matrixes[2])), 0))
    matrixes = (y_l, np.diag(np.ones_like(y_l)), y_r, np.diag(np.ones_like(y_r)))
    for v in range(start, finish, 4):
        pivot_1 = np.copy(vertebras_corners[v])

        side_x_fst = [None, None]

        delta_1 = vertebras_corners[v + 1] - vertebras_corners[v]
        if delta_1[0] == 0:
            side_x_fst = [int(vertebras_corners[v][0] - borders[0, 0]), int(vertebras_corners[v + 1][0] - borders[0, 0])]
        else:
            delta_1 /= np.max(np.abs(delta_1))

            while (pivot_1[1] <= vertebras_corners[v + 1][1]) and (side_x_fst[0] == None or side_x_fst[1] == None):
                if side_x_fst[0] == None and pivot_1[1] >= quantile_matrixes[0][int(pivot_1[0] - borders[0, 0])]:
                    side_x_fst[0] = int(pivot_1[0] - borders[0, 0])
                if side_x_fst[1] == None and pivot_1[1] >= quantile_matrixes[2][int(pivot_1[0] - borders[0, 0])]:
                    side_x_fst[1] = int(pivot_1[0] - borders[0, 0])
                pivot_1 += delta_1

        pivot_2 = np.copy(vertebras_corners[v + 2])

        side_x_sec = [None, None]

        delta_2 = vertebras_corners[v + 3] - vertebras_corners[v + 2]
        if delta_2[0] == 0:
            side_x_sec = [int(vertebras_corners[v + 2][0] - borders[0, 0]), int(vertebras_corners[v + 3][0] - borders[0, 0])]
        else:
            delta_2 /= np.max(np.abs(delta_2))

            while (pivot_2[1] <= vertebras_corners[v + 3][1]) and (side_x_sec[0] == None or side_x_sec[1] == None):
                if side_x_sec[0] == None and pivot_2[1] >= quantile_matrixes[0][int(pivot_2[0] - borders[0, 0])]:
                    side_x_sec[0] = int(pivot_2[0] - borders[0, 0])
                if side_x_sec[1] == None and pivot_2[1] >= quantile_matrixes[2][int(pivot_2[0] - borders[0, 0])]:
                    side_x_sec[1] = int(pivot_2[0] - borders[0, 0])
                pivot_2 += delta_2
        
        side_x = [*side_x_fst, *side_x_sec]

        buff_l = matrixes[0][side_x_fst[0]: side_x_sec[0] + 1]
        buff_r = matrixes[2][side_x_fst[1]: side_x_sec[1] + 1]

        W_l = matrixes[1][side_x_fst[0]: side_x_sec[0] + 1, side_x_fst[0]: side_x_sec[0] + 1]
        W_r = matrixes[3][side_x_fst[1]: side_x_sec[1] + 1, side_x_fst[1]: side_x_sec[1] + 1]

        A_left = get_Vandermond_matrix(0, 1, buff_l.shape[0], (2 if v == vertebras_corners.shape[0] - 4 else 1), lambda x, n, t: x ** t)
        A_right = get_Vandermond_matrix(0, 1, buff_r.shape[0], (2 if v == vertebras_corners.shape[0] - 4 else 1), lambda x, n, t: x ** t)

        l_coef = get_quantile_vertebra_side_part_brute(vertebras_corners[v: v + 4], side_x, borders, pixel_array, regression_params, (buff_l, W_l), A_left, "left")
        r_coef = get_quantile_vertebra_side_part_brute(vertebras_corners[v: v + 4], side_x, borders, pixel_array, regression_params, (buff_r, W_r), A_right, "right")

        y_l_new = get_lr_quantilepart((buff_l, W_l), regression_params["quantile_iterations"], l_coef, A_left)[0]
        y_r_new = get_lr_quantilepart((buff_r, W_r), regression_params["quantile_iterations"], r_coef, A_right)[0]

        pivot_a = np.copy(vertebras_corners[v])
        delta_a = vertebras_corners[v + 1] - vertebras_corners[v]

        pivot_b = np.array([side_x_fst[0] + borders[0, 0], y_l_new[0]])
        delta_b = np.array([side_x_sec[0] + borders[0, 0], y_l_new[-1]]) - pivot_b
        if pivot_a[0] == pivot_b[0] and (delta_a[0] == 0 or pivot_a[1] == pivot_b[1]):
            vertebras_corners[v] = pivot_b
        else:
            t = (pivot_b[0] + (delta_b[0] / (delta_b[1] + 1e-6)) * (pivot_a[1] - pivot_b[1]) - pivot_a[0]) / (delta_a[0] * (1 - (delta_b[0] * delta_a[1]) / (delta_b[1] * delta_a[0] + 1e-6)) + 1e-6)
            vertebras_corners[v] = pivot_a + t * delta_a

        pivot_b = np.array([side_x_fst[1] + borders[0, 0], y_r_new[0]])
        delta_b = np.array([side_x_sec[1] + borders[0, 0], y_r_new[-1]]) - pivot_b
        if pivot_a[0] == pivot_b[0] and (delta_a[0] == 0 or pivot_a[1] == pivot_b[1]):
            vertebras_corners[v + 1] = pivot_b
        else:
            t = (pivot_b[0] + (delta_b[0] / (delta_b[1] + 1e-6)) * (pivot_a[1] - pivot_b[1]) - pivot_a[0]) / (delta_a[0] * (1 - (delta_b[0] * delta_a[1]) / (delta_b[1] * delta_a[0] + 1e-6)) + 1e-6)
            vertebras_corners[v + 1] = pivot_a + t * delta_a

        pivot_a = np.copy(vertebras_corners[v + 2])
        delta_a = vertebras_corners[v + 3] - vertebras_corners[v + 2]

        pivot_b = np.array([side_x_fst[0] + borders[0, 0], y_l_new[0]])
        delta_b = np.array([side_x_sec[0] + borders[0, 0], y_l_new[-1]]) - pivot_b
        if pivot_a[0] == (pivot_b + delta_b)[0] and (delta_a[0] == 0 or pivot_a[1] == (pivot_b + delta_b)[1]):
            vertebras_corners[v + 2] = pivot_b + delta_b
        else:
            t = (pivot_b[0] + (delta_b[0] / (delta_b[1] + 1e-6)) * (pivot_a[1] - pivot_b[1]) - pivot_a[0]) / (delta_a[0] * (1 - (delta_b[0] * delta_a[1]) / (delta_b[1] * delta_a[0] + 1e-6)) + 1e-6)
            vertebras_corners[v + 2] = pivot_a + t * delta_a

        pivot_b = np.array([side_x_fst[1] + borders[0, 0], y_r_new[0]])
        delta_b = np.array([side_x_sec[1] + borders[0, 0], y_r_new[-1]]) - pivot_b
        if pivot_a[0] == (pivot_b + delta_b)[0] and (delta_a[0] == 0 or pivot_a[1] == (pivot_b + delta_b)[1]):
            vertebras_corners[v + 3] = pivot_b + delta_b
        else:
            t = (pivot_b[0] + (delta_b[0] / (delta_b[1] + 1e-6)) * (pivot_a[1] - pivot_b[1]) - pivot_a[0]) / (delta_a[0] * (1 - (delta_b[0] * delta_a[1]) / (delta_b[1] * delta_a[0] + 1e-6)) + 1e-6)
            vertebras_corners[v + 3] = pivot_a + t * delta_a
    
    if len(vertebras_corners) >= 4 and strip:
        vertebras_corners[0] = np.array([borders[0, 0], (quantile_matrixes[0][0] + quantile_matrixes[2][0]) / 2], dtype=np.float32)
        vertebras_corners[1] = np.array([borders[0, 0], (quantile_matrixes[0][0] + quantile_matrixes[2][0]) / 2], dtype=np.float32)
        vertebras_corners[-1] = np.array([borders[1, 0], (quantile_matrixes[0][-1] + quantile_matrixes[2][-1]) / 2], dtype=np.float32)
        vertebras_corners[-2] = np.array([borders[1, 0], (quantile_matrixes[0][-1] + quantile_matrixes[2][-1]) / 2], dtype=np.float32)

    return vertebras_corners

def get_vertebras_corners(data: np.ndarray, borders: np.ndarray, y: tuple[np.ndarray, np.ndarray, np.ndarray], threshold: float) -> np.ndarray:
    """
    Based on data image (0 <= pixel <= 1) and (middle regression, right-edge, left-edge) computes corners coords [row, col] of each vertebra.\n
    if line_mean >= threshold then it is vertebra, else - gap. 
    """
    vertebras = [] # to store final result

    gamma = 1e-6 # used as gradient of a straight horizontal line
    y_m, y_l, y_r = [i.astype(dtype=np.float32) for i in y]
    
    dy_m = np.concatenate([y_m[1:], y_m[-1:]]) - y_m; dy_m = np.where(dy_m != 0, dy_m, np.full_like(dy_m, gamma))
    grad = np.divide(-np.ones_like(y_m, dtype=np.float32), dy_m) # tan of normal

    r_list = get_rad_list(y)
    
    prev_points = [np.array([borders[0, 0], borders[0, 1]], dtype=np.float32), np.array([borders[0, 0], borders[1, 1]], dtype=np.float32)]
    prev_state = False # True - previous state was VERTEBRAE, False - GAP
    for x in range(grad.shape[0]):
        delta = np.array([1.0, grad[x]], dtype=np.float32); delta = delta / np.max(np.abs(delta))
        left_coords, right_coords, pivot = np.array([x, y_m[x]], dtype=np.float32), np.array([x, y_m[x]], dtype=np.float32), np.array([x, y_m[x]], dtype=np.float32)

        reach_edge = [False, False] # indicator of reaching the edge [left, right] by line
        summary = np.zeros(shape=2, dtype=np.float32) # saving [sum, steps]

        r = r_list[x]
        while not all(reach_edge):
            if not reach_edge[0]:
                if (0 <= left_coords[0] < y_m.shape[0]) and (0 < left_coords[1]) and np.sqrt(np.sum((pivot - left_coords) ** 2)) < r * 3:
                    summary += np.array([data[borders[0, 0] + int(left_coords[0]), int(left_coords[1])], 1], dtype=np.float32)
                    left_coords += (-1 if delta[1] > 0 else 1) * delta
                else:
                    reach_edge[0] = True

            if not reach_edge[1]:
                if (0 <= right_coords[0] < y_m.shape[0]) and (right_coords[1] < data.shape[1]) and np.sqrt(np.sum((pivot - right_coords) ** 2)) < r * 3:
                    summary += np.array([data[borders[0, 0] + int(right_coords[0]), int(right_coords[1])], 1], dtype=np.float32)
                    right_coords += (1 if delta[1] > 0 else -1) * delta
                else:
                    reach_edge[1] = True

        left_coords[0] += borders[0, 0]; right_coords[0] += borders[0, 0] # attaching to a pivot of a data
        
        # Edging the data
        if left_coords[0] > borders[1, 0]:
            left_coords[0] = borders[1, 0]
        if left_coords[0] < borders[0, 0]:
            left_coords[0] = borders[0, 0]
        
        if right_coords[0] > borders[1, 0]:
            right_coords[0] = borders[1, 0]
        if right_coords[0] < borders[0, 0]:
            right_coords[0] = borders[0, 0]

        if summary[1] > 0:
            if summary[0] / summary[1] >= threshold:
                if not prev_state:
                    vertebras += [left_coords, right_coords]
                    prev_state = True
            else:
                if prev_state:
                    vertebras += prev_points
                    prev_state = False
        else:
            if prev_state:
                vertebras += prev_points
                prev_state = False
        
        prev_points = [left_coords, right_coords]

    if len(vertebras) % 4 != 0:
        vertebras = [*vertebras, np.array([borders[1, 0], borders[1, 1]], dtype=np.float32), np.array([borders[1, 0], borders[0, 1]], dtype=np.float32)]

    if len(vertebras) >= 4:
        vertebras[0] = np.array([borders[0, 0], borders[0, 1]], dtype=np.float32)
        vertebras[1] = np.array([borders[0, 0], borders[1, 1]], dtype=np.float32)
        vertebras[-1] = np.array([borders[1, 0], borders[0, 1]], dtype=np.float32)
        vertebras[-2] = np.array([borders[1, 0], borders[1, 1]], dtype=np.float32)

    return np.array(vertebras, dtype=np.float32)

def corners_metric(pixel_array, vertebras_corners):
    if vertebras_corners.shape[0] > 96:
        return float("inf")

    pixel_array_cpy = np.where(pixel_array > 0.5, 1, 0)

    inter = np.zeros_like(pixel_array)
    for c in range(0, vertebras_corners.shape[0], 4):
        cv2.fillConvexPoly(inter, np.array([*vertebras_corners[c: c + 2][::, ::-1].astype(np.int32), *vertebras_corners[c + 2: c + 4][::-1, ::-1].astype(np.int32)]), 1)

    IOU_white = np.sum(np.where(np.multiply(inter, pixel_array_cpy) > 0.5, 1, 0)) / np.sum(np.where(inter + pixel_array_cpy > 0.5, 1, 0))
    
    return 1 - IOU_white

def get_corners_metric(borders, pixel_array, y, t):
    vertebras_corners = get_vertebras_corners(pixel_array, borders, y, t)
    return corners_metric(pixel_array, vertebras_corners)

def get_corners_threshold_brute(borders, pixel_array, threshold_e, y):
    e_log = int(np.log10(threshold_e)) * -1
    init = 0
    for t in range(1, e_log + 1):
        power = 10 ** -t
        gen = range(1, 10) if power == 0.1 else range(-9, 10)
        metrics = []
        with Pool() as p:
            metrics = p.starmap(get_corners_metric, [(borders, pixel_array, y, init + t * power) for t in gen])
        min_metric = [metrics[0], 0]
        for m in range(1, len(metrics)):
            if min_metric[0] >= metrics[m]: min_metric = [metrics[m], m]

        init += gen[min_metric[1]] * power

    return init

q_coef_left_side = get_quantile_part_brute(borders_side, pixel_array_side, regression_params, matrixes_side[:2], A_side, "left")
q_coef_right_side = get_quantile_part_brute(borders_side, pixel_array_side, regression_params, matrixes_side[2:], A_side, "right")
quantile_matrixes_left_side = get_lr_quantilepart(matrixes_side[:2], regression_params["quantile_iterations"], q_coef_left_side, A_side)
quantile_matrixes_right_side = get_lr_quantilepart(matrixes_side[2:], regression_params["quantile_iterations"], q_coef_right_side, A_side)

vertebras_threshold_side = get_corners_threshold_brute(borders_side, pixel_array_side, threshold_e, ((quantile_matrixes_left_side[0] + quantile_matrixes_right_side[0]) / 2, quantile_matrixes_left_side[0], quantile_matrixes_right_side[0]))
vertebras_corners_side = get_vertebras_corners(pixel_array_side, borders_side, ((quantile_matrixes_left_side[0] + quantile_matrixes_right_side[0]) / 2, quantile_matrixes_left_side[0], quantile_matrixes_right_side[0]), vertebras_threshold_side)

print(q_coef_left_side, q_coef_right_side, vertebras_threshold_side)

vertebras_corners_side = adjust_corners(matrixes_side, (*quantile_matrixes_left_side, *quantile_matrixes_right_side), np.copy(vertebras_corners_side), borders_side, pixel_array_side, regression_params, 0, vertebras_corners_side.shape[0])

# open right points
img_points = cv2.imread(os.path.join(DATA_DIR, f"niito/{f_name}.png"))
img_points = np.array(cv2.inRange(img_points, (0, 0, 210), (40, 40, 256)))
control_points = np.array([[i, j] for i in range(img_points.shape[0]) for j in range(img_points.shape[1]) if img_points[i, j] > 0.5], dtype=np.float32)

vertebras_list = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'Th1', 'Th2', 'Th3', 'Th4', 'Th5', 'Th6', 'Th7', 'Th8', 'Th9', 'Th10', 'Th11', 'Th12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1']
response = {i: [] for i in vertebras_list}
for i in range(96):
    point_1 = np.multiply(vertebras_corners_side[i], pixel_spacing)
    min_dist = float("inf")
    for p in control_points:
        new_p = np.multiply(p, pixel_spacing_real)
        dist = np.sqrt(
            np.sum(
                (point_1 - new_p) ** 2
            )
        )
        if dist < min_dist:
            min_dist = dist
    response[vertebras_list[int(i / 4)]].append(min_dist)

dt = pd.DataFrame(response)
dt.to_csv("response.csv")

for vertebras_corners in [[vertebras_corners_side, pixel_array_side, quantile_matrixes_left_side[0], quantile_matrixes_right_side[0], borders_side, (quantile_matrixes_left_side[0] + quantile_matrixes_right_side[0]) / 2]]:
    print(vertebras_corners[0].shape)

    for y in range(vertebras_corners[5].shape[0]):
        cv2.circle(vertebras_corners[1], (int(vertebras_corners[2][y]), int(vertebras_corners[4][0, 0] + y)), 2, 10, thickness=2)
        cv2.circle(vertebras_corners[1], (int(vertebras_corners[3][y]), int(vertebras_corners[4][0, 0] + y)), 2, 10, thickness=2)
        cv2.circle(vertebras_corners[1], (int(vertebras_corners[5][y]), int(vertebras_corners[4][0, 0] + y)), 2, 10, thickness=2)

    for c in range(0, vertebras_corners[0].shape[0], 2):
        cv2.circle(vertebras_corners[1], (int(vertebras_corners[0][c, 1]), int(vertebras_corners[0][c, 0])), 5, 5, thickness=2)
        cv2.circle(vertebras_corners[1], (int(vertebras_corners[0][c + 1, 1]), int(vertebras_corners[0][c + 1, 0])), 5, 5, thickness=2)
        cv2.line(vertebras_corners[1], vertebras_corners[0][c, ::-1].astype(np.int32), vertebras_corners[0][c + 1, ::-1].astype(np.int32), 8, thickness=2)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(9)
    fig.set_figheight(24)

    sns.heatmap(vertebras_corners[1])
plt.show()