import cv2
import numpy as np
import os
from func import *

DATA_DIR = os.path.abspath('../Data/spine-segmentation')

def regression_func(x, n, t):
    return (x ** t) * ((1 - x) ** (n - t))

def kernel_func(x):
    return (15/16) * (1 - x ** 2) ** 2 if abs(x) <= 1 else 0.0

regression_params = {
    "n": 10,
    "regression_func": regression_func,
    "quantile_part_e": 0.001,
    "quantile_iterations": 16
}

threshold_e = 0.001

def __open_png(file_path_side, file_path_frontal, new_image_size):
    img_side = cv2.imread(file_path_side)
    img_frontal = cv2.imread(file_path_frontal)

    if img_side.shape[0] < img_frontal.shape[0]:
        d = int((img_frontal.shape[0] - img_side.shape[0]) / 2)
        img_side = cv2.copyMakeBorder(img_side, d, d, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    else:
        d = int((img_side.shape[0] - img_frontal.shape[0]) / 2)
        img_frontal = cv2.copyMakeBorder(img_frontal, d, d, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))

    img_side_hip = np.copy(img_side)
    img_frontal_hip = np.copy(img_frontal)

    img_side = cv2.inRange(img_side, (0, 0, 210), (40, 40, 256))
    img_frontal = cv2.inRange(img_frontal, (0, 0, 210), (40, 40, 256))

    img_side_hip = cv2.inRange(img_side_hip, (210, 0, 0), (256, 40, 40))
    img_frontal_hip = cv2.inRange(img_frontal_hip, (210, 0, 0), (256, 40, 40))
    
    img_side = cv2.resize(img_side, new_image_size[::-1], interpolation=cv2.INTER_CUBIC)
    img_frontal = cv2.resize(img_frontal, new_image_size[::-1], interpolation=cv2.INTER_CUBIC)

    img_side_hip = cv2.resize(img_side_hip, new_image_size[::-1], interpolation=cv2.INTER_CUBIC)
    img_frontal_hip = cv2.resize(img_frontal_hip, new_image_size[::-1], interpolation=cv2.INTER_CUBIC)

    return (
        np.array(img_side, dtype="float32") / np.max(img_side), 
        np.array(img_frontal, dtype="float32") / np.max(img_frontal),
        np.array(img_side_hip, dtype="float32") / np.max(img_side_hip), 
        np.array(img_frontal_hip, dtype="float32") / np.max(img_frontal_hip)
    )

pixel_array_side, pixel_array_frontal, pixel_array_side_hip, pixel_array_frontal_hip = __open_png(os.path.join(DATA_DIR, "side.png"), os.path.join(DATA_DIR, "frontal.png"), (1760, 768))
pixel_spacing = np.array([0.5, 0.5])

def height_corr_frontal(pixel_array_frontal, pixel_array_side_hip, pixel_array_frontal_hip):
    height_corr = int(np.mean([i for i in range(pixel_array_side_hip.shape[0]) for j in range(pixel_array_side_hip.shape[1]) if pixel_array_side_hip[i, j] >= 0.5]) - np.mean([i for i in range(pixel_array_frontal_hip.shape[0]) for j in range(pixel_array_frontal_hip.shape[1]) if pixel_array_frontal_hip[i, j] >= 0.5]))
    
    if height_corr > 0:
        pixel_array_frontal = cv2.copyMakeBorder(pixel_array_frontal, 0, height_corr, 0, 0, cv2.BORDER_CONSTANT, 0)
        return pixel_array_frontal[height_corr: , :]
    else:
        pixel_array_frontal = cv2.copyMakeBorder(pixel_array_frontal, -height_corr, 0, 0, 0, cv2.BORDER_CONSTANT, 0)
        return pixel_array_frontal[:height_corr: , :]

pixel_array_frontal = height_corr_frontal(pixel_array_frontal, pixel_array_side_hip, pixel_array_frontal_hip)

borders_side = get_borders(pixel_array_side)
A_side = get_Vandermond_matrix(0, 1, borders_side[1, 0] + 1 - borders_side[0, 0], regression_params["n"], regression_params["regression_func"])

matrixes_side = get_lr_matrixes(pixel_array_side, borders_side)

borders_frontal = get_borders(pixel_array_frontal)
A_frontal = get_Vandermond_matrix(0, 1, borders_frontal[1, 0] + 1 - borders_frontal[0, 0], regression_params["n"], regression_params["regression_func"])

matrixes_frontal = get_lr_matrixes(pixel_array_frontal, borders_frontal)

import matplotlib.pyplot as plt
import seaborn as sns

from multiprocessing import Pool

def compute_regression_metric(borders, pixel_array, y, mode):
    inter = np.copy(pixel_array)
    for i in range(y.shape[0]):
        for j in (range(max(int(y[i]), borders[0, 1]), borders[1, 1] + 1) if mode == "left" else range(borders[0, 1], min(int(y[i]) + 1, borders[1, 1] + 1))):
            if inter[i + borders[0, 0], j] > 0.5:
                inter[i + borders[0, 0], j] = 2
            else:
                inter[i + borders[0, 0], j] = -2

    IOU_white = np.sum(np.where(inter > 1.5, 1, 0)) / np.sum(pixel_array)
    IOU_black = 1 - np.sum(np.where(inter < -1.5, 1, 0)) / (y.shape[0] * (borders[1, 1] - borders[0, 1] + 1)) 
                
    return 2 - (IOU_white + IOU_black)

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
        init += gen[np.argmin(metrics)] * power

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

def compute_regression_vertebra_metric(vertebra, side_x_fst, side_x_sec, borders, pixel_array, y, mode):
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

        pivot_b = np.array([side_x_fst[0] + borders[0, 0], y[0]])
        delta_b = np.array([side_x_sec[0] + borders[0, 0], y[-1]]) - pivot_b
        if pivot_a[0] == pivot_b[0] and (delta_a[0] == 0 or pivot_a[1] == pivot_b[1]):
            vertebra_c[0] = pivot_b
        else:
            t = (pivot_b[0] + (delta_b[0] / (delta_b[1] + 1e-6)) * (pivot_a[1] - pivot_b[1]) - pivot_a[0]) / (delta_a[0] * (1 - (delta_b[0] * delta_a[1]) / (delta_b[1] * delta_a[0] + 1e-6)) + 1e-6)
            vertebra_c[0] = pivot_a + t * delta_a

        pivot_a = np.copy(vertebra[2])
        delta_a = vertebra[3] - vertebra[2]

        pivot_b = np.array([side_x_fst[0] + borders[0, 0], y[0]])
        delta_b = np.array([side_x_sec[0] + borders[0, 0], y[-1]]) - pivot_b
        if pivot_a[0] == (pivot_b + delta_b)[0] and (delta_a[0] == 0 or pivot_a[1] == (pivot_b + delta_b)[1]):
            vertebra_c[2] = pivot_b + delta_b
        else:
            t = (pivot_b[0] + (delta_b[0] / (delta_b[1] + 1e-6)) * (pivot_a[1] - pivot_b[1]) - pivot_a[0]) / (delta_a[0] * (1 - (delta_b[0] * delta_a[1]) / (delta_b[1] * delta_a[0] + 1e-6)) + 1e-6)
            vertebra_c[2] = pivot_a + t * delta_a
    else:
        pivot_a = np.copy(vertebra[0])
        delta_a = vertebra[1] - vertebra[0]

        pivot_b = np.array([side_x_fst[1] + borders[0, 0], y[0]])
        delta_b = np.array([side_x_sec[1] + borders[0, 0], y[-1]]) - pivot_b
        if pivot_a[0] == pivot_b[0] and (delta_a[0] == 0 or pivot_a[1] == pivot_b[1]):
            vertebra_c[1] = pivot_b
        else:
            t = (pivot_b[0] + (delta_b[0] / (delta_b[1] + 1e-6)) * (pivot_a[1] - pivot_b[1]) - pivot_a[0]) / (delta_a[0] * (1 - (delta_b[0] * delta_a[1]) / (delta_b[1] * delta_a[0] + 1e-6)) + 1e-6)
            vertebra_c[1] = pivot_a + t * delta_a

        pivot_a = np.copy(vertebra[2])
        delta_a = vertebra[3] - vertebra[2]

        pivot_b = np.array([side_x_fst[1] + borders[0, 0], y[0]])
        delta_b = np.array([side_x_sec[1] + borders[0, 0], y[-1]]) - pivot_b
        if pivot_a[0] == (pivot_b + delta_b)[0] and (delta_a[0] == 0 or pivot_a[1] == (pivot_b + delta_b)[1]):
            vertebra_c[3] = pivot_b + delta_b
        else:
            t = (pivot_b[0] + (delta_b[0] / (delta_b[1] + 1e-6)) * (pivot_a[1] - pivot_b[1]) - pivot_a[0]) / (delta_a[0] * (1 - (delta_b[0] * delta_a[1]) / (delta_b[1] * delta_a[0] + 1e-6)) + 1e-6)
            vertebra_c[3] = pivot_a + t * delta_a
    
    cv2.fillConvexPoly(inter, np.array([*vertebra_c[0: 2][::, ::-1].astype(np.int32), *vertebra_c[2: 4][::-1, ::-1].astype(np.int32)]), 1)

    IOU_white = np.sum(np.where(inter + pixel_array > 1.5, 1, 0)) / np.sum(np.where(inter_v + pixel_array > 1.5, 1, 0))
    IOU_black = (np.sum(inter) - np.sum(np.where(inter + pixel_array > 1.5, 1, 0))) / (np.sum(inter_v) - np.sum(np.where(inter_v + pixel_array > 1.5, 1, 0)))

    return 1 - IOU_white + IOU_black

def get_quantile_metric_vertebra(vertebra, side_x_fst, side_x_sec, borders, pixel_array, regression_params, matrixes, A, t, mode):
    y_new = get_lr_quantilepart(matrixes, regression_params["quantile_iterations"], t, A)[0]

    return compute_regression_vertebra_metric(vertebra, side_x_fst, side_x_sec, borders, pixel_array, y_new, mode)

def get_quantile_vertebra_side_part_brute(vertebra, side_x_fst, side_x_sec, borders, pixel_array, regression_params, matrixes, A, mode):
    e_log = int(np.log10(regression_params["quantile_part_e"])) * -1
    init = 0
    for t in range(1, e_log + 1):
        power = 10 ** -t
        gen = range(1, 10) if power == 0.1 else range(-9, 10)
        metrics = []
        with Pool() as p:
            metrics = p.starmap(get_quantile_metric_vertebra, [(vertebra, side_x_fst, side_x_sec, borders, pixel_array, regression_params, matrixes, A, init + t * power, mode) for t in gen])
        init += gen[np.argmin(metrics)] * power

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
        
        buff_l = matrixes[0][side_x_fst[0]: side_x_sec[0] + 1]
        buff_r = matrixes[2][side_x_fst[1]: side_x_sec[1] + 1]

        W_l = matrixes[1][side_x_fst[0]: side_x_sec[0] + 1, side_x_fst[0]: side_x_sec[0] + 1]
        W_r = matrixes[3][side_x_fst[1]: side_x_sec[1] + 1, side_x_fst[1]: side_x_sec[1] + 1]

        A_left = get_Vandermond_matrix(0, 1, buff_l.shape[0], (2 if v == vertebras_corners.shape[0] - 4 else 1), lambda x, n, t: x ** t)
        A_right = get_Vandermond_matrix(0, 1, buff_r.shape[0], (2 if v == vertebras_corners.shape[0] - 4 else 1), lambda x, n, t: x ** t)

        l_coef = get_quantile_vertebra_side_part_brute(vertebras_corners[v: v + 4], side_x_fst, side_x_sec, borders, pixel_array, regression_params, (buff_l, W_l), A_left, "left")
        r_coef = get_quantile_vertebra_side_part_brute(vertebras_corners[v: v + 4], side_x_fst, side_x_sec, borders, pixel_array, regression_params, (buff_r, W_r), A_right, "right")

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

def get_vertebras_corners(data: np.ndarray, borders: np.ndarray, y: tuple[np.ndarray, np.ndarray, np.ndarray], threshold: float, strip=True) -> np.ndarray:
    """
    Based on data image (0 <= pixel <= 1) and (middle regression, right-edge, left-edge) computes corners coords [row, col] of each vertebra.\n
    if line_mean >= threshold then it is vertebra, else - gap. 
    """
    vertebras = [] # to store final result

    gamma = 1e-6 # used as gradient of a straight horizontal line
    y_m, y_l, y_r = [i.astype(dtype=np.float32) for i in y]
    
    dy_m = np.concatenate([y_m[1:], y_m[-1:]]) - y_m; dy_m = np.where(dy_m != 0, dy_m, np.full_like(dy_m, gamma))
    grad = np.divide(-np.ones_like(y_m, dtype=np.float32), dy_m) # tan of normal

    prev_state = False # True - previous state was VERTEBRA, False - GAP
    for x in range(grad.shape[0]):
        delta = np.array([1.0, grad[x]], dtype=np.float32); delta = delta / np.max(np.abs(delta))
        left_coords, right_coords, pivot = np.array([x, y_m[x]], dtype=np.float32), np.array([x, y_m[x]], dtype=np.float32), np.array([x, y_m[x]], dtype=np.float32)

        reach_edge = [False, False] # indicator of reaching the edge [left, right] by line
        summary = np.zeros(shape=2, dtype=np.float32) # saving [sum, steps]

        r = get_radius(y, pivot, delta)
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
                    vertebras += [left_coords, right_coords]
                    prev_state = False
        else:
            if prev_state:
                vertebras += [left_coords, right_coords]
                prev_state = False

    if len(vertebras) % 4 != 0:
        vertebras = [*vertebras, np.array([borders[1, 0], borders[1, 1]], dtype=np.float32), np.array([borders[1, 0], borders[0, 1]], dtype=np.float32)]

    if len(vertebras) >= 4 and strip:
        vertebras[0] = np.array([borders[0, 0], borders[0, 1]], dtype=np.float32)
        vertebras[1] = np.array([borders[0, 0], borders[1, 1]], dtype=np.float32)
        vertebras[-1] = np.array([borders[1, 0], borders[0, 1]], dtype=np.float32)
        vertebras[-2] = np.array([borders[1, 0], borders[1, 1]], dtype=np.float32)

    return np.array(vertebras, dtype=np.float32)

def corners_metric(pixel_array, vertebras_corners):
    if vertebras_corners.shape[0] > 96:
        return float("inf")
    inter = np.zeros_like(pixel_array)
    for c in range(0, vertebras_corners.shape[0], 4):
        cv2.fillConvexPoly(inter, np.array([*vertebras_corners[c: c + 2][::, ::-1].astype(np.int32), *vertebras_corners[c + 2: c + 4][::-1, ::-1].astype(np.int32)]), 1)

    IOU_white = np.sum(np.where(inter + pixel_array > 1.5, 1, 0)) / np.sum(np.where(inter + pixel_array > 0.5, 1, 0))
    IOU_black = np.sum(np.where(inter + pixel_array < 1.5, 1, 0)) / (pixel_array.shape[0] * pixel_array.shape[1] - np.sum(pixel_array))
                
    return 1 - IOU_white + IOU_black

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
        init += gen[np.argmin(metrics)] * power

    return init

q_coef_left = get_quantile_part_brute(borders_side, pixel_array_side, regression_params, matrixes_side[:2], A_side, "left")
q_coef_right = get_quantile_part_brute(borders_side, pixel_array_side, regression_params, matrixes_side[2:], A_side, "right")
quantile_matrixes_left = get_lr_quantilepart(matrixes_side[:2], regression_params["quantile_iterations"], q_coef_left, A_side)
quantile_matrixes_right = get_lr_quantilepart(matrixes_side[2:], regression_params["quantile_iterations"], q_coef_right, A_side)

vertebras_threshold_side = get_corners_threshold_brute(borders_side, pixel_array_side, threshold_e, ((quantile_matrixes_left[0] + quantile_matrixes_right[0]) / 2, quantile_matrixes_left[0], quantile_matrixes_right[0]))
vertebras_corners_side = get_vertebras_corners(pixel_array_side, borders_side, ((quantile_matrixes_left[0] + quantile_matrixes_right[0]) / 2, quantile_matrixes_left[0], quantile_matrixes_right[0]), vertebras_threshold_side)
print(q_coef_left, q_coef_right, vertebras_threshold_side)

q_coef_left_front = get_quantile_part_brute(borders_frontal, pixel_array_frontal, regression_params, matrixes_frontal[:2], A_frontal, "left")
q_coef_right_front = get_quantile_part_brute(borders_frontal, pixel_array_frontal, regression_params, matrixes_frontal[2:], A_frontal, "right")
quantile_matrixes_left_front = get_lr_quantilepart(matrixes_frontal[:2], regression_params["quantile_iterations"], q_coef_left_front, A_frontal)
quantile_matrixes_right_front = get_lr_quantilepart(matrixes_frontal[2:], regression_params["quantile_iterations"], q_coef_right_front, A_frontal)

vertebras_threshold_front = get_corners_threshold_brute(borders_frontal, pixel_array_frontal, threshold_e, ((quantile_matrixes_left_front[0] + quantile_matrixes_right_front[0]) / 2, quantile_matrixes_left_front[0], quantile_matrixes_right_front[0]))
vertebras_corners_front = get_vertebras_corners(pixel_array_frontal, borders_frontal, ((quantile_matrixes_left_front[0] + quantile_matrixes_right_front[0]) / 2, quantile_matrixes_left_front[0], quantile_matrixes_right_front[0]), vertebras_threshold_front)
print(q_coef_left_front, q_coef_right_front, vertebras_threshold_front)

vertebras_corners_side = adjust_corners(matrixes_side, (*quantile_matrixes_left, *quantile_matrixes_right), np.copy(vertebras_corners_side), borders_side, pixel_array_side, regression_params, 0, vertebras_corners_side.shape[0])

vertebras_corners_front = adjust_corners(matrixes_frontal, (*quantile_matrixes_left_front, *quantile_matrixes_right_front), np.copy(vertebras_corners_front), borders_frontal, pixel_array_frontal, regression_params, 0, 4)
vertebras_corners_front = adjust_corners(matrixes_frontal, (*quantile_matrixes_left_front, *quantile_matrixes_right_front), np.copy(vertebras_corners_front), borders_frontal, pixel_array_frontal, regression_params, vertebras_corners_front.shape[0] - 4, vertebras_corners_front.shape[0])

y_m = (quantile_matrixes_left_front[0] + quantile_matrixes_right_front[0]) / 2
y_l, y_r = quantile_matrixes_left_front[0], quantile_matrixes_right_front[0]
y_l_default, y_r_default = matrixes_frontal[0], matrixes_frontal[2]

# filling regression to borders_side

top_vec = vertebras_corners_front[3] - vertebras_corners_front[2]
top_vec = np.array([-top_vec[1], top_vec[0]])
top_vec /= abs(top_vec[0])

pivot = np.array([int(vertebras_corners_front[2][0]), y_l[int(vertebras_corners_front[2][0] - borders_frontal[0, 0])]])
add = []
while pivot[0] > borders_side[0, 0]:
    pivot += top_vec
    add.append(pivot[1])
y_l_default = np.concatenate([add[::-1], y_l_default[int(vertebras_corners_front[2][0] - borders_frontal[0, 0]):]])
W_default = matrixes_frontal[1][int(vertebras_corners_front[2][0] - borders_frontal[0, 0]):, int(vertebras_corners_front[2][0] - borders_frontal[0, 0]):]
W_default = np.multiply(np.pad(W_default, [(len(add), 0), (len(add), 0)], constant_values=[(1, 1), (1, 1)]), np.diag(np.ones_like(y_l_default)))
matrixes_frontal = (y_l_default, W_default, matrixes_frontal[2], matrixes_frontal[3])
y_l = np.concatenate([add[::-1], y_l[int(vertebras_corners_front[2][0] - borders_frontal[0, 0]):]])
W_default = quantile_matrixes_left_front[1][int(vertebras_corners_front[2][0] - borders_frontal[0, 0]):, int(vertebras_corners_front[2][0] - borders_frontal[0, 0]):]
W_default = np.multiply(np.pad(W_default, [(len(add), 0), (len(add), 0)], constant_values=[(1, 1), (1, 1)]), np.diag(np.ones_like(y_l)))
quantile_matrixes_left_front = (y_l, W_default)

pivot = np.array([int(vertebras_corners_front[3][0]), y_r[int(vertebras_corners_front[3][0] - borders_frontal[0, 0])]])
add = []
while pivot[0] > borders_side[0, 0]:
    pivot += top_vec
    add.append(pivot[1])
y_r_default = np.concatenate([add[::-1], y_r_default[int(vertebras_corners_front[3][0] - borders_frontal[0, 0]):]])
W_default = matrixes_frontal[3][int(vertebras_corners_front[3][0] - borders_frontal[0, 0]):, int(vertebras_corners_front[3][0] - borders_frontal[0, 0]):]
W_default = np.multiply(np.pad(W_default, [(len(add), 0), (len(add), 0)], constant_values=[(1, 1), (1, 1)]), np.diag(np.ones_like(y_r_default)))
matrixes_frontal = (matrixes_frontal[0], matrixes_frontal[1], y_r_default, W_default)
y_r = np.concatenate([add[::-1], y_r[int(vertebras_corners_front[3][0] - borders_frontal[0, 0]):]])
W_default = quantile_matrixes_right_front[1][int(vertebras_corners_front[3][0] - borders_frontal[0, 0]):, int(vertebras_corners_front[3][0] - borders_frontal[0, 0]):]
W_default = np.multiply(np.pad(W_default, [(len(add), 0), (len(add), 0)], constant_values=[(1, 1), (1, 1)]), np.diag(np.ones_like(y_r)))
quantile_matrixes_right_front = (y_r, W_default)

borders_frontal[0, 0] = borders_side[0, 0]

bot_vec = vertebras_corners_front[-3] - vertebras_corners_front[-4]
bot_vec = np.array([-bot_vec[1], bot_vec[0]])
bot_vec /= abs(bot_vec[0])

pivot = np.array([int(vertebras_corners_front[-3][0]), y_r[int(vertebras_corners_front[-3][0] - borders_frontal[0, 0])]])
add = []
while pivot[0] < borders_side[1, 0]:
    pivot -= bot_vec
    add.append(pivot[1])
y_r_default = np.concatenate([y_r_default[:int(vertebras_corners_front[-3][0] - borders_frontal[0, 0]) + 1], add])
W_default = matrixes_frontal[3][:int(vertebras_corners_front[-3][0] - borders_frontal[0, 0]) + 1, :int(vertebras_corners_front[-3][0] - borders_frontal[0, 0]) + 1]
W_default = np.multiply(np.pad(W_default, [(0, len(add)), (0, len(add))], constant_values=[(1, 1), (1, 1)]), np.diag(np.ones_like(y_r_default)))
matrixes_frontal = (matrixes_frontal[0], matrixes_frontal[1], y_r_default, W_default)
y_r = np.concatenate([y_r[:int(vertebras_corners_front[-3][0] - borders_frontal[0, 0]) + 1], add])
W_default = quantile_matrixes_right_front[1][:int(vertebras_corners_front[-3][0] - borders_frontal[0, 0]) + 1, :int(vertebras_corners_front[-3][0] - borders_frontal[0, 0]) + 1]
W_default = np.multiply(np.pad(W_default, [(0, len(add)), (0, len(add))], constant_values=[(1, 1), (1, 1)]), np.diag(np.ones_like(y_r)))
quantile_matrixes_right_front = (y_r, W_default)

pivot = np.array([int(vertebras_corners_front[-4][0]), y_l[int(vertebras_corners_front[-4][0] - borders_frontal[0, 0])]])
add = []
while pivot[0] < borders_side[1, 0]:
    pivot -= bot_vec
    add.append(pivot[1])
y_l_default = np.concatenate([y_l_default[:int(vertebras_corners_front[-4][0] - borders_frontal[0, 0]) + 1], add])
W_default = matrixes_frontal[1][:int(vertebras_corners_front[-4][0] - borders_frontal[0, 0]) + 1, :int(vertebras_corners_front[-4][0] - borders_frontal[0, 0]) + 1]
W_default = np.multiply(np.pad(W_default, [(0, len(add)), (0, len(add))], constant_values=[(1, 1), (1, 1)]), np.diag(np.ones_like(y_l_default)))
matrixes_frontal = (y_l_default, W_default, matrixes_frontal[2], matrixes_frontal[3])
y_l = np.concatenate([y_l[:int(vertebras_corners_front[-4][0] - borders_frontal[0, 0]) + 1], add])
W_default = quantile_matrixes_left_front[1][:int(vertebras_corners_front[-4][0] - borders_frontal[0, 0]) + 1, :int(vertebras_corners_front[-4][0] - borders_frontal[0, 0]) + 1]
W_default = np.multiply(np.pad(W_default, [(0, len(add)), (0, len(add))], constant_values=[(1, 1), (1, 1)]), np.diag(np.ones_like(y_l)))
quantile_matrixes_left_front = (y_l, W_default)

borders_frontal[1, 0] = borders_side[1, 0]

y_m = (y_l + y_r) / 2

vertebras_corners_front = get_vertebras_corners(pixel_array_frontal, borders_frontal, ((quantile_matrixes_left_front[0] + quantile_matrixes_right_front[0]) / 2, quantile_matrixes_left_front[0], quantile_matrixes_right_front[0]), vertebras_threshold_front, strip=False)
vertebras_corners_front = adjust_corners(matrixes_frontal, (*quantile_matrixes_left_front, *quantile_matrixes_right_front), np.copy(vertebras_corners_front), borders_frontal, pixel_array_frontal, regression_params, 0, vertebras_corners_front.shape[0], strip=False)

def adjast_vertebras_side(borders_side, vertebras_corners_side, y_m, gap_c):
    height_coef = [0.3857, 1.0925, 0.9491, 1.0714, 1.0666, 1.0625, 1.0441, 1.0845, 1.0259, 1.0379, 1.0243, 1.0, 1.0238, 1.0465, 1.0222, 1.0652, 1.0408, 1.0392, 1.0, 1.0188, 1.037, 0.9464, 3] # vertebra[i + 1].height / vertebra[i].height
    up_plates_coef = [1.0, 1.0493, 1.0, 1.0277, 1.0, 1.0, 1.0526, 1.0476, 1.0, 1.0, 1.0175, 1.0161, 1.0303, 1.0142, 1.014, 1.0, 1.0, 1.0135, 1.0263, 1.0, 1.0, 1.0, 1.0] # vertebra[i + 1].up_plate / vertebra[i].down_plate
    down_plates_coef = [1.0312, 1.0801, 1.0285, 1.0277, 1.0, 1.027, 1.1052, 1.1903, 1.08, 1.0555, 1.0876, 1.0644, 1.0605, 1.0142, 1.014, 1.0277, 1.0, 1.0269, 1.0263, 1.0256, 1.0, 1.0, 0.0] # vertebra[i + 1].down_plate / vertebra[i].down_plate

    gap_heights = []
    point = 4
    l = vertebras_corners_side.shape[0]
    while point < l:
        current_gap = np.sqrt(
            np.sum((((vertebras_corners_side[point - 1] + vertebras_corners_side[point - 2]) - (vertebras_corners_side[point] + vertebras_corners_side[point + 1])) ** 2) / 4)
        )

        gap_heights.append(current_gap)
        gap_heights.sort()

        m_gap = gap_heights[int(len(gap_heights) / 2)]

        incl_v = [] # array of elements like [height, plate_up_radius, plate_down_radius]
    
        while l + len(incl_v) * 4 <= 96:
            if len(incl_v) == 0:
                vertebra_down_plate_radius = np.sqrt(np.sum(((vertebras_corners_side[point - 1] - vertebras_corners_side[point - 2]) ** 2) / 4))
                
                next_vertebra_height = np.sqrt(
                    np.sum(
                        (((vertebras_corners_side[point - 1] + vertebras_corners_side[point - 2]) - (vertebras_corners_side[point - 3] + vertebras_corners_side[point - 4])) ** 2) / 4
                    )
                ) * height_coef[int(point / 4) - 1]
            else:
                vertebra_down_plate_radius = incl_v[-1][2]
                next_vertebra_height = incl_v[-1][0] * height_coef[int(point / 4) - 1 + len(incl_v)]

            if next_vertebra_height + np.sum([i[0] for i in incl_v]) + (len(incl_v) + 1) * m_gap * gap_c < current_gap:
                incl_v.append(
                    [
                        next_vertebra_height,
                        vertebra_down_plate_radius * up_plates_coef[int(point / 4) - 1 + len(incl_v)],
                        vertebra_down_plate_radius * down_plates_coef[int(point / 4) - 1 + len(incl_v)]
                    ]
                )
            else:
                break
        
        sum_l = np.sum([i[0] for i in incl_v])
        incl_gap = (current_gap - sum_l) / (len(incl_v) + 1)
        for incl in incl_v:
            pivot_x = ((vertebras_corners_side[point - 1] + vertebras_corners_side[point - 2]) / 2)[0]
            pivot_y = y_m[int(pivot_x) - borders_side[0, 0]]
            pivot = np.array([pivot_x, pivot_y], dtype=np.float32)
            end_pivot = np.copy(pivot)
            while np.sqrt(np.sum((end_pivot - pivot) ** 2)) < incl_gap:
                end_pivot[0] += 1.0
                end_pivot[1] = y_m[int(end_pivot[0]) - borders_side[0, 0]]
            
            grad_d = y_m[int(end_pivot[0]) + 1 - borders_side[0, 0]] - end_pivot[1]
            grad = 1e-6 if grad_d == 0 else grad_d
            grad = -1 / grad

            delta = np.array([1.0, grad], dtype=np.float32)
            t = np.sqrt((incl[1] ** 2) / np.sum(delta ** 2))

            v_points = np.array([end_pivot + t * delta, end_pivot - t * delta])
            v_points = v_points[v_points[:, 1].argsort()]
            vertebras_corners_side = np.concatenate([vertebras_corners_side[:point], v_points, vertebras_corners_side[point:]])
            point += 2

            pivot = np.copy(end_pivot)
            while np.sqrt(np.sum((end_pivot - pivot) ** 2)) < incl[0]:
                end_pivot[0] += 1.0
                end_pivot[1] = y_m[int(end_pivot[0]) - borders_side[0, 0]]

            grad_d = y_m[int(end_pivot[0]) + 1 - borders_side[0, 0]] - end_pivot[1]
            grad = 1e-6 if grad_d == 0 else grad_d
            grad = -1 / grad

            delta = np.array([1.0, grad], dtype=np.float32)
            t = np.sqrt((incl[2] ** 2) / np.sum(delta ** 2))

            v_points = np.array([end_pivot + t * delta, end_pivot - t * delta])
            v_points = v_points[v_points[:, 1].argsort()]
            vertebras_corners_side = np.concatenate([vertebras_corners_side[:point], v_points, vertebras_corners_side[point:]])
            point += 2

        if len(incl_v) == 0:
            point += 4

    return vertebras_corners_side

def add_to_96_side(borders_side, vertebras_corners_side, y_m):
    add = 1
    init = 0
    while True:
        vertebras_corners = np.copy(vertebras_corners_side)
        vertebras_corners = adjast_vertebras_side(borders_side, vertebras_corners, y_m, init)
        if vertebras_corners.shape[0] == 96:
            return vertebras_corners
        elif vertebras_corners.shape[0] < 96:
            init -= add
            add *= 0.1
        
        if init == 0 and vertebras_corners.shape[0] < 96:
            print("Couldn't add to 96")
            return vertebras_corners
        init += add

vertebras_corners_side = add_to_96_side(borders_side, vertebras_corners_side, (quantile_matrixes_left[0] + quantile_matrixes_right[0]) / 2)

def link_side2front(vertebras_corners_side, vertebras_corners_front, borders_frontal, y_m_frontal):
    vertebras_corners_front_new = np.zeros_like(vertebras_corners_side)
    for v_f in range(0, vertebras_corners_front.shape[0], 4):
        center_f = np.sum(vertebras_corners_front[v_f: v_f + 4], axis=0) / 4
        for v_s in range(0, vertebras_corners_side.shape[0], 4):
            if vertebras_corners_side[v_s + 1][0] <= center_f[0] <= vertebras_corners_side[v_s + 3][0]:
                top_r = np.sqrt(np.sum((vertebras_corners_front[v_f] - vertebras_corners_front[v_f + 1]) ** 2)) / 2
                bottom_r = np.sqrt(np.sum((vertebras_corners_front[v_f + 2] - vertebras_corners_front[v_f + 3]) ** 2)) / 2

                pivot = np.array([vertebras_corners_side[v_s + 1][0], y_m_frontal[int(vertebras_corners_side[v_s + 1][0]) - borders_frontal[0, 0]]])

                grad_d = y_m_frontal[int(vertebras_corners_side[v_s + 1][0]) + 1 - borders_frontal[0, 0]] - y_m_frontal[int(vertebras_corners_side[v_s + 1][0]) - borders_frontal[0, 0]]
                grad = 1e-6 if grad_d == 0 else grad_d
                grad = -1 / grad

                delta = np.array([1.0, grad], dtype=np.float32)
                t = np.sqrt((top_r ** 2) / np.sum(delta ** 2))

                v_points = np.array([pivot + t * delta, pivot - t * delta])
                v_points = v_points[v_points[:, 1].argsort()]

                vertebras_corners_front_new[v_s] = v_points[0]
                vertebras_corners_front_new[v_s + 1] = v_points[1]

                pivot = np.array([vertebras_corners_side[v_s + 3][0], y_m_frontal[int(vertebras_corners_side[v_s + 3][0]) - borders_frontal[0, 0]]])

                grad_d = y_m_frontal[int(vertebras_corners_side[v_s + 3][0]) + 1 - borders_frontal[0, 0]] - y_m_frontal[int(vertebras_corners_side[v_s + 3][0]) - borders_frontal[0, 0]]
                grad = 1e-6 if grad_d == 0 else grad_d
                grad = -1 / grad

                delta = np.array([1.0, grad], dtype=np.float32)
                t = np.sqrt((bottom_r ** 2) / np.sum(delta ** 2))

                v_points = np.array([pivot + t * delta, pivot - t * delta])
                v_points = v_points[v_points[:, 1].argsort()]

                vertebras_corners_front_new[v_s + 2] = v_points[0]
                vertebras_corners_front_new[v_s + 3] = v_points[1]

                break

    return vertebras_corners_front_new

vertebras_corners_front = link_side2front(vertebras_corners_side, vertebras_corners_front, borders_frontal, y_m)

def adjast_front(vertebras_corners_side, vertebras_corners_front, borders_frontal, y_m_frontal):
    plates_front = [
        [0, 18], [20, 21], [22, 23], [24, 25], [26, 28], [28, 30], [30, 30], [30, 30], [30, 30], [30, 31], [32, 33], [34, 35], [36, 37], [38, 39], [40, 41], [42, 43], [44, 45], [46, 46], [46, 47], [47, 48], [48, 49], [49, 50], [50, 51], [52, 0]
    ] # [up_plate_d, bottom_plate_d]

    for v_f in range(0, vertebras_corners_front.shape[0], 4):
        if vertebras_corners_front[v_f][0] != 0:
            for v_f_1 in range(0, v_f, 4):
                top_r = (plates_front[int(v_f_1 / 4)][0] / plates_front[int(v_f / 4)][0]) * np.sqrt(np.sum((vertebras_corners_front[v_f] - vertebras_corners_front[v_f + 1]) ** 2)) / 2
                bottom_r = (plates_front[int(v_f_1 / 4)][1] / plates_front[int(v_f / 4)][0]) * np.sqrt(np.sum((vertebras_corners_front[v_f] - vertebras_corners_front[v_f + 1]) ** 2)) / 2
                
                pivot = np.array([vertebras_corners_side[v_f_1 + 1][0], y_m_frontal[int(vertebras_corners_side[v_f_1 + 1][0]) - borders_frontal[0, 0]]])
                
                grad_d = y_m_frontal[int(vertebras_corners_side[v_f_1 + 1][0]) - borders_frontal[0, 0] + 1] - y_m_frontal[int(vertebras_corners_side[v_f_1 + 1][0]) - borders_frontal[0, 0]]
                grad = 1e-6 if grad_d == 0 else grad_d
                grad = -1 / grad

                delta = np.array([1.0, grad], dtype=np.float32)
                t = np.sqrt((top_r ** 2) / np.sum(delta ** 2))

                v_points = np.array([pivot + t * delta, pivot - t * delta])
                v_points = v_points[v_points[:, 1].argsort()]

                vertebras_corners_front[v_f_1] = v_points[0]
                vertebras_corners_front[v_f_1 + 1] = v_points[1]

                pivot = np.array([vertebras_corners_side[v_f_1 + 3][0], y_m_frontal[int(vertebras_corners_side[v_f_1 + 3][0]) - borders_frontal[0, 0]]])

                grad_d = y_m_frontal[int(vertebras_corners_side[v_f_1 + 3][0]) - borders_frontal[0, 0] + 1] - y_m_frontal[int(vertebras_corners_side[v_f_1 + 3][0]) - borders_frontal[0, 0]]
                grad = 1e-6 if grad_d == 0 else grad_d
                grad = -1 / grad

                delta = np.array([1.0, grad], dtype=np.float32)
                t = np.sqrt((bottom_r ** 2) / np.sum(delta ** 2))

                v_points = np.array([pivot + t * delta, pivot - t * delta])
                v_points = v_points[v_points[:, 1].argsort()]

                vertebras_corners_front[v_f_1 + 2] = v_points[0]
                vertebras_corners_front[v_f_1 + 3] = v_points[1]
            break

    for v_f in range(0, vertebras_corners_front.shape[0], 4):
        if vertebras_corners_front[v_f][0] == 0:
            top_r = (plates_front[int(v_f / 4)][0] / plates_front[int(v_f / 4) - 1][0]) * np.sqrt(np.sum((vertebras_corners_front[v_f - 4] - vertebras_corners_front[v_f - 3]) ** 2)) / 2
            bottom_r = (plates_front[int(v_f / 4)][1] / plates_front[int(v_f / 4) - 1][0]) * np.sqrt(np.sum((vertebras_corners_front[v_f - 4] - vertebras_corners_front[v_f - 3]) ** 2)) / 2
            
            pivot = np.array([vertebras_corners_side[v_f + 1][0], y_m_frontal[int(vertebras_corners_side[v_f + 1][0]) - borders_frontal[0, 0]]])
            
            grad_d = y_m_frontal[int(vertebras_corners_side[v_f + 1][0]) - borders_frontal[0, 0]] - y_m_frontal[int(vertebras_corners_side[v_f + 1][0]) - borders_frontal[0, 0] - 1]
            grad = 1e-6 if grad_d == 0 else grad_d
            grad = -1 / grad

            delta = np.array([1.0, grad], dtype=np.float32)
            t = np.sqrt((top_r ** 2) / np.sum(delta ** 2))

            v_points = np.array([pivot + t * delta, pivot - t * delta])
            v_points = v_points[v_points[:, 1].argsort()]

            vertebras_corners_front[v_f] = v_points[0]
            vertebras_corners_front[v_f + 1] = v_points[1]

            pivot = np.array([vertebras_corners_side[v_f + 3][0], y_m_frontal[int(vertebras_corners_side[v_f + 3][0]) - borders_frontal[0, 0]]])

            grad_d = y_m_frontal[int(vertebras_corners_side[v_f + 3][0]) - borders_frontal[0, 0]] - y_m_frontal[int(vertebras_corners_side[v_f + 3][0]) - borders_frontal[0, 0] - 1]
            grad = 1e-6 if grad_d == 0 else grad_d
            grad = -1 / grad

            delta = np.array([1.0, grad], dtype=np.float32)
            t = np.sqrt((bottom_r ** 2) / np.sum(delta ** 2))

            v_points = np.array([pivot + t * delta, pivot - t * delta])
            v_points = v_points[v_points[:, 1].argsort()]

            vertebras_corners_front[v_f + 2] = v_points[0]
            vertebras_corners_front[v_f + 3] = v_points[1]

    return vertebras_corners_front

vertebras_corners_front = adjast_front(vertebras_corners_side, vertebras_corners_front, borders_frontal, y_m)

for vertebras_corners in [[vertebras_corners_side, pixel_array_side, quantile_matrixes_left[0], quantile_matrixes_right[0], borders_side, (quantile_matrixes_left[0] + quantile_matrixes_right[0]) / 2], [vertebras_corners_front, pixel_array_frontal, y_l, y_r, borders_frontal, y_m]]:
    print(vertebras_corners[0].shape)

    for c in range(0, vertebras_corners[0].shape[0], 2):
        cv2.circle(vertebras_corners[1], (int(vertebras_corners[0][c, 1]), int(vertebras_corners[0][c, 0])), 5, 5, thickness=2)
        cv2.circle(vertebras_corners[1], (int(vertebras_corners[0][c + 1, 1]), int(vertebras_corners[0][c + 1, 0])), 5, 5, thickness=2)
        cv2.line(vertebras_corners[1], vertebras_corners[0][c, ::-1].astype(np.int32), vertebras_corners[0][c + 1, ::-1].astype(np.int32), 8, thickness=2)

    for y in range(vertebras_corners[5].shape[0]):
        # cv2.circle(vertebras_corners[1], (int(vertebras_corners[2][y]), int(vertebras_corners[4][0, 0] + y)), 2, 10, thickness=2)
        # cv2.circle(vertebras_corners[1], (int(vertebras_corners[3][y]), int(vertebras_corners[4][0, 0] + y)), 2, 10, thickness=2)
        cv2.circle(vertebras_corners[1], (int(vertebras_corners[5][y]), int(vertebras_corners[4][0, 0] + y)), 2, 10, thickness=2)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(9)
    fig.set_figheight(24)

    sns.heatmap(vertebras_corners[1])
plt.show()