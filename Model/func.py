import numpy as np
import tensorflow as tf
from itertools import product

from typing import Callable

def get_PE_matrix(rows: int, cols: int, n=100000) -> tf.Tensor:
    """
    Generates Positional Encoding Matrix
    """
    common_col = tf.range(0, rows, 1, dtype=tf.float32)
    return tf.constant(
            tf.transpose(
                tf.concat([tf.expand_dims(tf.math.cos(common_col / (n ** ((d - 1) / cols))), axis=0) if d % 2 else tf.expand_dims(tf.math.sin(common_col / (n ** (d / cols))), axis=0) for d in range(cols)], axis=0),
                perm=[1, 0]
            )
        )

def get_borders(data: np.ndarray) -> np.ndarray: 
    """
    Searching for borders of spine
    """
    IMG_SHAPE = data.shape
    borders = np.zeros((2, 2), dtype=np.int32) # [[up-left] and [down-right] coords]

    # search for row1
    break_flag = False
    for i in range(IMG_SHAPE[0]):
        for j in range(IMG_SHAPE[1]):
            if data[i, j]:
                borders[0, 0] = i
                break_flag = True
                break
        if break_flag: break

    # search for row2
    break_flag = False
    for i in range(IMG_SHAPE[0] - 1, -1, -1):
        for j in range(IMG_SHAPE[1]):
            if data[i, j]:
                borders[1, 0] = i
                break_flag = True
                break
        if break_flag: break

    # search for col1
    break_flag = False
    for i in range(IMG_SHAPE[1]):
        for j in range(IMG_SHAPE[0]):
            if data[j, i]:
                borders[0, 1] = i
                break_flag = True
                break
        if break_flag: break

    # search for col2
    break_flag = False
    for i in range(IMG_SHAPE[1] - 1, 0, -1):
        for j in range(IMG_SHAPE[0]):
            if data[j, i]:
                borders[1, 1] = i
                break_flag = True
                break
        if break_flag: break
    
    return borders

def get_lr_matrixes(data: np.ndarray, borders: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    Get left-edge and right-edge projections on col's axis\n
    t - resolution coefficient
    """
    y_len = borders[1, 0] + 1 - borders[0, 0] # height of the spine in pixels

    y_l, y_r = np.zeros(shape=y_len, dtype=np.float32), np.zeros(shape=y_len, dtype=np.float32) # left projection and right projection on col's axis
    exc_l, exc_r = np.zeros(shape=y_len, dtype=np.float32), np.zeros(shape=y_len, dtype=np.float32) # indicator whether gap (0) or vertebra (1)

    for i in range(y_len):
        # left hand
        for j in range(borders[0, 1], borders[1, 1] + 1):
            if data[i + borders[0, 0], j]: 
                y_l[i] = j
                exc_l[i] = 1
                break
        # right hand
        for j in range(borders[1, 1], borders[0, 1] - 1, -1):
            if data[i + borders[0, 0], j]: 
                y_r[i] = j
                exc_r[i] = 1
                break

    W_l, W_r = np.diag(exc_l), np.diag(exc_r) # making weight-matrix from indicator
    
    return (y_l, W_l, y_r, W_r)

def get_Vandermond_matrix(start: int, stop: int, num: int, n: int, f: Callable[[float, int, int], float]) -> np.ndarray:
    """
    Computes Vandermont's matrix of n cols and num rows.\n
    f(x, n, t) - function for each t in (0 <= t <= n) calls each x from linspace(start, stop, num)
    """
    A = np.linspace(start, stop, int(num), dtype=np.float32)
    return np.array([[f(x, n, t) for t in range(n + 1)] for x in A], dtype=np.float32)

def get_lr_quantilepart(lr_matrixes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], iter: int, q_part: float, A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes quantile regression through iter iterations over q_part SquareError.
    Where f(x, n) is used to compute Vandermont's matrix (n - last power in series)
    """
    y_l, W_l, y_r, W_r = lr_matrixes

    coef_l, coef_r = np.dot(np.linalg.inv(np.dot(np.dot(A.T, W_l), A)), np.dot(np.dot(A.T, W_l), y_l.T)), np.dot(np.linalg.inv(np.dot(np.dot(A.T, W_r), A)), np.dot(np.dot(A.T, W_r), y_r.T))
    y_l_q, y_r_q = np.dot(A, coef_l.T), np.dot(A, coef_r.T)

    for _ in range(iter):
        se_l, se_r = (y_l_q - y_l) ** 2, (y_r_q - y_r) ** 2
        part_l, part_r = (-np.sort(-se_l))[int((se_l.shape[0] - 1) * q_part)], (-np.sort(-se_r))[int((se_r.shape[0] - 1) * q_part)]

        W_l_tmp, W_r_tmp = np.diag(np.asarray(se_l < part_l, dtype=np.float32)), np.diag(np.asarray(se_r < part_r, dtype=np.float32))
        W_l, W_r = np.multiply(W_l_tmp, W_l), np.multiply(W_r_tmp, W_r)

        coef_l, coef_r = np.dot(np.linalg.inv(np.dot(np.dot(A.T, W_l), A)), np.dot(np.dot(A.T, W_l), y_l.T)), np.dot(np.linalg.inv(np.dot(np.dot(A.T, W_r), A)), np.dot(np.dot(A.T, W_r), y_r.T))
        y_l_q, y_r_q = np.dot(A, coef_l.T), np.dot(A, coef_r.T)
    
    return (y_l_q, W_l, y_r_q, W_r)

def get_lr_kernelpart(lr_matrixes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], quantile_matrixes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], K, h: float) -> tuple[np.ndarray, np.ndarray]:
    """
        Computes kernel regression with kernel K and step h
    """
    y_l, W_l, y_r, W_r = lr_matrixes
    y_l_q, W_l, y_r_q, W_r = quantile_matrixes

    if h == 0:
        return (y_l_q, y_r_q)

    y_l_k, y_r_k = np.dot(y_l, W_l), np.dot(y_r, W_r)
    y_l_k, y_r_k = np.where(y_l_k > 1e-6, y_l_k, y_l_q), np.where(y_r_k > 1e-6, y_r_k, y_r_q)

    A = np.linspace(0, 1, y_l.shape[0], dtype=np.float32); A = np.array([-A + A[i] for i in range(A.shape[0])]); A = np.array([[K(j) / h for j in i] for i in A], dtype=np.float32)
    A_l_w, A_r_w = np.dot(A, np.diag(y_l_k)), np.dot(A, np.diag(y_r_k))

    A = np.sum(A, axis=0)
    return (np.divide(np.sum(A_l_w, axis=0), A).T, np.divide(np.sum(A_r_w, axis=0), A).T)

def get_lr_approximation(data: np.ndarray, borders: np.ndarray, A: np.ndarray, K, n: int, iter: int, q_part: float, n_part: float, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes aproximation for left-edge and right-edge projections using quantile + kernel regression algorithm.
    n_part regulates amount of kernel regression part
    """
    matrixes = get_lr_matrixes(data, borders)
    
    quantile_matrixes = get_lr_quantilepart(matrixes, iter, q_part, A)
    kernel_matrixes = get_lr_kernelpart(matrixes, quantile_matrixes, K, h)

    y_l, W_l, y_r, W_r = matrixes
    y_l_q, W_l, y_r_q, W_r = quantile_matrixes
    y_l_k, y_r_k = kernel_matrixes

    y_l_r, y_r_r = y_l_q * (1 - n_part) + n_part * y_l_k, y_r_q * (1 - n_part) + n_part * y_r_k

    return ((y_l_r + y_r_r) / 2, y_l_r, y_r_r)

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

    prev_state = False # True - previous state was VERTEBRA, False - GAP
    x_step = (borders[1, 0] - borders[0, 0] + 1) / grad.shape[0]
    for x in range(grad.shape[0]):
        delta = np.array([x_step, grad[x]], dtype=np.float32); delta = delta / np.max(np.abs(delta))
        left_coords, right_coords = np.array([x * x_step, y_m[x]], dtype=np.float32), np.array([x * x_step, y_m[x]], dtype=np.float32)

        reach_edge = [False, False] # indicator of reaching the edge [left, right] by line
        summary = np.zeros(shape=2, dtype=np.float32) # saving [sum, steps]
        while not all(reach_edge):
            if not reach_edge[0]:
                if (0 <= left_coords[0] / x_step < y_m.shape[0]) and (borders[0, 1] <= left_coords[1]) and (y_l[int(left_coords[0] / x_step)] < left_coords[1]):
                    summary += np.array([data[borders[0, 0] + int(left_coords[0]), int(left_coords[1])], 1], dtype=np.float32)
                    left_coords += (-1 if delta[1] > 0 else 1) * delta
                else:
                    reach_edge[0] = True

            if not reach_edge[1]:
                if (0 <= right_coords[0] / x_step < y_m.shape[0]) and (right_coords[1] <= borders[1, 1]) and (right_coords[1] < y_r[int(right_coords[0] / x_step)]):
                    summary += np.array([data[borders[0, 0] + int(right_coords[0]), int(right_coords[1])], 1], dtype=np.float32)
                    right_coords += (1 if delta[1] > 0 else -1) * delta
                else:
                    reach_edge[1] = True
        
        left_coords[0] += borders[0, 0]; right_coords[0] += borders[0, 0] # attaching to a pivot of a data
        
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

    return np.array(vertebras, dtype=np.float32)

def pixels2niito(vertebras_corners_side: np.ndarray, vertebras_corners_frontal: np.ndarray, pixel_spacing: np.ndarray, vertebras_list: list[str]) -> dict["up": np.ndarray, "down": np.ndarray]:
    """
        Transfers pixel data into niito-pivot format dictionary [up-plate, down-plate]

        Parameters
        ----------

        vertebras_corners_side:
            Array of left-side projection points

        vertebras_corners_frontal:
            Array of frontal projection points

        pixel_spacing:
            Pixel spacing (horizontal, vertical)

        vertebras_list:
            List of vertebras names in output dictionary
    """
    if vertebras_corners_side.shape[0] != len(vertebras_list) * 4 or vertebras_corners_frontal.shape[0] != len(vertebras_list) * 4:
        raise Exception("In vertebras_corners should be exactly len(vertebras_list) * 4 points (try to pass different regression_settings)")

    vertebras_corners_side -= np.array([vertebras_corners_side[-4] for _ in range(vertebras_corners_side.shape[0])], dtype=np.float32)

    init_frontal = ((vertebras_corners_frontal[-4] + vertebras_corners_frontal[-3]) / 2)
    vertebras_corners_frontal -= np.array([init_frontal for _ in range(vertebras_corners_frontal.shape[0])], dtype=np.float32)

    tr = np.array([[-pixel_spacing[1], 0], [0, pixel_spacing[0]]], dtype=np.float32)
    vertebras_corners_side = np.dot(vertebras_corners_side, tr)[:, ::-1]
    vertebras_corners_frontal = np.dot(vertebras_corners_frontal, tr)[:, ::-1]

    response = {}
    for id, name in enumerate(vertebras_list):
        response[name] = np.array([
                [
                    vertebras_corners_side[id * 4 + 1, 0], 
                    (vertebras_corners_frontal[id * 4 + 1, 0] + vertebras_corners_frontal[id * 4, 0]) / 2,
                    (2 * vertebras_corners_side[id * 4 + 1, 1] + vertebras_corners_frontal[id * 4 + 1, 1] + vertebras_corners_frontal[id * 4, 1]) / 4
                ],
                [
                    (vertebras_corners_side[id * 4 + 1, 0] + vertebras_corners_side[id * 4, 0]) / 2,
                    vertebras_corners_frontal[id * 4 + 1, 0],
                    (vertebras_corners_side[id * 4 + 1, 1] + 2 * vertebras_corners_frontal[id * 4 + 1, 1] + vertebras_corners_side[id * 4, 1]) / 4
                ],
                [
                    vertebras_corners_side[id * 4, 0],
                    (vertebras_corners_frontal[id * 4, 0] + vertebras_corners_frontal[id * 4 + 1, 0]) / 2,
                    (2 * vertebras_corners_side[id * 4, 1] + vertebras_corners_frontal[id * 4, 1] + vertebras_corners_frontal[id * 4 + 1, 1]) / 4
                ],
                [
                    (vertebras_corners_side[id * 4 + 1, 0] + vertebras_corners_side[id * 4, 0]) / 2,
                    vertebras_corners_frontal[id * 4, 0],
                    (vertebras_corners_side[id * 4 + 1, 1] + 2 * vertebras_corners_frontal[id * 4, 1] + vertebras_corners_side[id * 4, 1]) / 4
                ],
                [
                    vertebras_corners_side[id * 4 + 3, 0], 
                    (vertebras_corners_frontal[id * 4 + 3, 0] + vertebras_corners_frontal[id * 4 + 2, 0]) / 2,
                    (2 * vertebras_corners_side[id * 4 + 3, 1] + vertebras_corners_frontal[id * 4 + 3, 1] + vertebras_corners_frontal[id * 4 + 2, 1]) / 4
                ],
                [
                    (vertebras_corners_side[id * 4 + 3, 0] + vertebras_corners_side[id * 4 + 2, 0]) / 2,
                    vertebras_corners_frontal[id * 4 + 3, 0],
                    (vertebras_corners_side[id * 4 + 3, 1] + 2 * vertebras_corners_frontal[id * 4 + 3, 1] + vertebras_corners_side[id * 4 + 2, 1]) / 4
                ],
                [
                    vertebras_corners_side[id * 4 + 2, 0],
                    (vertebras_corners_frontal[id * 4 + 2, 0] + vertebras_corners_frontal[id * 4 + 3, 0]) / 2,
                    (2 * vertebras_corners_side[id * 4 + 2, 1] + vertebras_corners_frontal[id * 4 + 2, 1] + vertebras_corners_frontal[id * 4 + 3, 1]) / 4
                ],
                [
                    (vertebras_corners_side[id * 4 + 3, 0] + vertebras_corners_side[id * 4 + 2, 0]) / 2,
                    vertebras_corners_frontal[id * 4 + 2, 0],
                    (vertebras_corners_side[id * 4 + 3, 1] + 2 * vertebras_corners_frontal[id * 4 + 2, 1] + vertebras_corners_side[id * 4 + 2, 1]) / 4
                ]
            ], dtype=np.float32)
    
    return response

def rotate(arr, angle_r):
    new_img = np.zeros_like(arr)

    origin_c = list(map(int, np.floor(np.array([arr.shape[1], arr.shape[0]]) / 2)))
    init_coords = list(product([c for c in range(arr.shape[1])], [r for r in range(arr.shape[0])]))
    
    origin = np.array([origin_c for c in init_coords])

    new_coords = (np.array([
                    [np.cos(angle_r), -np.sin(angle_r)], 
                    [np.sin(angle_r), np.cos(angle_r)]]) @ ((np.array(init_coords) - origin).T)) + (origin.T)
    new_coords = new_coords.astype(dtype=np.int32)
    
    for i in range(len(init_coords)):
        c, r = init_coords[i]
        if 0 <= new_coords[1][i] < arr.shape[0] and 0 <= new_coords[0][i] < arr.shape[1]:
            new_img[new_coords[1][i], new_coords[0][i]] = arr[r, c]
    
    return new_img