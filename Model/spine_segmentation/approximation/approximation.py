"""
    This file contains functions for edge-approximation of the spine
"""

import numpy as np
import numpy.typing as npt

from multiprocessing import Pool, Process, Manager
from multiprocessing.pool import ThreadPool

from spine_segmentation.approximation.typings import *
from spine_segmentation.vertebraes.typings import VERTEBRAES_R_POINTS_PRJ
from typing import Literal

from .metrics import *

def __first_arg(l: list[float]):
    return l[0]

def get_lr_matrixes(pixel_array: npt.NDArray[np.float32], borders: npt.NDArray[np.int32]) -> LR_MATRIXES:
    """
        Retruns `LR_MATRIXES` of distances from edges of image to the first-occured spine-pixels and their weights

        Parameters:
        -----------
            pixel_array:
                2d array of pixels - spine image
            \n
            borders:
                Bounding rectangle of the spine - `[up-left, down-right]` coords
    """
    y_len = borders[1, 0] + 1 - borders[0, 0] # height of the spine in pixels

    y_l, y_r = np.zeros(shape=y_len, dtype=np.float32), np.zeros(shape=y_len, dtype=np.float32) # left projection and right projection on col's axis
    exc_l, exc_r = np.zeros(shape=y_len, dtype=np.float32), np.zeros(shape=y_len, dtype=np.float32) # indicator whether gap (0) or vertebra (1)

    for i in range(y_len):
        # left hand
        for j in range(borders[0, 1], borders[1, 1] + 1):
            if pixel_array[i + borders[0, 0], j]: 
                y_l[i] = j
                exc_l[i] = 1
                break
        # right hand
        for j in range(borders[1, 1], borders[0, 1] - 1, -1):
            if pixel_array[i + borders[0, 0], j]: 
                y_r[i] = j
                exc_r[i] = 1
                break

    W_l, W_r = np.diag(exc_l), np.diag(exc_r) # making weight-matrix from indicator
    
    return LR_MATRIXES(
            left_edge_array=y_l,
            left_edge_weights=W_l,
            right_edge_array=y_r,
            right_edge_weights=W_r
        )

def get_Vandermonde_matrix(start: float, stop: float, shape: tuple[int, int], f: REGRESSION_FUNC_PROTOCOL) -> npt.NDArray[np.float32]:
    """
    Computes Vandermonde's matrix of shape = `shape`

    Parameters:
    -----------
        start:
            Start point of interval
        \n
        stop:
            End point of interval
        \n
        shape:
            Shape of return matrix
        \n
        f(x, n, t):
            Regression function to compute term
    """

    A = np.linspace(start, stop, int(shape[0]), dtype=np.float32)
    return np.array([[f(x, shape[1], t) for t in range(shape[1] + 1)] for x in A], dtype=np.float32)

def get_lr_Vandermonde_matrix(start: float, stop: float, lr_matrixes: LR_MATRIXES, f: REGRESSION_FUNC_PROTOCOL, n: int) -> VANDERMONDE_MATRIXES:
    """
    Computes Vandermonde's matrix of shape = `shape` (left and right)

    Parameters:
    -----------
        start:
            Start point of interval
        \n
        stop:
            End point of interval
        \n
        lr_matrixes:
            Matrixes obtained from `get_lr_matrixes` of `LR_MATRIXES` type
        \n
        f(x, t, n):
            regression function to compute term
        \n
        n:
            Number of cols in matrix (number of terms in polynom)
    """
    
    with Pool(2) as p:
        A = p.starmap(get_Vandermonde_matrix, [(start, stop, (lr_matrixes.left_edge_array.shape[0], n), f), (start, stop, (lr_matrixes.right_edge_array.shape[0], n), f)])

    return VANDERMONDE_MATRIXES(
        left=A[0],
        right=A[1]
    )

#   -------------------------------------------------------------------
#   Spine-edge approximation
#   -------------------------------------------------------------------

def __compute_q_coef(y_init: npt.NDArray[np.float32], W_init: npt.NDArray[np.float32], A: npt.NDArray[np.float32], regression_params: Q_REGRESSION_PARAMS, quantile: float) -> npt.NDArray[np.float32]:
    """
        Computes quantile regression coefficients for polynom (one edge)

        Parameters:
            y_init:
                Array of `distances` [left-to-right or right-to-left] to the first occurance of spine-pixel
            \n
            W_init:
                Diagonal 2d-array where diagonal elements are weights for `y_init`
            \n
            A:
                Vandermonde matrix
            \n
            regression_params:
                Regression parameters of `Q_REGRESSION_PARAMS` type
            \n
            quantile:
                Quantile, value in [0, 1)
    """

    divide_ind = int(y_init.shape[0] * (1 - quantile)) if quantile < 1 else y_init.shape[0] - 1

    for _ in range(regression_params.q_iter):
        c = np.dot(np.linalg.pinv(np.dot(np.dot(A.T, W_init), A)), np.dot(np.dot(A.T, W_init), y_init.T))
        y_new = np.dot(A, c.T)

        se = (y_new - y_init) ** 2
        se_q = np.sort(se)[divide_ind]

        W_tmp = np.where(se < se_q, 1 - quantile, quantile)
        W_init = np.multiply(np.diag(W_tmp), W_init)
    
    return np.dot(np.linalg.pinv(np.dot(np.dot(A.T, W_init), A)), np.dot(np.dot(A.T, W_init), y_init.T))

def __get_q_metric_and_c(index: int, borders: npt.NDArray[np.int32], pixel_array: npt.NDArray[np.float32], regression_params: Q_REGRESSION_PARAMS, y_init: npt.NDArray[np.float32], W_init: npt.NDArray[np.float32], A: npt.NDArray[np.float32], quantile: float, mode: Literal["left", "right"]) -> tuple[float, npt.NDArray[np.float32], int]:
    """
        Returns List[metric, quantile coefficients, index]

        Parameters:
        -----------
            index:
                Index in search list
            \n
            borders:
                Bounding rectangle of the spine - `[up-left, down-right]` coords
            \n
            pixel_array:
                2d array of pixels - spine image
            \n
            regression_params:
                Regression parameters of `Q_REGRESSION_PARAMS` type
            \n
            y_init:
                Array of `distances` [left-to-right or right-to-left] to the first occurance of spine-pixel
            \n
            W_init:
                Diagonal 2d-array where diagonal elements are weights for `y_init`
            \n
            A:
                Vandermonde matrix
            \n
            quantile:
                Quantile, value in [0, 1)
            \n
            mode:
                Which side will be computed `(left | right)`
    """

    c = __compute_q_coef(y_init, W_init, A, regression_params, quantile)
    y_new = np.dot(A, c.T)

    return (lr_q_metric(pixel_array, borders, y_new, mode), c, index)

def __search_q_coef(borders: npt.NDArray[np.int32], pixel_array: npt.NDArray[np.float32], regression_params: Q_REGRESSION_PARAMS, y_init: npt.NDArray[np.float32], W_init: npt.NDArray[np.float32], A: npt.NDArray[np.float32], mode: Literal["left", "right"], return_coefs: dict) -> None:
    """
        Searches for quantile regression coefficients

        Parameters:
        -----------
            borders:
                Bounding rectangle of the spine - `[up-left, down-right]` coords
            \n
            pixel_array:
                2d array of pixels - spine image
            \n
            regression_params:
                Regression parameters of `Q_REGRESSION_PARAMS` type
            \n
            y_init:
                Array of `distances` [left-to-right or right-to-left] to the first occurance of spine-pixel
            \n
            W_init:
                Diagonal 2d-array where diagonal elements are weights for `y_init`
            \n
            A:
                Vandermonde matrix
            \n
            mode:
                Which side will be computed `(left | right)`
            \n
            return_coefs:
                Dictionary to write in computed coefs
    """
    e_log = int(np.log10(regression_params.q_part_e)) * -1
    init = 0
    with ThreadPool() as p:
        for t in range(1, e_log + 1):
            power = 10 ** -t
            gen = range(1, 10) if t == 1 else range(-9, 10)
            metrics = p.starmap(__get_q_metric_and_c, [(t, borders, pixel_array, regression_params, y_init, W_init, A, init + t * power, mode) for t in gen])

            min_metric = min(metrics, key=__first_arg)
            init += min_metric[2] * power

    return_coefs[mode] = min_metric[1]

def get_lr_q_coefs(borders: npt.NDArray[np.int32], pixel_array: npt.NDArray[np.float32], lr_matrixes: LR_MATRIXES, regression_params: Q_REGRESSION_PARAMS, Vandermonde_matrixes: VANDERMONDE_MATRIXES) -> Q_COEFS:
    """
        Computes quantile regression coefficients for polynom (right and left edges)

        Parameters:
            borders:
                Bounding rectangle of the spine - `[up-left, down-right]` coords
            \n
            pixel_array:
                2d array of pixels - spine image
            \n
            lr_matrixes:
                Matrixes obtained from `get_lr_matrixes` of `LR_MATRIXES` type
            \n
            regression_params:
                Regression parameters of `Q_REGRESSION_PARAMS` type
            \n
            Vandermonde_matrixes:
                Vandermonde_matrixes of `VANDERMONDE_MATRIXES` type
    """

    manager = Manager()
    c_dict = manager.dict()

    left_process = Process(target=__search_q_coef, args=(borders, pixel_array, regression_params, lr_matrixes.left_edge_array, lr_matrixes.left_edge_weights, Vandermonde_matrixes.left, "left", c_dict))
    right_process = Process(target=__search_q_coef, args=(borders, pixel_array, regression_params, lr_matrixes.right_edge_array, lr_matrixes.right_edge_weights, Vandermonde_matrixes.right, "right", c_dict))

    left_process.start()
    right_process.start()

    left_process.join()
    right_process.join()

    return Q_COEFS(c_left=c_dict["left"], c_right=c_dict["right"])