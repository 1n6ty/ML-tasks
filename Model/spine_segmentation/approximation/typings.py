"""
    This file contains typings for approximation functions
"""

import numpy as np
import numpy.typing as npt

from typing import NamedTuple, Protocol

class LR_MATRIXES(NamedTuple):
    """
        Lr_matrixes format

        Variables:
        ----------
            left_edge_array:
                Array of `distances`[left-to-right] to the first occurance of spine-pixel
            \n
            left_edge_weights:
                Diagonal 2d-array where diagonal elements are weights for `left_edge_array`
            \n
            right_edge_array:
                Array of `distances`[right-to-left] to the first occurance of spine-pixel
            \n
            right_edge_weights:
                Diagonal 2d-array where diagonal elements are weights for `right_edge_array`        
    """

    left_edge_array: npt.NDArray[np.float32]
    left_edge_weights: npt.NDArray[np.float32]

    right_edge_array: npt.NDArray[np.float32]
    right_edge_weights: npt.NDArray[np.float32]

class VANDERMONDE_MATRIXES(NamedTuple):
    """
        Vandermonde matrixes format

        Variables:
        ----------
            left:
                Vandermonde matrix for left edge
            \n
            right:
                Vandermonde matrix for right edge
    """

    left: npt.NDArray[np.float32]
    right: npt.NDArray[np.float32]

class REGRESSION_FUNC_PROTOCOL(Protocol):
    """
        Protocol for regression term-function

        Parameters:
        ----------
            x:
                Variable from interval
            \n
            t:
                Index of term
            \n
            n:
                Number of terms
        
        Returns:
        --------
            Computed term of x
        
    """
    def __call__(self, x: np.float32, t: int, n: int) -> np.float32: ...

class Q_REGRESSION_PARAMS(NamedTuple):
    """
        Regression parameters format

        Variables:
        ----------
            n:
                Number of terms in polynom
            \n
            regression_func:
                Function used to compute terms in polinom `f(x, t, n)`
            \n
            q_part_e:
                Accuracy of the quantile\n
                `-log10(q_part_e)` - number of digits of quantile after point
            \n
            q_iter:
                Iterations in quantile regression algorithm
    """

    n: int
    regression_func: REGRESSION_FUNC_PROTOCOL
    q_part_e: float
    q_iter: int

class Q_COEFS(NamedTuple):
    """
        Coefficients for polinom dirived from quantile regression algorithm

        Variables:
        ----------
            c_left:
                Array of coefficients for left edge
            \n
            c_right:
                Array of coefficients for right edge
    """

    c_left: npt.NDArray[np.float32]
    c_right: npt.NDArray[np.float32]