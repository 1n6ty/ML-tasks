import numpy as np

def _get_signed_angle(v1: np.ndarray[np.float32], v2: np.ndarray[np.float32]) -> np.float32:
    """Computes arctan2 function of two vectors `v1` and `v2`.

        Args
        ----
            v1 (np.ndarray[np.float32])
                First vector
            v2 (np.ndarray[np.float32])
                Second vector
    """
    return np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], np.dot(v1, v2))

def _compute_errq_jac(t: np.ndarray[np.float32], tau: np.float32, l: np.ndarray[np.float32]):
    err = (l[1:] - np.array([t[0] + t[1] * i for i in l[:-1]]))
    return [
        -1 * np.sum(
            np.where(err >= 0, tau * 1, (tau - 1))
        ),
        -1 * np.sum(
            np.where(err >= 0, [tau * i for i in l[:-1]], [(tau - 1) * i for i in l[:-1]])
        )
    ]

def _compute_errq(t: np.ndarray[np.float32], tau: np.float32, l: np.ndarray[np.float32]):
    err = (l[1:] - np.array([t[0] + t[1] * i for i in l[:-1]]))
    return np.sum(
        np.where(err >= 0, tau * err, (tau - 1) * err)
    )

def _path_func(t, dx, dy):
    return np.sqrt(dx(t) ** 2 + dy(t) ** 2)
