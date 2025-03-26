"""
    This file contains typings used in `Medical_Parameters` class
"""

import numpy as np

_R_POINT = np.ndarray[np.ndarray[np.float32, 3], 8]
"""
    Basic type of reference points for one vertebrae

    0: 1-st point
    1: 2-nd point
    ...
    7: 8-th point

    Related to Gladcov's work
"""

_R_POINT_PRJ = np.ndarray[np.ndarray[np.float32, 2], 4]
"""
    Basic type of reference points' projection for one vertebrae 

    0: bottom-left point
    1: top-left point
    2: top-right point
    3: bottom-right point
"""

VERTEBRAES_R_POINTS = np.ndarray[_R_POINT, 24]
"""
    Vertebraes' reference points fromat
"""

VERTEBRAES_R_POINTS_PRJ = np.ndarray[_R_POINT_PRJ, 24]
"""
    Vertebraes' reference points' projection fromat
"""

SF_R_POINT_PRJ = np.ndarray[VERTEBRAES_R_POINTS_PRJ, 2]
"""
    Side `[0]` and frontal `[1]` projections
"""