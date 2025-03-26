"""
    This file contains class of medical parameters for vertebraes based on vertebraes reference points
"""

import numpy as np
import pandas as pd

from spine_segmentation.vertebraes.typings import SF_R_POINT_PRJ

_V_LIST: list[str] = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'Th1', 'Th2', 'Th3', 'Th4', 'Th5', 'Th6', 'Th7', 'Th8', 'Th9', 'Th10', 'Th11', 'Th12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1']
_G_LIST: list[str] = [f"{_V_LIST[i]}-{_V_LIST[i + 1]}" for i in range(23)]

_COLUMNS_VERTEBRAES_SIDE: list[str] = [
    "Сагиттальный размер покровной замыкательной пластинки",
    "Сагиттальный размер базальной замыкательной пластинки",
    "Вертикальный размер тела позвонка по переднему контуру",
    "Вертикальный размер тела позвонка по заднему контуру",
    "Угол клиновидности тела позвонка",
    "Угол наклона переднего контура тела позвонка к оси Z",
    "Угол наклона верхней замыкательной пластинки тела позвонка к оси Z",
    "Угол наклона нижней замыкательной пластинки тела позвонка к оси Z",
    "Угол наклона замыкательной пластинки позвонка S1 к оси X"
]
_COLUMNS_VERTEBRAES_FRONTAL: list[str] = [

]

_COLUMNS_GAP_SIDE: list[str] = [
    "Угол между телами позвонков",
    "Высота диска спереди",
    "Высота диска сзади",
    "Угол клиновидности диска",
    "Линейное смещение верхнего позвонка относительно нижнего в плоскости диска",
    "Угловое смещение верхнего позвонка относительно нижнего в плоскости диска",
    "Угол между передним контуром позвонка L5 и замыкательной пластинкой S1"
]
_COLUMNS_GAP_FRONTAL: list[str] = [

]
# TODO for segments find best approximated with circle

class Medical_Parameters:
    """
        Medical parameters of vertebraes based on vertebraes reference points 

        Provides methods to alter and get parameters in pandas format # TODO
    """

    def __init__(self, points: SF_R_POINT_PRJ) -> None:
        """
            Computes medical parameters

            Parameters:
                points:
                    Vertebraes' reference points
        """

        self.points = points
        self._params_side: dict[str, pd.DataFrame] = {
            "vertebrae": pd.DataFrame(data=np.zeros((len(_V_LIST), len(_COLUMNS_VERTEBRAES_SIDE))), index=_V_LIST, columns=_COLUMNS_VERTEBRAES_SIDE, dtype=float),
            "gap": pd.DataFrame(data=np.zeros((len(_G_LIST), len(_COLUMNS_GAP_SIDE))), index=_G_LIST, columns=_COLUMNS_GAP_SIDE, dtype=float)
        }

    def _compute_vertebraes_parameters_side(self) -> None:
        """
            Computes vertebraes' parameters from side projection
        """

        for v_ind, v in enumerate(self.points[0]):
            self._params_side["vertebrae"].iloc[v_ind, 0] = np.sqrt((v[1][0] - v[2][0]) ** 2 + (v[1][1] - v[2][1]) ** 2)
            self._params_side["vertebrae"].iloc[v_ind, 1] = np.sqrt((v[0][0] - v[3][0]) ** 2 + (v[0][1] - v[3][1]) ** 2)
            self._params_side["vertebrae"].iloc[v_ind, 2] = np.sqrt((v[0][0] - v[1][0]) ** 2 + (v[0][1] - v[1][1]) ** 2)
            self._params_side["vertebrae"].iloc[v_ind, 3] = np.sqrt((v[2][0] - v[3][0]) ** 2 + (v[2][1] - v[3][1]) ** 2)
            self._params_side["vertebrae"].iloc[v_ind, 4] = np.arccos(
                ((v[1][0] - v[0][0]) * (v[2][0] - v[3][0]) + (v[1][1] - v[0][1]) * (v[2][1] - v[3][1])) / (self._params_side["vertebrae"].iloc[v_ind, 2] * self._params_side["vertebrae"].iloc[v_ind, 3])
            )
            self._params_side["vertebrae"].iloc[v_ind, 5] = np.arctan((v[0][0] - v[1][0]) / (v[1][1] - v[0][1]))
            self._params_side["vertebrae"].iloc[v_ind, 6] = np.arctan((v[1][0] - v[2][0]) / (v[1][1] - v[2][1]))
            self._params_side["vertebrae"].iloc[v_ind, 7] = np.arctan((v[0][0] - v[3][0]) / (v[0][1] - v[3][1]))
        
        self._params_side["vertebrae"].iloc[23, 8] = np.arcsin(
            (self.points[0][23][2][1] - self.points[0][23][1][1]) / np.sqrt((self.points[0][23][2][0] - self.points[0][23][1][0]) ** 2 + (self.points[0][23][2][1] - self.points[0][23][1][1]) ** 2)
        )

    def _compute_gap_parameters_side(self) -> None:
        """
            Computes gaps' parameters from side projection
        """

        for v_ind, [v1, v2] in enumerate(zip(self.points[0][:-1], self.points[0][1:])):
            self._params_side["gap"].iloc[v_ind, 0] = np.arccos(
                ((v1[1][0] - v1[0][0]) * (v2[1][0] - v2[0][0]) + (v1[1][1] - v1[0][1]) * (v2[1][1] - v2[0][1])) / np.sqrt(((v1[1][0] - v1[0][0]) ** 2 + (v1[1][1] - v1[0][1]) ** 2) * ((v2[1][0] - v2[0][0]) ** 2 + (v2[1][1] - v2[0][1]) ** 2))
            )
            self._params_side["gap"].iloc[v_ind, 1] = np.sqrt((v1[0][0] - v2[1][0]) ** 2 + (v1[0][1] - v2[1][1]) ** 2)
            self._params_side["gap"].iloc[v_ind, 2] = np.sqrt((v1[3][0] - v2[2][0]) ** 2 + (v1[3][1] - v2[2][1]) ** 2)
            self._params_side["gap"].iloc[v_ind, 3] = np.arccos(
                ((v1[0][0] - v2[1][0]) * (v1[3][0] - v2[2][0]) + (v1[0][1] - v2[1][1]) * (v1[3][1] - v2[2][1])) / (self._params_side["gap"].iloc[v_ind, 1] * self._params_side["gap"].iloc[v_ind, 2])
            )
            # TODO finish