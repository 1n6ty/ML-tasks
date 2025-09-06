import numpy as np
import cv2

from segmentation.interpolation.path import vPath

from typing import Literal

class Vertebrae:
    """Class of basic vertebrae.

        This class is used to compute and store raw data about vertebrae. It computes
        path of the vertebrae and reference points.

        Attributes
        ----------
            mask_xy (np.ndarray[np.int32] | None)
                2D array of vertebrae mask `(x, y)` coordinates
            reference_points (np.ndarray[np.int32])
                Array of shape `(4, 2)` - (x, y) coordinates of reference points
            central_point (np.ndarray[np.float32])
                Center of vertebrae
            vpath (vPath)
                Skeleton-Path of vertebrae
            height (np.float32)
                Alias for vpath.length
            next_gap_vpath (vPath)
                Skeleton-Path of gap after current vertebrae
            prev_gap_vpath (vPath)
                Skeleton-Path of gap before current vertebrae
            upper_plate_middle_point (np.ndarray[np.float32])
                Middle point of upper plate of vertebrae
            bottom_plate_middle_point (np.ndarray[np.float32])
                Middle point of bottom plate of vertebrae
            upper_plate_normal (np.ndarray[np.int32])
                Normal vector of upper plate of vertebrae `from 2nd to 3rd point`
            bottom_plate_normal (np.ndarray[np.int32])
                Normal vector of bottom plate of vertebrae `from 1st to 4th point`
            
    """
    def __init__(self, mask_xy: np.ndarray[np.int32] | None = None, reference_points: np.ndarray[np.int32] | None = None) -> None:
        """Vertebrae constructor.

            Args
            ----
                mask_xy (np.ndarray[np.int32] | None)
                    2D array of vertebrae mask `(x, y)` coordinates
                reference_points (np.ndarray[np.int32] | None)
                    Array of shape `(4, 2)` - (x, y) coordinates of reference points
        """
        if not (mask_xy is None):
            self.mask_xy: np.ndarray[np.int32] = mask_xy

            if reference_points is None:
                self.reference_points: np.ndarray[np.int32] = Vertebrae._compute_reference_points(mask_xy)
        if not(reference_points is None):
            self.reference_points: np.ndarray[np.int32] = reference_points
        
        if mask_xy is None and reference_points is None:
            raise AttributeError("Vertebrae constructor needs mask or reference_points array.")
        
        self.central_point: np.ndarray[np.float32] = np.average(self.reference_points, axis=0)

        self.upper_plate_middle_point: np.ndarray[np.float32] | None = None
        self.bottom_plate_middle_point: np.ndarray[np.float32] | None = None

        self.upper_plate_normal: np.ndarray[np.int32] | None = None
        self.bottom_plate_normal: np.ndarray[np.int32] | None = None

        self.vpath: vPath | None = None
        self.prev_gap_vpath: vPath | None = None
        self.next_gap_vpath: vPath | None = None

        self.height: np.float32 | None = None

    def set_vpath(self, prev_vertebrae: 'Vertebrae' = None, next_vertebrae: 'Vertebrae' = None) -> None:
        """Setter for vpath variables.

            Sets vpath variable with skeleton class (parametrized middle curve of vertebrae).
            Sets vpath for gaps between previous, current and next vertebraes. Also alliases 
            height -> vpath.length.

            Args
            ----
                prev_vertebrae (Vertebrae)
                    Previous vertebrae object. If `None` then no gap_vpath will be set
                next_vertebrae (Vertebrae)
                    Next vertebrae object. If `None` then no gap_vpath will be set
        """
        total_t: np.float32 = 0
        if not (prev_vertebrae is None):
            self.prev_gap_vpath = vPath(
                prev_vertebrae.upper_plate_middle_point,
                self.bottom_plate_middle_point,
                prev_vertebrae.upper_plate_normal,
                self.bottom_plate_normal,
                prev_vertebrae.vpath.start_t + prev_vertebrae.height
            )
            total_t += prev_vertebrae.vpath.start_t + prev_vertebrae.height + self.prev_gap_vpath.length
        self.vpath = vPath(
            self.bottom_plate_middle_point,
            self.upper_plate_middle_point,
            self.bottom_plate_normal,
            self.upper_plate_normal,
            total_t
        )
        self.height = self.vpath.length
        total_t += self.height
        if not (next_vertebrae is None):
            self.prev_gap_vpath = vPath(
                self.upper_plate_middle_point,
                next_vertebrae.bottom_plate_middle_point,
                self.upper_plate_normal,
                next_vertebrae.bottom_plate_normal,
                total_t
            )

    @staticmethod
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

    def order_reference_points(self, nearest_vertebrae_central_point: np.ndarray[np.float32], central_point_type: Literal["up", "down"]) -> None:
        """Orders reference points like in Gladkov's work.

            Orders points like: first point of the vertebrae is left-bottom one, than others clock-wise.
            This method relies on nearest vertebrae's central point to determinate up and down plates.
        
            Args
            ----
                nearest_vertebrae_central_point (np.ndarray[np.float32])
                    Nearest vertebrae's central point, up or down one
                central_point_type (Literal["up", "down"])
                    Position of central point relative to the current vertebrae `up` or `down`
        """
        nearest_indexes: np.ndarray = np.argpartition(np.linalg.norm(self.reference_points - nearest_vertebrae_central_point, axis=1), 2)[:2]

        if central_point_type == "up":
            up: np.ndarray[np.int32] = self.reference_points[nearest_indexes]
            down: np.ndarray[np.int32] = np.delete(self.reference_points, nearest_indexes, axis=0)
        elif central_point_type == "down":
            up: np.ndarray[np.int32] = np.delete(self.reference_points, nearest_indexes, axis=0)
            down: np.ndarray[np.int32] = self.reference_points[nearest_indexes]
        
        down_avg: np.ndarray[np.float32] = np.average(down, axis=0)
        main_vec: np.ndarray[np.float32] = np.average(up, axis=0) - down_avg
        self.reference_points = self.reference_points[np.argsort([Vertebrae._get_signed_angle(main_vec, self.reference_points[j] - down_avg) for j in range(self.reference_points.shape[0])])]

        self.upper_plate_normal = self.reference_points[2] - self.reference_points[1]
        self.bottom_plate_normal = self.reference_points[3] - self.reference_points[0]

        self.upper_plate_middle_point = np.average(up, axis=0)
        self.bottom_plate_middle_point = down_avg

    @staticmethod
    def _compute_cnt_weight(cnt: np.ndarray[np.int32], index: np.int32) -> np.float32:
        """Computes weight by rule: hypotenuse substruct legs.

            Args
            ----
                cnt (np.ndarray[np.int32])
                    Vertebrae contour
                index (np.int32)
                    Index, around which, weight will be computed
            
            Returns
            -------
                weight (np.float32)
                    Computed weight of `index` point
        """
        return np.linalg.norm(cnt[index] - cnt[index - 1]) + np.linalg.norm(cnt[index] - cnt[(index + 1) % cnt.shape[0]]) - np.linalg.norm(cnt[(index + 1) % cnt.shape[0]] - cnt[index - 1])
    
    @staticmethod
    def _compute_reference_points(mask_xy: np.ndarray[np.int32]) -> np.ndarray[np.int32]:
        """Computes reference points of vertebrae, based on `mask`.

            Args
            ----
                mask_xy (np.ndarray[np.int32])
                    2D array of vertebrae mask `(x, y)` coordinates
            
            Returns
            -------
                reference points (np.ndarray[np.int32])
                    Array of shape `(4, 2)` - (x, y) coordinates of reference points
        """
        cnt: np.ndarray[np.int32] = np.copy(mask_xy)
        
        weights: np.ndarray[np.float32] = np.array([Vertebrae._compute_cnt_weight(cnt, i) for i in range(cnt.shape[0])], dtype=np.float32)
        while cnt.shape[0] > 4:
            min_weight_index: np.int32 = np.argmin(weights)
            
            cnt = np.delete(cnt, min_weight_index, axis=0)
            weights = np.delete(weights, min_weight_index, axis=0)

            weights[min_weight_index % cnt.shape[0]] = Vertebrae._compute_cnt_weight(cnt, min_weight_index % cnt.shape[0])
            weights[min_weight_index - 1] = Vertebrae._compute_cnt_weight(cnt, min_weight_index - 1)
        
        return cnt
    