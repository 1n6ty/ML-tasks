import numpy as np
import cv2
import os

from ultralytics import YOLO
from ultralytics.engine.results import Results

from segmentation.elements.vertebrae import Vertebrae
from segmentation.interpolation.path import lPath
import segmentation.heal

class Spine:
    """Class for spine building and computing parameters.

        Builds up the spine-array (array of vertebraes) for frontal and side projections,
        sorts them, heals concatenated vertebraes and fills large gaps with vertebraes,
        links two projections and computes medical parameters related to Gladkov's work.

        Attributes
        ----------
            vertebraes (List[Vertebrae])
                List of vertebraes in spine
    """
    def __init__(self, side_pixel_array: np.ndarray[np.uint8], front_pixel_array: np.ndarray[np.uint8], segmentation_model: YOLO) -> None:
        """Constructor for spine.
        
            Finds vertebraes on pixel arrays, sorts, heals and reveals missing ones,
            links projections and computes medical parameters.

            Args
            ----
                side_pixel_array: (np.ndarray[np.uint8])
                    Gray-Scaled pixel array of side projection of patient X-Ray
                front_pixel_array: (np.ndarray[np.uint8])
                    Gray-Scaled pixel array of frontal projection of patient X-Ray
                segmentation_model (YOLO):
                    Segmentation model that can find vertebraes
        """
        cv2.imwrite("tmp.png", side_pixel_array)

        model_response: list[Results] = segmentation_model.predict("tmp.png", save=False, show=False, show_boxes=False, conf=0.5)

        if os.path.exists("tmp.png"):
            os.remove("tmp.png")

        self.vertebraes: list[Vertebrae] = [Vertebrae(mask_xy=vm.astype(np.int32)) for vm in model_response[0].masks.xy]
        
        self._order_vertebraes()
        self._order_vertebraes_reference_points()
        self._set_lpaths()

        prev_v_r = np.array([v.reference_points for v in self.vertebraes])

        self.vertebraes.pop(-4)
        self.vertebraes.pop(-4)
        self.vertebraes.pop(-4)

        self.vertebraes = segmentation.heal.heal(self.vertebraes)
        
        err = np.abs(prev_v_r - np.array([v.reference_points for v in self.vertebraes]))
        err = np.concatenate(err, axis=0)
        print(np.sort(np.linalg.norm(err, axis=1)))
        # spine_conv: np.ndarray = np.zeros_like(side_pixel_array, dtype=np.uint8)

        # for v in self.vertebraes:
        #     cv2.fillConvexPoly(spine_conv, v.mask_xy, 255, 1)
        #     cv2.polylines(spine_conv, [v.reference_points], True, 200, 3)
        #     # for t in np.linspace(0, 1, 200):
        #     #     if not (v.next_gap_vpath is None):
        #     #         p = v.next_gap_vpath.f(t).astype(np.int32)
        #     #         spine_conv[p[1], p[0]] = 150
        #     #     if not (v.prev_gap_vpath is None):
        #     #         p = v.prev_gap_vpath.f(t).astype(np.int32)
        #     #         spine_conv[p[1], p[0]] = 150
        #     #     p = v.vpath.f(t).astype(np.int32)
        #     #     spine_conv[p[1], p[0]] = 150
        # # for n in np.linspace(t_spl[0], t_spl[-1], 6000):
        # #     spine_conv[int(cs_y(n)), int(cs_x(n))] = 150
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # sns.heatmap(spine_conv)
        # plt.show()

    def _set_lpaths(self) -> None:
        """Sets lpath for vertebraes and gaps between them.
        """
        vertebraes_extended: list[Vertebrae] = [None, *self.vertebraes, None]
        for vi in range(1, len(vertebraes_extended) - 1):
            vertebraes_extended[vi].set_vpath(vertebraes_extended[vi - 1], vertebraes_extended[vi + 1], 4)

    def _order_vertebraes_reference_points(self) -> None:
        """Orders reference points of vertebraes.

            Orders reference points of vertebraes related by Gladkov's work.
            First point of the vertebrae is left-bottom one, than others clock-wise.
        """
        vertebraes_length: np.int32 = len(self.vertebraes)
        if vertebraes_length > 1:
            self.vertebraes[0].order_reference_points(self.vertebraes[1].central_point, "up")
            for i in range(1, vertebraes_length):
                self.vertebraes[i].order_reference_points(self.vertebraes[i - 1].central_point, "down")

    def _order_vertebraes(self) -> None:
        """Orders vertebraes from S1 to C2.
        """
        vertebraes_central_points: np.ndarray[np.float32] = np.concatenate([[v.central_point] for v in self.vertebraes], axis=0)
        self.vertebraes: list[Vertebrae] = [
            self.vertebraes[i] 
            for i in Spine._TSP_solve(vertebraes_central_points, prefix=np.array([np.argmax(vertebraes_central_points[:, 1])], dtype=np.int32))
        ]

    @staticmethod
    def _TSP_solve(central_points: np.ndarray[np.float32], prefix: np.ndarray[np.int32]) -> np.ndarray[np.int32]:
        """Solves TSP problem in gready way.

            Solves TCP problem in gready way to obtain right order of vertebraes.
        
            Args
            ----
                central_points (np.ndarray[np.float32])
                    Array of central points of vertebraes to be ordered
                prefix (np.ndarray[np.int32])
                    Ordered indexes, which go first
            
            Returns
            -------
                indexes (np.ndarray[np.int32])
                    indexes of the right order of vertebraes; from S1 to C2 if `prefix` is S1-index
        """
        if central_points.shape[0] == prefix.shape[0]:
            return prefix

        distances: np.float32 = np.linalg.norm(central_points - central_points[prefix[-1]], axis=1)
        distances[prefix] = np.inf

        return Spine._TSP_solve(central_points, np.concatenate([prefix, np.array([np.argmin(distances)], dtype=np.int32)]))