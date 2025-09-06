import numpy as np
import cv2
import scipy.optimize

from segmentation.elements.vertebrae import Vertebrae
from segmentation.interpolation.path import vPath

class Heal:
    """Class for healing spine.

        Provide methods to heal vertebraes in spine - divide sticked vertebraes, 
        divide overlaped masks and insert unseen ones.

        Attributes
        ----------

    """
    @staticmethod
    def _make_canvas(vertebrae: Vertebrae, l: np.float32) -> tuple[np.ndarray[np.uint8], np.ndarray[np.int32]]:
        """Makes canvas for "Sliding normal" method.

            Args
            ----
                vertebrae (Vertebrae)
                    Vertebrae to be healed
                l (np.float32)
                    Half of the line "normal" length
                    
            Returns
            -------
                tuple (tuple[np.ndarray[np.uint8], np.ndarray[np.int32]])
                    canvas with vertebrae mask at it (np.ndarray[uint8]) and transform vector to new coord system (np.ndarray[np.int32])
        """
        
        max_c: np.ndarray[np.int32] = np.max(vertebrae.mask_xy, axis=0).astype(np.int32)
        min_c: np.ndarray[np.int32] = np.min(vertebrae.mask_xy, axis=0).astype(np.int32)

        transform_vec: np.ndarray[np.int32] = (np.array([l, l]) - (max_c + min_c) / 2).astype(np.int32)

        tmp_canvas: np.ndarray[np.uint8] = np.zeros(2 * np.array([l, l], dtype=np.int32), dtype=np.uint8)
        
        cv2.fillConvexPoly(tmp_canvas, vertebrae.mask_xy + transform_vec, 255, 1)

        return (tmp_canvas, transform_vec)

    @staticmethod
    def _line_intersect(canvas: np.ndarray[np.uint8], p1: np.ndarray[np.int32], p2: np.ndarray[np.float32]) -> np.float32:
        """Computes intersection coef over line.

            Draws line from `start_p` in the dirrection of `normal_v` and back,
            computes intersection over union with vertebrae mask.
        
            Args
            ----
                canvas (np.ndarray[np.uint8])
                    Canvas with vertebrae mask (255 - vertebrae pixel, 0 otherwise)
                p1 (np.ndarray[np.int32])
                    First point of a line
                p1 (np.ndarray[np.int32])
                    Second point of a line
            
            Returns
            -------
                coef (np.float32)
                    Intersection over union coef
        """
        canvas_empty: np.ndarray[np.uint8] = np.zeros_like(canvas)
        
        cv2.line(
            canvas_empty, 
            p1,
            p2,
            255,
            1,
            0
        )
        return np.sum(cv2.bitwise_and(canvas, canvas_empty)) / np.sum(canvas_empty)

    @staticmethod
    def _get_dividing_rects(vertebrae: Vertebrae, canvas: tuple[np.ndarray[np.uint8], np.ndarray[np.int32]], n: np.int32, tau: np.float32, l: np.float32) -> np.ndarray[np.int32]:
        """Computes dividing rects for sticked vertebraes.

            Uses "Sliding normal" method. Computes dividing rects based on `tau` as a threshold:
            if intersection over union of a line and vertebrae is greater then `tau` -> vertebrae,
            else -> gap between vertebraes.

            Args
            ----
                vertebrae (Vertebrae)
                    The vertebrae to be healed
                canvas (tuple[np.ndarray[np.uint8], np.ndarray[np.int32]])
                    Canvas with vertebrae mask at it and transform vector to new coord system
                n (np.int32)
                    Partition number for iterations
                tau (np.float32)
                    Threshold number
                l (np.float32)
                    Half of the line "normal" length
            
            Returns
            -------
                dividing_lines (np.ndarray[np.int32])
                    Array of `shape = [k, 4]` - k rects.
                    In new coord system!
        """
        tmp_canvas, transform_vec = canvas

        d_lines: list[np.ndarray[np.int32]] = []
        is_prev_vertebrae: bool = False
        for t in np.linspace(0, 1, n):
            p: np.ndarray[np.float32] = vertebrae.vpath.f(t) + transform_vec
            normal: np.ndarray[np.float32] = vertebrae.vpath.fn(t)
            p1: np.ndarray[np.int32] = (p + normal * l).astype(np.int32)
            p2: np.ndarray[np.int32] = (p - normal * l).astype(np.int32)
            
            spine_conv = np.zeros((np.max(vertebrae.mask_xy), np.max(vertebrae.mask_xy)), dtype=np.uint8)
            cv2.fillConvexPoly(spine_conv, vertebrae.mask_xy, 255, 1)
            cv2.line(spine_conv, p1, p2, 180, 1, 0)
            for n in np.linspace(0, 1, 2000):
                    spine_conv[*vertebrae.vpath.f(n)[::-1].astype(np.int32)] = 180
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.heatmap(spine_conv)
            plt.show()

            c: np.float32 = Heal._line_intersect(tmp_canvas, p1, p2)

            if (c >= tau and (not is_prev_vertebrae)) or (c < tau and is_prev_vertebrae):
                d_lines.append(np.array([p1, p2]))
                is_prev_vertebrae = not is_prev_vertebrae
        
        d_lines[0] = np.array([vertebrae.reference_points[0], vertebrae.reference_points[3]])
        if len(d_lines) % 2:
            d_lines.append(np.array([p1, p2]))
        d_lines[-1] = np.array([vertebrae.reference_points[1], vertebrae.reference_points[2]])

        return np.array([[d_lines[l][0], *d_lines[l + 1], d_lines[l][1]] for l in range(0, len(d_lines), 2)], dtype=np.int32)

    @staticmethod
    def _compute_dividing_error(tau: np.float32, vertebrae: Vertebrae, canvas: tuple[np.ndarray[np.uint8], np.ndarray[np.int32]], n: np.int32, l: np.float32) -> np.float32:
        """Computes error of found dividing lines.
        
            With use of intersection over union of vertebrae mask and new mask from dividing lines
            computes error.
        
            Args
            ----
                vertebrae (Vertebrae)
                    The vertebrae to be healed
                canvas (tuple[np.ndarray[np.uint8], np.ndarray[np.int32]])
                    Canvas with vertebrae mask at it and transform vector to new coord system
                n (np.int32)
                    Partition number for iterations
                tau (np.float32)
                    Threshold number
                l (np.float32)
                    Half of the line "normal" length
            
            Returns
            -------
                Error (np.float32)
                    IOU error
        """
        d_rects: np.ndarray[np.int32] = Heal._get_dividing_rects(vertebrae, canvas, n, tau, l)

        tmp_canvas, transform_vec = canvas

        new_tmp_canvas: np.ndarray[np.uint8] = np.zeros_like(tmp_canvas, dtype=np.uint8)
        cv2.fillPoly(new_tmp_canvas, d_rects, 255, 0)
        
        return 1 - np.sum(cv2.bitwise_and(tmp_canvas, new_tmp_canvas)) / np.sum(cv2.bitwise_or(tmp_canvas, new_tmp_canvas))
    
    @staticmethod
    def _divide_sticked_single(vertebrae: Vertebrae, n: np.int32) -> list[Vertebrae]:
        """Divides sticked masks inside one vertebrae.

            By "Sliding Normal" method divides sticked vertebrae.

            Args
            ----
                vertebrae (Vertebrae)
                    Vertebrae to be healed.
                n (np.int32)
                    Partition number for iterations
            
            Returns
            -------
                vertebraes (list[Vertebrae])
                    List of unsticked vertebraes.
        """
        
        l: np.float32 = np.max([np.linalg.norm(vertebrae.central_point - m) for m in vertebrae.mask_xy])

        canvas: tuple = Heal._make_canvas(vertebrae, l)
        tmp_canvas, transform_vec = canvas

        tau = 0.5
        # tau = scipy.optimize.minimize(
        #     Heal._compute_dividing_error,
        #     0.5,
        #     method="BFGS",
        #     args=(vertebrae, canvas, n, l)
        # ).x
        print(tau)
        print("REF-------------")
        print(vertebrae.reference_points)
        d_rects: np.ndarray[np.int32] = Heal._get_dividing_rects(vertebrae, canvas, n, tau, l)

        res: list[Vertebrae] = []
        for d in d_rects:
            tmp_canvas_cpy: np.ndarray[np.uint8] = np.copy(tmp_canvas)
            cv2.fillConvexPoly(tmp_canvas_cpy, d, 255, 0)

            mask = np.where(
                cv2.bitwise_and(tmp_canvas_cpy, tmp_canvas) >= 128
            )
            mask_xy: np.ndarray[np.int32] = np.concatenate(
                [np.expand_dims(mask[1], axis=1), np.expand_dims(mask[0], axis=1)],
                axis=1
            )
            res.append(Vertebrae(mask_xy - transform_vec))
            print(res[-1].reference_points)
        return res

    @staticmethod
    def divide_sticked(vertebraes: list[Vertebrae], n: np.int32) -> list[Vertebrae]:
        """Divides sticked vertebraes.

            By "Sliding Normal" method divides sticked vertebraes.

            Args
            ----
                vertebraes (list[Vertebrae])
                    List of vertebraes to be healed.
                n (np.int32)
                    Partition number for iterations
            
            Returns
            -------
                vertebraes (list[Vertebrae])
                    List of unsticked vertebraes.
        """
        v: np.int32 = 1
        while v < len(vertebraes) - 1:
            if v < len(vertebraes) - 2:
                comb_points: np.ndarray[np.int32] = np.concatenate([vertebraes[v].reference_points, vertebraes[v + 1].reference_points], axis=0)

                max_v: np.ndarray[np.int32] = np.max(comb_points, axis=0)
                min_v: np.ndarray[np.int32] = np.min(comb_points, axis=0)
                
                tmp_canvas_fst: np.ndarray[np.uint8] = np.zeros((max_v - min_v)[::-1], dtype=np.uint8)
                tmp_canvas_sec: np.ndarray[np.uint8] = np.zeros((max_v - min_v)[::-1], dtype=np.uint8)

                cv2.fillConvexPoly(tmp_canvas_fst, vertebraes[v].reference_points - min_v, 255, 1)
                cv2.fillConvexPoly(tmp_canvas_sec, vertebraes[v + 1].reference_points - min_v, 255, 1)

                if cv2.countNonZero(cv2.bitwise_and(tmp_canvas_fst, tmp_canvas_sec)):
                    tmp_canvas: np.ndarray[np.uint8] = np.zeros(max_v[::-1], dtype=np.uint8)
                    cv2.fillConvexPoly(tmp_canvas, vertebraes[v].mask_xy, 255, 1)
                    cv2.fillConvexPoly(tmp_canvas, vertebraes[v + 1].mask_xy, 255, 1)

                    new_vertebrae: Vertebrae = Vertebrae(
                        np.squeeze(
                            max(
                                cv2.findContours(tmp_canvas, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0], 
                                key=cv2.contourArea
                            )
                        ),
                        np.array([vertebraes[v].reference_points[0], vertebraes[v + 1].reference_points[1], vertebraes[v + 1].reference_points[2], vertebraes[v].reference_points[3]])
                    )

                    new_vertebrae.order_reference_points(vertebraes[v - 1].central_point, "down")
                    new_vertebrae.set_vpath(vertebraes[v - 1], vertebraes[v + 2])

                    vertebraes = vertebraes[:v] + [new_vertebrae] + vertebraes[v + 2:]

                    new_vertebraes: list[Vertebrae] = Heal._divide_sticked_single(vertebraes[v], n)
                    
                    vertebraes = vertebraes[:v] + new_vertebraes + vertebraes[v + 1:]
                    for i in range(len(new_vertebraes)):
                        vertebraes[v + i].order_reference_points(vertebraes[v + i - 1].central_point, "down")
                        vertebraes[v + i].set_vpath(vertebraes[v + i - 1], vertebraes[v + i + 1])
            v += 1
        
        return vertebraes