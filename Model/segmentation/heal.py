import numpy as np
import cv2

from segmentation.elements.vertebrae import Vertebrae

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.optimize, scipy.interpolate, scipy.integrate, scipy.stats

import segmentation.utils

def compute_ref_err(vertebraes, cs_x, cs_y, dcs_x, dcs_y, bcoefs = None, ucoefs = None, t = None):
    s = []
    for vind in range(1, len(vertebraes) - 2):
        if not (bcoefs is None):
            next_bottom_t = bcoefs[0] + bcoefs[1] * t[0][vind * 2]
            part = (next_bottom_t - vertebraes[vind].vpath.start_t - vertebraes[vind].height) / (vertebraes[vind + 1].vpath.start_t - vertebraes[vind].vpath.start_t - vertebraes[vind].height)
            bottom_plate_length = (1 - part) * np.linalg.norm(vertebraes[vind].upper_plate_normal) + part * np.linalg.norm(vertebraes[vind + 1].bottom_plate_normal)
            bottom_middle_point = np.array([cs_x(next_bottom_t), cs_y(next_bottom_t)], dtype=np.float32)
            bottom_normal = np.array([dcs_y(next_bottom_t), -dcs_x(next_bottom_t)], dtype=np.float32)
            bottom_normal /= np.linalg.norm(bottom_normal)

            s = np.append(s, (
                    vertebraes[vind + 1].reference_points[[0, 3]] - np.array([bottom_middle_point + bottom_normal * bottom_plate_length / 2, bottom_middle_point - bottom_normal * bottom_plate_length / 2], dtype=np.float32)
                ) ** 2)
        
        if not (ucoefs is None):
            next_upper_t = ucoefs[0] + ucoefs[1] * t[0][vind * 2 + 1]
            part = (next_upper_t - vertebraes[vind].vpath.start_t - vertebraes[vind].height) / (vertebraes[vind + 1].vpath.start_t - vertebraes[vind].vpath.start_t - vertebraes[vind].height)
            upper_plate_length = (1 - part) * np.linalg.norm(vertebraes[vind].upper_plate_normal) + part * np.linalg.norm(vertebraes[vind + 1].bottom_plate_normal)
            upper_middle_point = np.array([cs_x(next_upper_t), cs_y(next_upper_t)], dtype=np.float32)
            upper_normal = np.array([dcs_y(next_upper_t), -dcs_x(next_upper_t)], dtype=np.float32)
            upper_normal /= np.linalg.norm(upper_normal)

            s = np.append(s, (
                    vertebraes[vind + 1].reference_points[[1, 2]] - np.array([upper_middle_point + upper_normal * upper_plate_length / 2, upper_middle_point - upper_normal * upper_plate_length / 2], dtype=np.float32)
                ) ** 2
            )
    return np.median(s)

def heal(vertebraes: list[Vertebrae]) -> list[Vertebrae]:
    f, ax = plt.subplots(nrows=1, ncols=2)
    
    xy = np.concatenate(
        [
            [
                
                [v.bottom_plate_middle_point[0], v.upper_plate_middle_point[0]],
                [v.bottom_plate_middle_point[1], v.upper_plate_middle_point[1]],
            ]
            for v in vertebraes
        ],
        axis=1,
        dtype=np.float32
    )

    len_old = 0
    t = np.concatenate(
        [
            [
                [v.vpath.start_t, v.vpath.start_t + v.height]
            ]
            for v in vertebraes
        ],
        axis=1,
        dtype=np.float32
    )
    
    cs_x = scipy.interpolate.CubicSpline(t[0], xy[0], bc_type="natural")
    cs_y = scipy.interpolate.CubicSpline(t[0], xy[1], bc_type="natural")
    dcs_x, dcs_y = cs_x.derivative(), cs_y.derivative()
    
    for _ in range(50):
        if np.abs(t[0][-1] - len_old) > 1e-9:
            break
        len_old = t[0][-1]
        t = np.concatenate(
            [
                [
                    [
                        scipy.integrate.quad(segmentation.utils._path_func, 0, v, args=(dcs_x, dcs_y), limit=100)[0]
                    ]
                ]
                for v in t[0]
            ],
            axis=1,
            dtype=np.float32
        )
        cs_x = scipy.interpolate.CubicSpline(t[0], xy[0], bc_type="natural")
        cs_y = scipy.interpolate.CubicSpline(t[0], xy[1], bc_type="natural")
        dcs_x, dcs_y = cs_x.derivative(), cs_y.derivative()
    

    l = np.array([scipy.integrate.quad(segmentation.utils._path_func, 0, vertebraes[vind].vpath.start_t, args=(dcs_x, dcs_y), limit=100)[0] for vind in range(1, len(vertebraes) - 1)])
    # for v in vertebraes[1:-1]:
    #     l.append(l[-1] + v.next_gap_vpath.length + v.height)
    # l = np.array(l)

    bottom_coefs: np.ndarray[np.float32] = None
    err = float("inf")
    for i in np.linspace(0, 1, 1000):
        tmp = scipy.optimize.minimize(
            segmentation.utils._compute_errq,
            [0, 0],
            args=(i, l),
            tol=1e-9,
            method="BFGS",
            jac=segmentation.utils._compute_errq_jac
        ).x

        ue = compute_ref_err(vertebraes, cs_x, cs_y, dcs_x, dcs_y, tmp, None, t)
        if ue < err:
            bottom_coefs = tmp
            err = ue
    n = [bottom_coefs[0] + bottom_coefs[1] * i for i in l[:-1]]
    #bottom_coefs[0] -= np.max(n - l[1:])
    print(compute_ref_err(vertebraes, cs_x, cs_y, dcs_x, dcs_y, bottom_coefs, None, t))
    print("Bottom coefs:")
    print(err)
    lx = sns.lineplot(y=n, x=l[:-1], ax=ax[0])
    lx = sns.lineplot(y=l[1:], x=l[:-1], ax=ax[0])
    lx.set(title="Bottom")
    
    l = np.array([scipy.integrate.quad(segmentation.utils._path_func, 0, vertebraes[vind].vpath.start_t + vertebraes[vind].height, args=(dcs_x, dcs_y), limit=100)[0] for vind in range(1, len(vertebraes) - 1)])

    # for v in vertebraes[1:-1]:
    #     l.append(l[-1] + v.prev_gap_vpath.length + v.height)
    # l = np.array(l)
    upper_coefs: np.ndarray[np.float32] = None
    err = float("inf")
    for i in np.linspace(0, 1, 1000):
        tmp = scipy.optimize.minimize(
            segmentation.utils._compute_errq,
            [0, 0],
            args=(i, l),
            tol=1e-9,
            method="BFGS",
            jac=segmentation.utils._compute_errq_jac
        ).x
        
        ue = compute_ref_err(vertebraes, cs_x, cs_y, dcs_x, dcs_y, None, tmp, t)
        if ue < err:
            upper_coefs = tmp
            err = ue
    n = [upper_coefs[0] + upper_coefs[1] * i for i in l[:-1]]
    #upper_coefs[0] -= np.max(n - l[1:])
    print(compute_ref_err(vertebraes, cs_x, cs_y, dcs_x, dcs_y, None, upper_coefs, t))
    print("Upper coefs:")
    print(err)
    lx = sns.lineplot(y=n, x=l[:-1], ax=ax[1])
    lx = sns.lineplot(y=l[1:], x=l[:-1], ax=ax[1])
    lx.set(title="Upper")

    f, ax = plt.subplots(nrows=1, ncols=2)

    spine_conv: np.ndarray = np.zeros((4200, 2000), dtype=np.uint8)
    for v in vertebraes:
        cv2.fillConvexPoly(spine_conv, v.mask_xy, 255, 1)
        cv2.polylines(spine_conv, [v.reference_points], True, 200, 3)
    for i in np.linspace(t[0][0], t[0][-1], 3000, dtype=np.int32):
        spine_conv[int(cs_y(i)), int(cs_x(i))] = 150

    sns.heatmap(spine_conv, ax=ax[0])

    # Heal sticked vertebraes under one rect
    # vind: np.int32 = 1
    # while vind < len(vertebraes) - 1:
    #     current_upper_t = upper_coefs[0] + upper_coefs[1] * (vertebraes[vind - 1].vpath.start_t + vertebraes[vind - 1].height)

    #     next_bottom_t = bottom_coefs[0] + bottom_coefs[1] * vertebraes[vind].vpath.start_t
    #     next_upper_t = upper_coefs[0] + upper_coefs[1] * current_upper_t
    #     next_middle_t = (next_bottom_t + next_upper_t) / 2

    #     if vertebraes[vind].vpath.start_t < next_middle_t < vertebraes[vind].vpath.start_t + vertebraes[vind].height:
    #         div_t = (current_upper_t + next_bottom_t) / 2

    #         new_normal = np.array([dcs_y(div_t), -dcs_x(div_t)])
    #         new_normal /= np.linalg.norm(new_normal)

    #         new_plate_length = np.linalg.norm(vertebraes[vind].reference_points[0] - vertebraes[vind].reference_points[3])

    #         new_middle_point = np.array([cs_x(div_t), cs_y(div_t)])
            
    #         current_cut_points = np.array(
    #             [
    #                 vertebraes[vind].bottom_plate_middle_point - vertebraes[vind].bottom_plate_normal,
    #                 new_middle_point + new_normal * new_plate_length,
    #                 new_middle_point - new_normal * new_plate_length,
    #                 vertebraes[vind].bottom_plate_middle_point + vertebraes[vind].bottom_plate_normal
    #             ],
    #             dtype=np.int32
    #         )
    #         tmp_canvas_1 = np.zeros(np.max(vertebraes[vind].mask_xy, axis=0)[::-1], dtype=np.uint8)
    #         cv2.fillConvexPoly(tmp_canvas_1, vertebraes[vind].mask_xy, 255, 0)

    #         tmp_canvas_2 = np.zeros(np.max(vertebraes[vind].mask_xy, axis=0)[::-1], dtype=np.uint8)
    #         cv2.fillConvexPoly(tmp_canvas_2, current_cut_points, 255, 0)

    #         new_mask_xy = np.squeeze(
    #                 max(
    #                 cv2.findContours(
    #                     cv2.bitwise_and(tmp_canvas_1, tmp_canvas_2),
    #                     cv2.RETR_TREE,
    #                     cv2.CHAIN_APPROX_NONE
    #                 )[0],
    #                 key=cv2.contourArea
    #             )
    #         )

    #         new_current_vertebrae = Vertebrae(
    #             new_mask_xy
    #         )

    #         next_cut_points = np.array(
    #             [
    #                 new_middle_point + new_normal * new_plate_length,
    #                 vertebraes[vind].upper_plate_middle_point - vertebraes[vind].upper_plate_normal,
    #                 vertebraes[vind].upper_plate_middle_point + vertebraes[vind].upper_plate_normal,
    #                 new_middle_point - new_normal * new_plate_length,
    #             ],
    #             dtype=np.int32
    #         )

    #         tmp_canvas_1 = np.zeros(np.max(vertebraes[vind].mask_xy, axis=0)[::-1], dtype=np.uint8)
    #         cv2.fillConvexPoly(tmp_canvas_1, vertebraes[vind].mask_xy, 255, 0)

    #         tmp_canvas_2 = np.zeros(np.max(vertebraes[vind].mask_xy, axis=0)[::-1], dtype=np.uint8)
    #         cv2.fillConvexPoly(tmp_canvas_2, next_cut_points, 255, 0)

    #         new_mask_xy = np.squeeze(
    #             max(
    #                 cv2.findContours(
    #                     cv2.bitwise_and(tmp_canvas_1, tmp_canvas_2),
    #                     cv2.RETR_TREE,
    #                     cv2.CHAIN_APPROX_NONE
    #                 )[0],
    #                 key=cv2.contourArea
    #             )
    #         )

    #         new_next_vertebrae = Vertebrae(
    #             new_mask_xy
    #         )

    #         vertebraes = vertebraes[:vind] + [new_current_vertebrae, new_next_vertebrae] + vertebraes[vind + 1:]

    #         vertebraes[vind].order_reference_points(vertebraes[vind - 1].central_point, "down")
    #         vertebraes[vind + 1].order_reference_points(vertebraes[vind].central_point, "down")

    #         vertebraes[vind].set_vpath(vertebraes[vind - 1], vertebraes[vind + 1])
    #         vertebraes[vind + 1].set_vpath(vertebraes[vind], vertebraes[vind + 2])

    #     vind += 1
    
    # Heal vertebraes (unseen ones)

    vind: np.int32 = 1
    while vind < len(vertebraes) - 1:
        next_bottom_t = bottom_coefs[0] + bottom_coefs[1] * (t[0][vind * 2])
        next_upper_t = upper_coefs[0] + upper_coefs[1] * (t[0][vind * 2 + 1])
        next_middle_t = (next_bottom_t + next_upper_t) / 2

        if vertebraes[vind].vpath.start_t + vertebraes[vind].height < next_middle_t < vertebraes[vind + 1].vpath.start_t:
            part = (next_upper_t - vertebraes[vind].vpath.start_t - vertebraes[vind].height) / (vertebraes[vind + 1].vpath.start_t - vertebraes[vind].vpath.start_t - vertebraes[vind].height)
            upper_plate_length = (1 - part) * np.linalg.norm(vertebraes[vind].upper_plate_normal) + part * np.linalg.norm(vertebraes[vind + 1].bottom_plate_normal)
            upper_middle_point = np.array([cs_x(next_upper_t), cs_y(next_upper_t)], dtype=np.float32)
            upper_normal = np.array([dcs_y(next_upper_t), -dcs_x(next_upper_t)], dtype=np.float32)
            upper_normal /= np.linalg.norm(upper_normal)

            part = (next_bottom_t - vertebraes[vind].vpath.start_t - vertebraes[vind].height) / (vertebraes[vind + 1].vpath.start_t - vertebraes[vind].vpath.start_t - vertebraes[vind].height)
            bottom_plate_length = (1 - part) * np.linalg.norm(vertebraes[vind].upper_plate_normal) + part * np.linalg.norm(vertebraes[vind + 1].bottom_plate_normal)
            bottom_middle_point = np.array([cs_x(next_bottom_t), cs_y(next_bottom_t)], dtype=np.float32)
            bottom_normal = np.array([dcs_y(next_bottom_t), -dcs_x(next_bottom_t)], dtype=np.float32)
            bottom_normal /= np.linalg.norm(bottom_normal)
            
            new_ref_points = np.array(
                [
                    bottom_middle_point + bottom_normal * bottom_plate_length / 2,
                    upper_middle_point + upper_normal * upper_plate_length / 2,
                    upper_middle_point - upper_normal * upper_plate_length / 2,
                    bottom_middle_point - bottom_normal * bottom_plate_length / 2,
                ],
                dtype=np.int32
            )

            new_vertebrae = Vertebrae(
                new_ref_points
            )

            vertebraes = vertebraes[:vind + 1] + [new_vertebrae] + vertebraes[vind + 1:]

            vertebraes[vind + 1].order_reference_points(vertebraes[vind].central_point, "down")

            vertebraes[vind + 1].set_vpath(vertebraes[vind], vertebraes[vind + 2])

        vind += 1

    # Heal sticked vertebraes (intersect)

    # vind: np.int32 = 1
    # while vind < len(vertebraes) - 2:
    #     comb_points: np.ndarray[np.int32] = np.concatenate([vertebraes[vind].reference_points, vertebraes[vind + 1].reference_points], axis=0)

    #     max_v: np.ndarray[np.int32] = np.max(comb_points, axis=0)
        
    #     tmp_canvas_fst: np.ndarray[np.uint8] = np.zeros(max_v[::-1], dtype=np.uint8)
    #     tmp_canvas_sec: np.ndarray[np.uint8] = np.zeros(max_v[::-1], dtype=np.uint8)

    #     cv2.fillConvexPoly(tmp_canvas_fst, vertebraes[vind].reference_points, 255, 1)
    #     cv2.fillConvexPoly(tmp_canvas_sec, vertebraes[vind + 1].reference_points, 255, 1)

    #     intersection: np.ndarray[np.int32] = np.where(cv2.bitwise_and(tmp_canvas_fst, tmp_canvas_sec))

    #     if intersection[0].shape[0] > 1:
    #         cut_length = np.linalg.norm(vertebraes[vind].reference_points[0] - vertebraes[vind].reference_points[3])

    #         try:
    #             slope, intercept, r, p, se = scipy.stats.linregress(intersection[1], intersection[0])

    #             mx = np.median(intersection[1])

    #             cut = np.array(
    #                 [
    #                     [mx - cut_length, intercept + slope * (mx - cut_length)],
    #                     [mx + cut_length, intercept + slope * (mx + cut_length)]
    #                 ],
    #                 dtype=np.int32
    #             )
                
    #         except Exception as e:
    #             x = intersection[1][0]
    #             y = np.median(intersection[0])

    #             cut = np.array(
    #                 [
    #                     [x, y + cut_length],
    #                     [x, y - cut_length]
    #                 ],
    #                 dtype=np.int32
    #             )
            
    #         main_vec = vertebraes[vind + 1].bottom_plate_middle_point - vertebraes[vind].bottom_plate_middle_point
    #         cut = cut[np.argsort([Vertebrae._get_signed_angle(main_vec, cut[0] - vertebraes[vind].bottom_plate_middle_point) for j in [0, 1]])]

    #         current_cut_points = np.array(
    #             [
    #                 vertebraes[vind].bottom_plate_middle_point - vertebraes[vind].bottom_plate_normal,
    #                 cut[1],
    #                 cut[0],
    #                 vertebraes[vind].bottom_plate_middle_point + vertebraes[vind].bottom_plate_normal
    #             ],
    #             dtype=np.int32
    #         )
    #         tmp_canvas_1 = np.zeros(np.max(vertebraes[vind].mask_xy, axis=0)[::-1], dtype=np.uint8)
    #         cv2.fillConvexPoly(tmp_canvas_1, vertebraes[vind].mask_xy, 255, 0)

    #         tmp_canvas_2 = np.zeros(np.max(vertebraes[vind].mask_xy, axis=0)[::-1], dtype=np.uint8)
    #         cv2.fillConvexPoly(tmp_canvas_2, current_cut_points, 255, 0)

    #         new_mask_xy = np.squeeze(
    #                 max(
    #                 cv2.findContours(
    #                     cv2.bitwise_and(tmp_canvas_1, tmp_canvas_2),
    #                     cv2.RETR_TREE,
    #                     cv2.CHAIN_APPROX_NONE
    #                 )[0],
    #                 key=cv2.contourArea
    #             )
    #         )

    #         new_current_vertebrae = Vertebrae(
    #             new_mask_xy
    #         )

    #         next_cut_points = np.array(
    #             [
    #                 cut[1],
    #                 vertebraes[vind + 1].upper_plate_middle_point - vertebraes[vind + 1].upper_plate_normal,
    #                 vertebraes[vind + 1].upper_plate_middle_point + vertebraes[vind + 1].upper_plate_normal,
    #                 cut[0],
    #             ],
    #             dtype=np.int32
    #         )

    #         tmp_canvas_1 = np.zeros(np.max(vertebraes[vind + 1].mask_xy, axis=0)[::-1], dtype=np.uint8)
    #         cv2.fillConvexPoly(tmp_canvas_1, vertebraes[vind + 1].mask_xy, 255, 0)

    #         tmp_canvas_2 = np.zeros(np.max(vertebraes[vind + 1].mask_xy, axis=0)[::-1], dtype=np.uint8)
    #         cv2.fillConvexPoly(tmp_canvas_2, next_cut_points, 255, 0)

    #         new_mask_xy = np.squeeze(
    #             max(
    #                 cv2.findContours(
    #                     cv2.bitwise_and(tmp_canvas_1, tmp_canvas_2),
    #                     cv2.RETR_TREE,
    #                     cv2.CHAIN_APPROX_NONE
    #                 )[0],
    #                 key=cv2.contourArea
    #             )
    #         )

    #         new_next_vertebrae = Vertebrae(
    #             new_mask_xy
    #         )

    #         vertebraes = vertebraes[:vind] + [new_current_vertebrae, new_next_vertebrae] + vertebraes[vind + 2:]

    #         vertebraes[vind].order_reference_points(vertebraes[vind - 1].central_point, "down")
    #         vertebraes[vind + 1].order_reference_points(vertebraes[vind].central_point, "down")

    #         vertebraes[vind].set_vpath(vertebraes[vind - 1], vertebraes[vind + 1])
    #         vertebraes[vind + 1].set_vpath(vertebraes[vind], vertebraes[vind + 2])

        # vind += 1

    spine_conv: np.ndarray = np.zeros((4200, 2000), dtype=np.uint8)
    for v in vertebraes:
        cv2.fillConvexPoly(spine_conv, v.mask_xy, 255, 1)
        cv2.polylines(spine_conv, [v.reference_points], True, 200, 3)
    for i in np.linspace(t[0][0], t[0][-1], 3000):
        spine_conv[int(cs_y(i)), int(cs_x(i))] = 150

    sns.heatmap(spine_conv, ax=ax[1])

    plt.show()

    return vertebraes

class nHeal:
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
        max_xy: np.int32 = np.max(vertebrae.mask_xy, axis=0)
        tmp_canvas: np.ndarray[np.uint8] = np.zeros(max_xy[::-1] + np.array([l, l], dtype=np.int32), dtype=np.uint8)
        
        cv2.fillConvexPoly(tmp_canvas, vertebrae.mask_xy, 255, 1)

        return tmp_canvas

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
        d_lines: list[np.ndarray[np.int32]] = []
        is_prev_vertebrae: bool = False
        for t in np.linspace(0, 1, n):
            p: np.ndarray[np.float32] = vertebrae.lPath.f(t)
            normal: np.ndarray[np.float32] = vertebrae.lPath.fn(t)
            p1: np.ndarray[np.int32] = (p + normal * l).astype(np.int32)
            p2: np.ndarray[np.int32] = (p - normal * l).astype(np.int32)

            c: np.float32 = nHeal._line_intersect(canvas, p1, p2)

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
        d_rects: np.ndarray[np.int32] = nHeal._get_dividing_rects(vertebrae, canvas, n, tau, l)

        new_tmp_canvas: np.ndarray[np.uint8] = np.zeros_like(canvas, dtype=np.uint8)
        cv2.fillPoly(new_tmp_canvas, d_rects, 255, 0)
        
        return 1 - np.sum(cv2.bitwise_and(canvas, new_tmp_canvas)) / np.sum(canvas)
    
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

        tmp_canvas: tuple = nHeal._make_canvas(vertebrae, l)

        # tau = scipy.optimize.minimize(
        #     Heal._compute_dividing_error,
        #     0.5,
        #     method="BFGS",
        #     args=(vertebrae, tmp_canvas, n, l)
        # ).x
        tau = 0.5
        print(tau)
        d_rects: np.ndarray[np.int32] = nHeal._get_dividing_rects(vertebrae, tmp_canvas, n, tau, l)

        res: list[Vertebrae] = []
        for d in d_rects:
            tmp_canvas_cpy: np.ndarray[np.uint8] = np.zeros_like(tmp_canvas)
            cv2.fillConvexPoly(tmp_canvas_cpy, d, 255, 0)

            new_vertebrae: Vertebrae = Vertebrae(
                np.squeeze(
                    max(
                        cv2.findContours(cv2.bitwise_and(tmp_canvas_cpy, tmp_canvas), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0], 
                        key=cv2.contourArea
                    )
                )
            )
            
            res.append(new_vertebrae)
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
                    new_vertebrae.set_lPath(vertebraes[v - 1], vertebraes[v + 2])

                    vertebraes = vertebraes[:v] + [new_vertebrae] + vertebraes[v + 2:]

            new_vertebraes: list[Vertebrae] = nHeal._divide_sticked_single(vertebraes[v], n)
            vertebraes = vertebraes[:v] + new_vertebraes + vertebraes[v + 1:]
            for i in range(len(new_vertebraes)):
                vertebraes[v + i].order_reference_points(vertebraes[v + i - 1].central_point, "down")
            for i in range(len(new_vertebraes)):
                vertebraes[v + i].set_lPath(vertebraes[v + i - 1], vertebraes[v + i + 1])
            v += 1
        
        return vertebraes