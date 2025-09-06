"""
    This file contains methods to calculate reference points of vertebraes
"""

import numpy as np
import numpy.typing as npt

import cv2

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

from typing import Literal

from spine_segmentation.vertebraes.typings import VERTEBRAES_R_POINTS_PRJ, VERTEBRAES_R_POINTS

gamma = 1e-6 # used as 0 in derivative

height_coef = [0.3857, 1.0925, 0.9491, 1.0714, 1.0666, 1.0625, 1.0441, 1.0845, 1.0259, 1.0379, 1.0243, 1.0, 1.0238, 1.0465, 1.0222, 1.0652, 1.0408, 1.0392, 1.0, 1.0188, 1.037, 0.9464, 3] # vertebrae[i + 1].height / vertebrae[i].height
up_plates_coef = [1.0, 1.0493, 1.0, 1.0277, 1.0, 1.0, 1.0526, 1.0476, 1.0, 1.0, 1.0175, 1.0161, 1.0303, 1.0142, 1.014, 1.0, 1.0, 1.0135, 1.0263, 1.0, 1.0, 1.0, 1.0] # vertebrae[i + 1].up_plate / vertebrae[i].down_plate
down_plates_coef = [1.0312, 1.0801, 1.0285, 1.0277, 1.0, 1.027, 1.1052, 1.1903, 1.08, 1.0555, 1.0876, 1.0644, 1.0605, 1.0142, 1.014, 1.0277, 1.0, 1.0269, 1.0263, 1.0256, 1.0, 1.0, 0.0] # vertebrae[i + 1].down_plate / vertebrae[i].down_plate
plates_front = [
    [18, 18], [20, 21], [22, 23], [24, 25], [26, 28], [28, 30], [30, 30], [30, 30], [30, 30], [30, 31], [32, 33], [34, 35], [36, 37], [38, 39], [40, 41], [42, 43], [44, 45], [46, 46], [46, 47], [47, 48], [48, 49], [49, 50], [50, 51], [52, 52]
] # [up_plate_d, bottom_plate_d]

gap_heights_coef = [3.46, 3.71, 3.94, 3.94, 4.00, 4.02, 4.37, 4.04, 3.97, 4.44, 4.16, 4.25, 4.56, 4.78, 4.01, 4.21, 2.81, 2.88, 2.79, 2.26, 2.28, 1.90] # vertebrae[i].height / vertebra[i].down_disc_height

def __get_real_threshold_and_corner_point(pixel_array: npt.NDArray[np.float32], borders: npt.NDArray[np.int32], pivot: npt.NDArray[np.float32], delta: npt.NDArray[np.float32], y: npt.NDArray[np.float32], mode: Literal["left", "right"], r_coef = 9) -> tuple[tuple[float, float], npt.NDArray[np.float32]]:
    """
        Computes list of [[sum, pixels_count], corner_point] over line from pivot to `r_coef * y`

        Parameters:
        -----------
            pixel_array:
                2d array of pixels - spine image
            \n
            borders:
                Bounding rectangle of the spine - `[up-left, down-right]` coords
            \n
            pivot:
                Point to start from
            \n
            delta:
                Direction to go to
            \n
            y:
                Edge of spine
            \n
            mode:
                In which direction algoritm goes `(left or right)`
            \n
            r_coef:
                Coefficient to multiply the square of radius on `default value is 9`
    """

    reach_edge = False # indicator of reaching the edge

    r = 0
    delta_multiplier = (-1 if delta[1] > 0 else 1) if mode == "left" else (1 if delta[1] > 0 else -1)
    coords = np.copy(pivot)
    line = np.zeros_like(pixel_array, dtype=np.float32)
    while not reach_edge:
        if not reach_edge:
            if (0 <= coords[0] < y.shape[0]) and (0 <= borders[0, 0] + coords[0] < pixel_array.shape[0]) and (0 < coords[1] < pixel_array.shape[1]) and (r == 0 or np.sum((pivot - coords) ** 2) <= r * r_coef):
                if (r == 0) and ((mode == "left" and y[int(coords[0])] >= coords[1]) or (mode == "right" and y[int(coords[0])] <= coords[1])):
                    r = np.sum((pivot - coords) ** 2)
                line[borders[0, 0] + int(coords[0]), int(coords[1])] = 1.0
                coords += delta * delta_multiplier
            else:
                reach_edge = True

    return ((np.sum(np.multiply(pixel_array, line)), np.sum(line)), coords)

def _get_vertebraes_dividing_lines(pixel_array: npt.NDArray[np.float32], borders: npt.NDArray[np.int32], y: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]], threshold: float) -> npt.NDArray[np.float32]:
    """
    Based on data image (0 <= pixel <= 1) and (middle regression, left-edge, right-edge) computes dividing line for each vertebraes' pair

    Parameters:
    -----------
        pixel_array:
            2d array of pixels - spine image
        \n
        borders:
            Bounding rectangle of the spine - `[up-left, down-right]` coords
        \n
        y:
            Tuple of computed edges and their mean `(mean, left, right)`
        \n
        threshold:
            if `<= line_mean` - then it is vertebrae, gap otherwise
    """
    vertebraes = [] # to store final result

    y_m, y_l, y_r = y
    
    dy_m = np.concatenate([y_m[1:], y_m[-1:]]) - y_m; dy_m = np.where(dy_m != 0, dy_m, np.full_like(dy_m, gamma))
    norm_tan = np.divide(-np.ones_like(y_m, dtype=np.float32), dy_m) # tan of normal
    
    prev_state = False # True - previous state was VERTEBRAE, False - GAP
    with ThreadPool(2) as p:
        for x in range(norm_tan.shape[0]):
            delta = np.array([1.0, norm_tan[x]], dtype=np.float32); delta = delta / np.max(np.abs(delta))
            pivot = np.array([x, y_m[x]], dtype=np.float32)
            summary = np.zeros((2, ), dtype=np.float32)

            corner_points = p.starmap(__get_real_threshold_and_corner_point, [(pixel_array, borders, pivot, delta, y_l, "left"), (pixel_array, borders, pivot, delta, y_r, "right")])
            
            summary[0] = corner_points[0][0][0] + corner_points[1][0][0]
            summary[1] = corner_points[0][0][1] + corner_points[1][0][1]

            left_coords = corner_points[0][1]; right_coords = corner_points[1][1]
            left_coords[0] += borders[0, 0]; right_coords[0] += borders[0, 0] # attaching to a pivot of a data

            if summary[1] > 0:
                if summary[0] / summary[1] >= threshold:
                    if not prev_state:
                        vertebraes += [left_coords, right_coords]
                        prev_state = True
                else:
                    if prev_state:
                        vertebraes += [left_coords, right_coords]
                        prev_state = False
            else:
                if prev_state:
                    vertebraes += [left_coords, right_coords]
                    prev_state = False

    if len(vertebraes) % 4 != 0:
        vertebraes = [*vertebraes, np.array([borders[1, 0], borders[1, 1]], dtype=np.float32), np.array([borders[1, 0], borders[0, 1]], dtype=np.float32)]

    if len(vertebraes) >= 4:
        vertebraes[0] = np.array([borders[0, 0], borders[0, 1]], dtype=np.float32)
        vertebraes[1] = np.array([borders[0, 0], borders[1, 1]], dtype=np.float32)
        vertebraes[-2] = np.array([borders[1, 0], borders[0, 1]], dtype=np.float32)
        vertebraes[-1] = np.array([borders[1, 0], borders[1, 1]], dtype=np.float32)

    return np.array(vertebraes, dtype=np.float32)[:96]

def _compute_div_lines_metric_and_points(pixel_array: np.ndarray[np.float32], borders: np.ndarray[np.int32], y: tuple[np.ndarray[np.float32], np.ndarray[np.float32], np.ndarray[np.float32]], threshold: float) -> tuple[float, np.ndarray[np.float32]]:
    """
        Computes metric of dividing lines and returns metric with them

        Parameters:
        -----------
            pixel_array:
                2d array of pixels - spine image
            \n
            borders:
                Bounding rectangle of the spine - `[up-left, down-right]` coords
            \n
            y:
                Tuple of computed edges and their mean `(mean, left, right)`
            \n
            threshold:
                if `<= line_mean` - then it is vertebrae, gap otherwise
    """
    div_lines = _get_vertebraes_dividing_lines(pixel_array, borders, y, threshold)

    pixel_array_zero = np.zeros_like(pixel_array, dtype=np.float32)
    for i in range(0, len(div_lines), 4):
        cv2.fillConvexPoly(pixel_array_zero, np.array([*(div_lines[i: i + 2]), *(div_lines[i + 2: i + 4])], dtype=np.int32)[:, ::-1], 1.0)
    
    inter = np.multiply(pixel_array, pixel_array_zero)
    m = np.sum(inter) / (np.sum(pixel_array) + np.sum(pixel_array_zero) - np.sum(inter))

    return (1 - m, div_lines)

def get_mean_vertebraes_dividing_lines(pixel_array: np.ndarray[np.float32], borders: np.ndarray[np.int32], y: tuple[np.ndarray[np.float32], np.ndarray[np.float32], np.ndarray[np.float32]], precision: float) -> np.ndarray[np.float32]:
    """
        Based on data image (0 <= pixel <= 1) and (middle regression, left-edge, right-edge) computes dividing line for each vertebraes' pair

        Parameters:
        -----------
            pixel_array:
                2d array of pixels - spine image
            \n
            borders:
                Bounding rectangle of the spine - `[up-left, down-right]` coords
            \n
            y:
                Tuple of computed edges and their mean `(mean, left, right)`
            \n
            precision:
                Presicion of dividing line coefs
    """
    e_log = int(np.log10(precision)) * -1
    init = 0
    with Pool() as p:
        for t in range(1, e_log + 1):
            power = 10 ** -t
            gen = range(1, 10) if t == 1 else range(-9, 10)
            mdts = p.starmap(_compute_div_lines_metric_and_points, [(pixel_array, borders, y, init + t * power) for t in gen])

            min_mdt = min(zip(mdts, range(1, 10) if t == 1 else range(-9, 10)), key=lambda x: x[0][0])
            init += min_mdt[1] * power
    
    dividing_lines = min_mdt[0][1]
    return np.array([*(dividing_lines[: 2]), *np.concatenate([(dividing_lines[l: l + 2] + dividing_lines[l + 2: l + 4]) / 2 for l in range(2, dividing_lines.shape[0] - 4, 4)]), *(dividing_lines[-2: ])])

def __compute_cnt_point_weight(weights, ind, cnt):
    weights[ind % cnt.shape[0]] = np.sqrt(
        np.sum((cnt[ind - 1] - cnt[ind % cnt.shape[0]]) ** 2)) + np.sqrt(np.sum((cnt[ind % cnt.shape[0]] - cnt[(ind + 1) % cnt.shape[0]]) ** 2)) - np.sqrt(np.sum((cnt[ind - 1] - cnt[(ind + 1) % cnt.shape[0]]) ** 2)
    )

def compute_vertebraes_points(pixel_array: npt.NDArray[np.float32], mean_dividing_lines: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Based on data image (0 <= pixel <= 1) and dividing lines computes reference points' projections of each vertebrae

    Parameters:
    -----------
        pixel_array:
            2d array of pixels - spine image
        \n
        mean_dividing_lines:
            Dividing lines for the spine (obtained from `get_mean_vertebraes_dividing_lined`)
    """

    points = []

    for v in range(0, mean_dividing_lines.shape[0] - 2, 2):
        vertebrae_pixel_array = np.multiply(
            pixel_array,
            cv2.fillConvexPoly(np.zeros_like(pixel_array), np.array([*(mean_dividing_lines[v: v + 2]), *(mean_dividing_lines[v + 2: v + 4][::-1])], dtype=np.int32)[:, ::-1], 255)
        )

        _, tresh = cv2.threshold(vertebrae_pixel_array.astype(np.uint8), 127, 255, 0)

        cnt = np.squeeze(
            max(
                cv2.findContours(tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0], 
                key=cv2.contourArea
            )
        )
        cnt_cpy = np.copy(cnt)

        weights = np.zeros((cnt_cpy.shape[0], ), dtype=np.float32)
        for ind in range(cnt_cpy.shape[0]):
            __compute_cnt_point_weight(weights, ind, cnt_cpy)

        while cnt_cpy.shape[0] > 4:
            less_ind = np.argmin(weights)

            cnt_cpy = np.concatenate([cnt_cpy[:less_ind], cnt_cpy[less_ind + 1:]], axis=0)
            weights = np.concatenate([weights[:less_ind], weights[less_ind + 1:]], axis=0)

            __compute_cnt_point_weight(weights, less_ind - 1, cnt_cpy)
            __compute_cnt_point_weight(weights, less_ind, cnt_cpy)
        if cnt_cpy.shape[0] >= 4:
            points = [*points, *cnt_cpy[:, ::-1]]
    return np.array(points, dtype=np.float32)

def formating_prj_vertebraes_points(borders: np.ndarray[np.int32], points: np.ndarray[np.float32], y_m: np.ndarray[np.float32]) -> VERTEBRAES_R_POINTS_PRJ:
    """
        Formats points of projection related to Gladcov's work

        Parameters:
        -----------
            borders:
                Bounding rectangle of the spine - `[up-left, down-right]` coords
            \n
            points:
                Array of points for each vertebrae projection
            \n
            y_m: 
                Spine approximation central contour 
    """

    y = np.concatenate(
        [
            np.expand_dims(
                np.arange(0, y_m.shape[0], dtype=np.float32),
                axis=1
            ) + borders[0, 0].astype(np.float32),
            np.expand_dims(y_m, axis=1)
        ], 
        axis=1
    )

    y = np.expand_dims(y, axis=1)
    p = np.expand_dims(points, axis=0)

    distances_sq = np.sum(
        (np.concatenate([y for i in range(points.shape[0])], axis=1) - np.concatenate([p for i in range(y.shape[0])], axis=0)) ** 2,
        axis=2
    )
    
    prj = np.argmin(distances_sq, axis=0)

    new_points = []
    for i in range(0, prj.shape[0], 4):
        prj_p = prj[i: i + 4]
        part = points[i: i + 4]
        first_pair = sorted(prj[i: i + 4])[:2]
        for j in range(4):
            if (prj_p[j - 1] in first_pair) and (prj_p[j - 2] in first_pair):
                new_points.append([part[j + t] for t in range(0, -4, -1)])
                break
    
    return np.array(new_points, dtype=np.float32)

def append_vertebrae(borders: np.ndarray[np.int32], points: VERTEBRAES_R_POINTS_PRJ, vertebrae_index: int, y_m: np.ndarray[np.float32], start_x: int, stop_x: int, r: float) -> VERTEBRAES_R_POINTS_PRJ:
    """
        Appends vertebrae starting from start_x

        Parameters
        ----------
            borders
                Bounding rectangle of the spine - `[up-left, down-right]` coords
            \n
            points
                Array of points for each vertebrae projection
            \n
            vertebrae_index
                Where to insert
            \n
            y_m
                Spine approximation central contour 
            \n
            start_x
                Starting position (where top intersects with y_m)
            \n
            stop_x
                Intersects bottom with y_m
            \n
            r
                Radius of the caps (top and bottom)
    """
    y_m = np.append(y_m, [y_m[-1]])
    new_points = np.concatenate([points[:vertebrae_index], np.zeros((1, 4, 2), dtype=np.float32), points[vertebrae_index:]], axis=0)

    d = y_m[start_x + 1] - y_m[start_x]; d = d if d != 0 else gamma
    normal = -1 / d
    norm_v = np.array([1, normal] if normal > 0 else [-1, normal * -1], dtype=np.float32) * r / np.abs(normal)

    new_points[vertebrae_index][1] = np.array([start_x + borders[0, 0], y_m[start_x]], dtype=np.float32) + norm_v
    new_points[vertebrae_index][2] = np.array([start_x + borders[0, 0], y_m[start_x]], dtype=np.float32) - norm_v

    d = y_m[stop_x + 1] - y_m[stop_x]; d = d if d != 0 else gamma
    normal = -1 / d
    norm_v = np.array([1, normal] if normal > 0 else [-1, normal * -1], dtype=np.float32) * r / np.abs(normal)

    new_points[vertebrae_index][0] = np.array([stop_x + borders[0, 0], y_m[stop_x]], dtype=np.float32) + norm_v
    new_points[vertebrae_index][3] = np.array([stop_x + borders[0, 0], y_m[stop_x]], dtype=np.float32) - norm_v

    return new_points

def fullfill_vertebraes(borders: np.ndarray[np.int32], points: VERTEBRAES_R_POINTS_PRJ, y_m: np.ndarray[np.float32]) -> VERTEBRAES_R_POINTS_PRJ:
    """
        Fullfill large gaps with vertebraes

        Parameters
        -----------
            borders
                Bounding rectangle of the spine - `[up-left, down-right]` coords
            \n
            points
                Array of points for each vertebrae projection
            \n
            y_m
                Spine approximation central contour
    """
    new_points = np.copy(points)

    L = np.min([np.linalg.norm((points[i + 1][1] + points[i + 1][2]) - (points[i][0] + points[i][3])) for i in range(points.shape[0] - 1)]) / 2
    for vertebrae_i in range(23):
        if vertebrae_i + 1 >= new_points.shape[0]:
            break
        p_mean_down_plate = (new_points[vertebrae_i][0] + new_points[vertebrae_i][3]) / 2
        p_mean_up_plate = (new_points[vertebrae_i][1] + new_points[vertebrae_i][2]) / 2
        p_mean_up_plate_next = (new_points[vertebrae_i + 1][1] + new_points[vertebrae_i + 1][2]) / 2
        p_mean_down_plate_next = (new_points[vertebrae_i + 1][0] + new_points[vertebrae_i + 1][3]) / 2

        i_height = np.linalg.norm(p_mean_up_plate - p_mean_down_plate)
        i1_height = np.linalg.norm(p_mean_down_plate_next - p_mean_up_plate_next)
        i_gap = np.linalg.norm(p_mean_down_plate - p_mean_up_plate_next)

        r_b = np.linalg.norm(new_points[vertebrae_i][0] - new_points[vertebrae_i][3]) / 2
        r_t = np.linalg.norm(new_points[vertebrae_i + 1][1] - new_points[vertebrae_i + 1][2]) / 2
        
        n = min(int((i_gap - L) / ((i_height + i1_height) / 2 + L)), 24 - new_points.shape[0])
        if n <= 0: continue

        l_pass = l_new = (i_gap - n * i_height - (i1_height - i_height) * n * (n + 1) / 2 / (n + 1)) / (n + 1)
        x_start, x_stop = -1, int(p_mean_down_plate[0] - borders[0, 0])
        i = 1
        for x in range((p_mean_down_plate[0] - borders[0, 0]).astype(np.int32), (p_mean_up_plate_next[0] - borders[0, 0]).astype(np.int32)):
            if x_start == -1:
                if np.linalg.norm(
                    np.array([x, y_m[x]], dtype=np.float32) - np.array([x_stop, y_m[x_stop]], dtype=np.float32)
                ) >= l_pass:
                    x_start = x
                    l_pass = i_height + (i1_height - i_height) * i / (n + 1)
            else:
                if np.linalg.norm(
                    np.array([x, y_m[x]], dtype=np.float32) - np.array([x_start, y_m[x_start]], dtype=np.float32)
                ) >= l_pass:
                    new_points = append_vertebrae(
                        borders,
                        new_points, 
                        vertebrae_i + i, 
                        y_m, 
                        x_start, 
                        x,
                        r_b + (r_t - r_b) * i / (n + 1)
                    )
                    i += 1
                    x_start, x_stop = -1, x
                    l_pass = l_new
        L = l_new

    return new_points

def link_projections(points_side: VERTEBRAES_R_POINTS_PRJ, points_frontal: VERTEBRAES_R_POINTS_PRJ, borders_frontal: np.ndarray[np.int32], y_m_frontal: np.ndarray[np.float32]) -> VERTEBRAES_R_POINTS:
    """
        Links two projections (side and frontal) together returns new frontal\n
        (Supposing that side projection is full and frontal projection has at least one vertebrae)

        Parameters:
        -----------
            points_side:
                Reference points to side vertebraes
            \n
            points_frontal:
                Reference points to frontal vertebraes
            \n
            borders_frontal:
                Bounding rectangle of the spine - `[up-left, down-right]` coords
            \n
            y_m_frontal:
                Central regression of frontal projection
    """

    linked_frontal_points = []

    for side_p in points_side:
        center_x = np.sum(side_p, axis=0)[0] / 4
        for frontal_p in points_frontal:
            if (frontal_p[1] + frontal_p[2])[0] / 2 <= center_x <= (frontal_p[0] + frontal_p[3])[0] / 2:
                linked_frontal_points.append(frontal_p)
                break
        else:
            linked_frontal_points.append([])
    
    link_point_i = 0
    while link_point_i < len(linked_frontal_points):
        if len(linked_frontal_points[link_point_i]) != 0:
            if link_point_i - 1 >= 0 and len(linked_frontal_points[link_point_i - 1]) == 0:
                up_x = (points_side[link_point_i - 1][1] + points_side[link_point_i - 1][2])[0] / 2
                down_x = (points_side[link_point_i - 1][0] + points_side[link_point_i - 1][3])[0] / 2
                plate_h = np.sqrt(
                    np.sum(
                        (linked_frontal_points[link_point_i][1] - linked_frontal_points[link_point_i][2]) ** 2
                    )
                ) / 2

                new_vertebrae = []
                if not (borders_frontal[0, 0] <= up_x < borders_frontal[1, 0]):
                    d = (linked_frontal_points[link_point_i][0] + linked_frontal_points[link_point_i][3]) / 2 - (linked_frontal_points[link_point_i][1] + linked_frontal_points[link_point_i][2]) / 2
                    d /= d[0]
                    d[1] = d[1] if d[1] != 0 else gamma
                    normal = -1 / d[1]

                    pivot = np.array(
                        [
                            up_x, 
                            (linked_frontal_points[link_point_i][1] + linked_frontal_points[link_point_i][2])[1] / 2 + d[1] * (up_x - (linked_frontal_points[link_point_i][1] + linked_frontal_points[link_point_i][2])[0] / 2)
                        ], 
                        dtype=np.float32
                    )
                else:
                    d = y_m_frontal[int(up_x - borders_frontal[0, 0] + 1)] - y_m_frontal[int(up_x - borders_frontal[0, 0])]
                    d = d if d != 0 else gamma
                    normal = -1 / d

                    pivot = np.array(
                        [
                            up_x, 
                            y_m_frontal[int(up_x - borders_frontal[0, 0])]
                        ], 
                        dtype=np.float32
                    )
                
                delta = np.array(
                    [1.0, normal]
                ) * plate_h * (plates_front[link_point_i - 1][0] / plates_front[link_point_i][0]) / np.sqrt(1 + normal ** 2)
                if normal < 0:
                    new_vertebrae = [
                        pivot + delta,
                        pivot - delta
                    ]
                else:
                    new_vertebrae = [
                        pivot - delta,
                        pivot + delta
                    ]
                
                if not (borders_frontal[0, 0] <= down_x < borders_frontal[1, 0]):
                    d = (linked_frontal_points[link_point_i][0] + linked_frontal_points[link_point_i][3]) / 2 - (linked_frontal_points[link_point_i][1] + linked_frontal_points[link_point_i][2]) / 2
                    d /= d[0]
                    d[1] = d[1] if d[1] != 0 else gamma
                    normal = -1 / d[1]

                    pivot = np.array(
                        [
                            down_x, 
                            (linked_frontal_points[link_point_i][1] + linked_frontal_points[link_point_i][2])[1] / 2 + d[1] * (down_x - (linked_frontal_points[link_point_i][1] + linked_frontal_points[link_point_i][2])[0] / 2)
                        ], 
                        dtype=np.float32
                    )
                else:
                    d = y_m_frontal[int(down_x - borders_frontal[0, 0] + 1)] - y_m_frontal[int(down_x - borders_frontal[0, 0])]
                    d = d if d != 0 else gamma
                    normal = -1 / d

                    pivot = np.array(
                        [
                            down_x, 
                            y_m_frontal[int(down_x - borders_frontal[0, 0])]
                        ], 
                        dtype=np.float32
                    )
                
                delta = np.array(
                    [1.0, normal]
                ) * plate_h * (plates_front[link_point_i - 1][1] / plates_front[link_point_i][0]) / np.sqrt(1 + normal ** 2)
                if normal < 0:
                    new_vertebrae = [
                        pivot + delta,
                        *new_vertebrae,
                        pivot - delta
                    ]
                else:
                    new_vertebrae = [
                        pivot - delta,
                        *new_vertebrae,
                        pivot + delta
                    ]
                linked_frontal_points[link_point_i - 1] = np.array(new_vertebrae)
                link_point_i -= 1
                continue
            if link_point_i + 1 < len(linked_frontal_points) and len(linked_frontal_points[link_point_i + 1]) == 0:
                up_x = (points_side[link_point_i + 1][1] + points_side[link_point_i + 1][2])[0] / 2
                down_x = (points_side[link_point_i + 1][0] + points_side[link_point_i + 1][3])[0] / 2
                plate_h = np.sqrt(
                    np.sum(
                        (linked_frontal_points[link_point_i][0] - linked_frontal_points[link_point_i][3]) ** 2
                    )
                ) / 2

                new_vertebrae = []
                if not (borders_frontal[0, 0] <= up_x < borders_frontal[1, 0]):
                    d = (linked_frontal_points[link_point_i][0] + linked_frontal_points[link_point_i][3]) / 2 - (linked_frontal_points[link_point_i][1] + linked_frontal_points[link_point_i][2]) / 2
                    d /= d[0]
                    d[1] = d[1] if d[1] != 0 else gamma
                    normal = -1 / d[1]

                    pivot = np.array(
                        [
                            up_x, 
                            (linked_frontal_points[link_point_i][0] + linked_frontal_points[link_point_i][3])[1] / 2 + d[1] * (up_x - (linked_frontal_points[link_point_i][0] + linked_frontal_points[link_point_i][3])[0] / 2)
                        ], 
                        dtype=np.float32
                    )
                else:
                    d = y_m_frontal[int(up_x - borders_frontal[0, 0] + 1)] - y_m_frontal[int(up_x - borders_frontal[0, 0])]
                    d = d if d != 0 else gamma
                    normal = -1 / d

                    pivot = np.array(
                        [
                            up_x, 
                            y_m_frontal[int(up_x - borders_frontal[0, 0])]
                        ], 
                        dtype=np.float32
                    )
                
                delta = np.array(
                    [1.0, normal]
                ) * plate_h * (plates_front[link_point_i + 1][0] / plates_front[link_point_i][1]) / np.sqrt(1 + normal ** 2)
                if normal < 0:
                    new_vertebrae = [
                        pivot + delta,
                        pivot - delta
                    ]
                else:
                    new_vertebrae = [
                        pivot - delta,
                        pivot + delta
                    ]
                
                if not (borders_frontal[0, 0] <= down_x < borders_frontal[1, 0]):
                    d = (linked_frontal_points[link_point_i][0] + linked_frontal_points[link_point_i][3]) / 2 - (linked_frontal_points[link_point_i][1] + linked_frontal_points[link_point_i][2]) / 2
                    d /= d[0]
                    d[1] = d[1] if d[1] != 0 else gamma
                    normal = -1 / d[1]

                    pivot = np.array(
                        [
                            down_x, 
                            (linked_frontal_points[link_point_i][0] + linked_frontal_points[link_point_i][3])[1] / 2 + d[1] * (down_x - (linked_frontal_points[link_point_i][0] + linked_frontal_points[link_point_i][3])[0] / 2)
                        ], 
                        dtype=np.float32
                    )
                else:
                    d = y_m_frontal[int(down_x - borders_frontal[0, 0] + 1)] - y_m_frontal[int(down_x - borders_frontal[0, 0])]
                    d = d if d != 0 else gamma
                    normal = -1 / d

                    pivot = np.array(
                        [
                            down_x, 
                            y_m_frontal[int(down_x - borders_frontal[0, 0])]
                        ], 
                        dtype=np.float32
                    )
                
                delta = np.array(
                    [1.0, normal]
                ) * plate_h * (plates_front[link_point_i + 1][1] / plates_front[link_point_i][1]) / np.sqrt(1 + normal ** 2)
                if normal < 0:
                    new_vertebrae = [
                        pivot + delta,
                        *new_vertebrae,
                        pivot - delta
                    ]
                else:
                    new_vertebrae = [
                        pivot - delta,
                        *new_vertebrae,
                        pivot + delta
                    ]
                linked_frontal_points[link_point_i + 1] = np.array(new_vertebrae)
                link_point_i += 1
                continue
        link_point_i += 1
    
    return np.array(linked_frontal_points, dtype=np.float32)