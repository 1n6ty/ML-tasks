import numpy as np
import cv2
from itertools import product

get_ratio = lambda x, x_min, x_max: (x - x_min) / (x_max - x_min)

def get_edgefunc_coef(data, border_coords, n = 10):
    y_l, y_r = [], []
    x_l, x_r = [], []
    A_l, A_r = [], []
    for i in range(border_coords[0][1], border_coords[1][1] + 1):
        # left hand
        for j in range(border_coords[0][0], border_coords[1][0]):
            if data[i, j]: 
                y_l.append(j)
                x_l.append(i)

                x = get_ratio(i, border_coords[0][1], border_coords[1][1])
                A_l.append([(x ** t) * ((1 - x) ** (n - t)) for t in range(0, n + 1)])
                break
        
        # right hand
        for j in range(border_coords[1][0], border_coords[0][0] - 1, -1):
            if data[i, j]: 
                y_r.append(j)
                x_r.append(i)

                x = get_ratio(i, border_coords[0][1], border_coords[1][1])
                A_r.append([(x ** t) * ((1 - x) ** (n - t)) for t in range(0, n + 1)])
                break

    y_l, y_r = np.array(y_l, dtype=np.float32), np.array(y_r, dtype=np.float32)
    A_l, A_r = np.array(A_l, dtype=np.float32), np.array(A_r, dtype=np.float32)

    c_l, c_r = np.dot(np.linalg.inv(np.dot(A_l.T, A_l)), np.dot(A_l.T, y_l.T)), np.dot(np.linalg.inv(np.dot(A_r.T, A_r)), np.dot(A_r.T, y_r.T))
    return c_l, c_r

def search_for_borders(data):
    IMG_SHAPE = data.shape
    x1, y1, x2, y2 = 0, 0, 0, 0

    # search for y1
    break_flag = False
    for i in range(IMG_SHAPE[0]):
        for j in range(IMG_SHAPE[1]):
            if data[i, j]:
                y1 = i
                break_flag = True
                break
        if break_flag: break

    # search for y2
    break_flag = False
    for i in range(IMG_SHAPE[0] - 1, 0, -1):
        for j in range(IMG_SHAPE[1]):
            if data[i, j]:
                y2 = i
                break_flag = True
                break
        if break_flag: break

    # search for x1
    break_flag = False
    for i in range(IMG_SHAPE[1]):
        for j in range(IMG_SHAPE[0]):
            if data[j, i]:
                x1 = i
                break_flag = True
                break
        if break_flag: break

    # search for x2
    break_flag = False
    for i in range(IMG_SHAPE[1] - 1, 0, -1):
        for j in range(IMG_SHAPE[0]):
            if data[j, i]:
                x2 = i
                break_flag = True
                break
        if break_flag: break
    
    return ((x1, y1), (x2, y2))

def normalOverEdge(border_coords, x, y, y_l, y_r, gamma = 1e6):
    delta = y[x - border_coords[0][1] + 1] - y[x - border_coords[0][1] - 1]
    f_normal = -2 / delta if delta != 0 else gamma

    denominator = max([1, abs(f_normal)])
    x_c, y_c = 1, f_normal
    x_c, y_c = x_c / denominator, y_c / denominator
    
    reach_right_edge, reach_left_edge = False, False

    cur_pos, cur_neg = [x, int(y[x - border_coords[0][1]])], [x, int(y[x - border_coords[0][1]])]
    while (not (reach_right_edge and reach_left_edge)):
        if not reach_left_edge:
            cur_pos = [cur_pos[0] + x_c, cur_pos[1] + y_c]
            reach_left_edge = not ((border_coords[0][1] <= int(cur_pos[0]) <= border_coords[1][1]) and y_l[int(cur_pos[0]) - border_coords[0][1]] <= int(cur_pos[1]) and y_r[int(cur_pos[0]) - border_coords[0][1]] >= int(cur_pos[1]))
        
        if not reach_right_edge:
            cur_neg = [cur_neg[0] - x_c, cur_neg[1] - y_c]
            reach_right_edge = not ((border_coords[0][1] <= int(cur_neg[0]) <= border_coords[1][1]) and y_r[int(cur_neg[0]) - border_coords[0][1]] >= int(cur_neg[1]) and y_l[int(cur_neg[0]) - border_coords[0][1]] <= int(cur_neg[1]))
    
    return sorted([cur_pos, cur_neg], key=lambda x: x[1])

def get_vertebras_corners(data, border_coords, y, y_l, y_r, threshold, gamma = 1e6):
    vertebras_corners = []
    prev_edge = True
    prev_coords = []
    for i in range(border_coords[0][1] + 1, border_coords[1][1]):
        delta = y[i - border_coords[0][1] + 1] - y[i - border_coords[0][1] - 1]
        f_normal = -2 / delta if delta != 0 else gamma

        denominator = max([1, abs(f_normal)])
        x_c, y_c = 1, f_normal
        x_c, y_c = x_c / denominator, y_c / denominator
        
        reach_right_edge, reach_left_edge = False, False

        s, it = data[i, int(y[i - border_coords[0][1]])], 1
        cur_pos, cur_neg = [i, int(y[i - border_coords[0][1]])], [i, int(y[i - border_coords[0][1]])]
        while (not (reach_right_edge and reach_left_edge)):
            if not reach_left_edge:
                cur_pos = [cur_pos[0] + x_c, cur_pos[1] + y_c]
                if (border_coords[0][1] <= int(cur_pos[0]) <= border_coords[1][1]) and y_l[int(cur_pos[0]) - border_coords[0][1]] <= int(cur_pos[1]) and y_r[int(cur_pos[0]) - border_coords[0][1]] >= int(cur_pos[1]):
                    s += data[int(cur_pos[0]), int(cur_pos[1])]
                    it += 1
                else:
                    reach_left_edge = True
            
            if not reach_right_edge:
                cur_neg = [cur_neg[0] - x_c, cur_neg[1] - y_c]
                if (border_coords[0][1] <= int(cur_neg[0]) <= border_coords[1][1]) and y_r[int(cur_neg[0]) - border_coords[0][1]] >= int(cur_neg[1]) and y_l[int(cur_neg[0]) - border_coords[0][1]] <= int(cur_neg[1]):
                    s += data[int(cur_neg[0]), int(cur_neg[1])]
                    it += 1
                else:
                    reach_right_edge = True
        s /= it
        cur_coords = sorted([cur_pos, cur_neg], key=lambda x: x[1])

        if ((i == border_coords[0][1] + 1) or (i == border_coords[1][1] - 1) or (not prev_edge)) and s >= threshold:
            vertebras_corners = [*vertebras_corners, *cur_coords]
        elif prev_edge and s < threshold:
            vertebras_corners = [*vertebras_corners, *prev_coords]
        
        prev_coords = cur_coords
        if s >= threshold:
            prev_edge = True
        else:
            prev_edge = False
    
    return vertebras_corners

def approximateYbezie_lcr(border_coords, edge_coef, n):
    avg_c = np.float32((edge_coef[0] + edge_coef[1]) / 2)
    A = [
        [
            (get_ratio(i, border_coords[0][1], border_coords[1][1]) ** t) * ((1 - get_ratio(i, border_coords[0][1], border_coords[1][1])) ** (n - t)) 
            for t in range(0, n + 1)
        ]
        for i in range(border_coords[0][1], border_coords[1][1] + 1)
    ]
    return [np.dot(A, avg_c.T), np.dot(A, edge_coef[0]), np.dot(A, edge_coef[1])]

def adjast_vertebras(border_coords, vertebras_corners, y, y_l, y_r, gamma = 1e6):
    corners_l = len(vertebras_corners)

    if (corners_l % 4) == 0:
        vertebra_gap_lengths = [
            np.sqrt((vertebras_corners[i][0] - vertebras_corners[i + 2][0]) ** 2 + (vertebras_corners[i][1] - vertebras_corners[i + 2][1]) ** 2)
        for i in range(corners_l - 2)
        ]
        
        vertebra_lengths, gap_lengths = [], []
        for i in range(0, len(vertebra_gap_lengths), 4):
            vertebra_lengths = [*vertebra_lengths, sum(vertebra_gap_lengths[i: i + 2]) / 2]
            gap_lengths = [*gap_lengths, sum(vertebra_gap_lengths[i + 2: i + 4]) / 2]

        gap_lengths = sorted(gap_lengths)
        length_vertebra_l, length_gap_l = len(vertebra_lengths), len(gap_lengths)

        avg_vertebra_length = vertebra_lengths[length_vertebra_l // 2] if (length_vertebra_l % 2) else (vertebra_lengths[length_vertebra_l // 2] + vertebra_lengths[length_vertebra_l // 2 - 1]) / 2
        avg_gap_length = gap_lengths[length_gap_l // 2] if (length_gap_l % 2) else (gap_lengths[length_gap_l // 2] + gap_lengths[length_gap_l // 2 - 1]) / 2

        i = 0
        v_c = 0
        while i < corners_l:
            d_vertebra_x = (vertebras_corners[i + 2][0] - vertebras_corners[i][0] + vertebras_corners[i + 3][0] - vertebras_corners[i + 1][0]) / 2
            d_vertebra_y = (vertebras_corners[i + 2][1] - vertebras_corners[i][1] + vertebras_corners[i + 3][1] - vertebras_corners[i + 1][1]) / 2

            l_vertebra = np.sqrt(d_vertebra_x ** 2 + d_vertebra_y ** 2)

            ratio = avg_vertebra_length / l_vertebra

            if 0.5 >= ratio:
                x = int(vertebras_corners[i][0] + avg_vertebra_length)
                vertebras_corners = [
                    *(vertebras_corners[: i + 2]),
                    *normalOverEdge(border_coords, x, y, y_l, y_r, gamma),
                    *(vertebras_corners[i + 2: ])
                ]

                x = int(vertebras_corners[i][0] + avg_vertebra_length + avg_gap_length)
                vertebras_corners = [
                    *(vertebras_corners[: i + 4]),
                    *normalOverEdge(border_coords, x, y, y_l, y_r, gamma),
                    *(vertebras_corners[i + 4: ])
                ]

                vertebra_lengths = [
                    *(vertebra_lengths[: v_c]),
                    avg_vertebra_length,
                    vertebra_lengths[v_c] - avg_gap_length - avg_vertebra_length,
                    *(vertebra_lengths[v_c + 1:])
                ]

                corners_l += 4
            elif 2 <= ratio:
                if i == len(vertebras_corners) - 4:
                    i -= 4

                    vertebra_lengths = [
                        *(vertebra_lengths[: v_c - 1]),
                        vertebra_lengths[v_c - 1] + vertebra_lengths[v_c]
                    ]
                    v_c -= 1
                else:
                    vertebra_lengths = [
                        *(vertebra_lengths[: v_c]),
                        vertebra_lengths[v_c + 1] + vertebra_lengths[v_c]
                    ]
                
                vertebras_corners = [
                    *(vertebras_corners[0: i + 2]),
                    *(vertebras_corners[i + 6: ])
                ]

                corners_l -= 4
                i -= 4
            
            i += 4

            avg_vertebra_length = vertebra_lengths[v_c]
            v_c += 1
        return vertebras_corners
    
def MSE(vertebras_corners, vertebras_corners_true, gamma = 1e6):
    len_vertebras_corners, len_vertebras_corners_true = len(vertebras_corners), len(vertebras_corners_true)
    min_len = min(len_vertebras_corners, len_vertebras_corners_true)
    return np.sum([
        np.sqrt((vertebras_corners[i][0] - vertebras_corners_true[i][0]) ** 2 + (vertebras_corners[i][1] - vertebras_corners_true[i][1]) ** 2) 
    for i in range(min_len)
    ]) + np.sum([
        gamma 
    for i in range(min_len, len_vertebras_corners)
    ]) + np.sum([
        gamma
    for i in range(min_len, len_vertebras_corners_true)
    ])

def rotate(arr, angle_r):
    new_img = np.zeros_like(arr)

    origin_c = list(map(int, np.floor(np.array([arr.shape[1], arr.shape[0]]) / 2)))
    init_coords = list(product([c for c in range(arr.shape[1])], [r for r in range(arr.shape[0])]))
    
    origin = np.array([origin_c for c in init_coords])

    new_coords = (np.array([
                    [np.cos(angle_r), -np.sin(angle_r)], 
                    [np.sin(angle_r), np.cos(angle_r)]]) @ ((np.array(init_coords) - origin).T)) + (origin.T)
    new_coords = new_coords.astype(dtype=np.int32)
    
    for i in range(len(init_coords)):
        c, r = init_coords[i]
        if 0 <= new_coords[1][i] < arr.shape[0] and 0 <= new_coords[0][i] < arr.shape[1]:
            new_img[new_coords[1][i], new_coords[0][i]] = arr[r, c]
    
    return new_img

def img2niito_front(vertebras_corners, pixel_spacing):
    init_v = [vertebras_corners[-4][0] * -1, vertebras_corners[-4][1] * -1]

    for i in range(len(vertebras_corners)):
        vertebras_corners[i] = [(vertebras_corners[-4][1] + init_v[1]) * pixel_spacing, (vertebras_corners[-4][0] + init_v[0]) * -pixel_spacing]
    
    vertebras_corners = vertebras_corners[-96:] # for case where only 24 vertebras are seen

    obj = {}

    i = 95
    for name in ["S1", "L5", "L4", "L3", "L2", "L1",
                 "Th12", "Th11", "Th10", "Th9", "Th8", "Th7", "Th6", "Th5", "Th4", "Th3", "Th2", "Th1",
                 "C7", "C6", "C5", "C4", "C3", "C2"]:
        obj[name] = [
            [0.1, *(vertebras_corners[i - 1][::-1])],
            [0.1, *(vertebras_corners[i - 3][::-1])],
            [0.1, *(vertebras_corners[i - 2][::-1])],
            [0.1, *(vertebras_corners[i][::-1])]
        ]

        i -= 4
    
    return obj

def img2niito_side(vertebras_corners, pixel_spacing):
    init_v = [vertebras_corners[-4][0] * -1, vertebras_corners[-4][1] * -1]

    for i in range(len(vertebras_corners)):
        vertebras_corners[i] = [(vertebras_corners[-4][1] + init_v[1]) * pixel_spacing, (vertebras_corners[-4][0] + init_v[0]) * -pixel_spacing]
    
    vertebras_corners = vertebras_corners[-96:] # for case where only 24 vertebras are seen

    obj = {}

    i = 95
    for name in ["S1", "L5", "L4", "L3", "L2", "L1",
                 "Th12", "Th11", "Th10", "Th9", "Th8", "Th7", "Th6", "Th5", "Th4", "Th3", "Th2", "Th1",
                 "C7", "C6", "C5", "C4", "C3", "C2"]:
        obj[name] = [
            [*(vertebras_corners[i - 1]), 0.1],
            [*(vertebras_corners[i - 3]), 0.1],
            [*(vertebras_corners[i - 2]), 0.1],
            [*(vertebras_corners[i]), 0.1]
        ]

        i -= 4
    
    return obj