from pathlib import Path
import pydicom
import os

import seaborn as sns
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.engine.results import Results

import numpy as np
import cv2

DATA_DIR = Path(__file__).resolve().parent.parent / "Data/spine-segmentation/"
test_file_path: Path = DATA_DIR / "t_side.png"

if test_file_path.suffix == ".dcm":
    pixel_array: np.ndarray = pydicom.dcmread(test_file_path).pixel_array
else:
    pixel_array: np.ndarray = cv2.cvtColor(cv2.imread(test_file_path), cv2.COLOR_BGR2GRAY)

cv2.imwrite("tmp.png", pixel_array)

model: YOLO = YOLO(Path(__file__).resolve().parent / "weights/best.pt")
model_response: list[Results] = model.predict("tmp.png", save=False, show=False, show_boxes=False, conf=0.5)

if os.path.exists("tmp.png"):
    os.remove("tmp.png")

vertebraes: list[np.ndarray] = [np.array(v, dtype=np.int32) for v in model_response[0].masks.xy]
vertebrae_central_points: np.ndarray = np.array([np.average(v, axis=0) for v in vertebraes], dtype=np.int32)





def TSP_solve(points: np.ndarray, prefix: np.ndarray) -> np.ndarray:
    if points.shape[0] == prefix.shape[0]:
        return prefix
    
    points_cpy: np.ndarray = points.copy()

    distances: np.float32 = np.linalg.norm(points_cpy - points_cpy[prefix[-1]], axis=1)
    distances[prefix] = np.inf

    return TSP_solve(points, np.concatenate([prefix, np.array([np.argmin(distances)], dtype=np.int32)]))

indexes: np.ndarray = TSP_solve(vertebrae_central_points, prefix=np.array([np.argmax(vertebrae_central_points[:, 1])], dtype=np.int32))
vertebraes: list[np.ndarray] = [vertebraes[i] for i in indexes]
vertebrae_central_points: np.ndarray = vertebrae_central_points[indexes]











def get_signed_angle(v1: np.ndarray, v2: np.ndarray) -> np.float32:
    return np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], np.dot(v1, v2))

for i in range(vertebrae_central_points.shape[0]):
    if i + 1 < vertebrae_central_points.shape[0]:
        nearest_indexes: np.ndarray = np.argpartition(np.linalg.norm(reference_points[i] - vertebrae_central_points[i + 1], axis=1), 2)[:2]

        up: np.ndarray = reference_points[i][nearest_indexes]
        down: np.ndarray = np.delete(reference_points[i], nearest_indexes, axis=0)
    else:
        nearest_indexes: np.ndarray = np.argpartition(np.linalg.norm(reference_points[i] - vertebrae_central_points[i - 1], axis=1), 2)[:2]

        up: np.ndarray = np.delete(reference_points[i], nearest_indexes, axis=0)
        down: np.ndarray = reference_points[i][nearest_indexes]
    
    main_vec: np.ndarray = np.average(up, axis=0) - np.average(down, axis=0)
    reference_points[i] = reference_points[i][np.argsort([get_signed_angle(main_vec, reference_points[i][j] - np.average(down, axis=0)) for j in range(reference_points[i].shape[0])])]



vertebrae_heights: np.ndarray = np.array([np.linalg.norm(vertebrae_polynom_points[i][0] - vertebrae_polynom_points[i][1]) for i in range(reference_points.shape[0])], dtype=np.float32)



import scipy.integrate
def int_path_length(t: np.float32, c: np.ndarray, n: np.int32) -> np.float32:
    A: np.ndarray = np.array(
        [
            [0] + [i * (t ** (i - 1)) for i in range(1, n)] + [0 for i in range(n)],
            [0 for i in range(n)] + [0] + [i * (t ** (i - 1)) for i in range(1, n)],
        ], 
        dtype=np.float32
    )
    p: np.ndarray = np.squeeze(A @ c)
    return np.sqrt(p[0] ** 2 + p[1] ** 2)

vertebrae_plates_lengths: np.ndarray = np.array([[np.linalg.norm(reference_points[i][0] - reference_points[i][3]), np.linalg.norm(reference_points[i][1] - reference_points[i][2])] for i in range(vertebrae_central_points.shape[0])], dtype=np.float32)
vertebrae_polynom_points: np.ndarray = np.array([[(reference_points[i][0] + reference_points[i][3]) / 2, (reference_points[i][1] + reference_points[i][2]) / 2] for i in range(vertebrae_central_points.shape[0])], dtype=np.float32)

vertebrae_heights: np.ndarray = np.array([np.linalg.norm(vertebrae_polynom_points[i][0] - vertebrae_polynom_points[i][1]) for i in range(reference_points.shape[0])], dtype=np.float32)
median_vertebrae_height: np.float32 = np.median(vertebrae_heights)

v_ind = 0
while v_ind < reference_points.shape[0]:

    down_plate: np.ndarray = (reference_points[v_ind, 3] - reference_points[v_ind, 0]) / 2
    up_plate: np.ndarray = (reference_points[v_ind, 2] - reference_points[v_ind, 1]) / 2
    down_plate_norm: np.ndarray = down_plate / np.linalg.norm(down_plate)
    up_plate_norm: np.ndarray = up_plate / np.linalg.norm(up_plate)

    n: np.int32 = 4
    A_vand: np.ndarray = np.array(
        [
            [0 ** i for i in range(n)] + [0 for i in range(n)],
            [0 for i in range(n)] + [0 ** i for i in range(n)],
            [1 ** i for i in range(n)] + [0 for i in range(n)],
            [0 for i in range(n)] + [1 ** i for i in range(n)],
            [0] + [down_plate_norm[0]] + [0 for i in range(n - 2)] + [0] + [down_plate_norm[1]] + [0 for i in range(n - 2)],
            [0] + [up_plate_norm[0] * i for i in range(1, n)] + [0] + [up_plate_norm[1] * i for i in range(1, n)],
            [0, 0] + [2] + [0 for i in range(n - 3)] + [0 for i in range(n)],
            [0 for i in range(n)] + [0, 0] + [2] + [0 for i in range(n - 3)],
            [0, 0] + [2] + [i * (i - 1) for i in range(3, n)] + [0 for i in range(n)],
            [0 for i in range(n)] + [0, 0] + [2] + [i * (i - 1) for i in range(3, n)],
        ],
        dtype=np.float32
    )
    b: np.ndarray = np.array(
        [
            [vertebrae_polynom_points[v_ind, 0, 0]],
            [vertebrae_polynom_points[v_ind, 0, 1]],
            [vertebrae_polynom_points[v_ind, 1, 0]],
            [vertebrae_polynom_points[v_ind, 1, 1]],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
        ],
        dtype=np.float32
    )
    c: np.ndarray = np.linalg.solve(A_vand.T @ A_vand, A_vand.T @ b)

    path_length: np.float32 = scipy.integrate.quad(int_path_length, 0, 1, args=(c, n))[0]
    insert_vertebraes: np.float32 = path_length / median_vertebrae_height
    
    if v_ind != 0 and v_ind != reference_points.shape[0] - 1 and insert_vertebraes > 2: # TODO only for side projection / for frontal access for 0 and -1 indexes
        indexes: list[int] = [v_ind + v_ind_new for v_ind_new in range(np.int32(insert_vertebraes))]
            
        res: np.float32 = 1 / insert_vertebraes
        gap_t: np.float32 = res * (insert_vertebraes % 1) / (np.int32(insert_vertebraes) + 1)

        break_t: list[tuple[np.float32, np.float32]] = [((gap_t * i + res * i), (res * (i + 1) + gap_t * i)) for i in range(np.int32(insert_vertebraes))]
        norm_t: list[tuple[np.ndarray, np.float32]] = [((1 - t) * down_plate_norm + t * up_plate_norm, t) for t in np.average(break_t, axis=1)]
        
        new_ref_points: list = []
        for [[t1, t2], [norm, t]] in zip(break_t, norm_t):
            A_p: np.ndarray = np.array(
                [
                    [t ** i for i in range(n)] + [0 for i in range(n)],
                    [0 for i in range(n)] + [t ** i for i in range(n)]
                ],
                dtype=np.float32
            )
            start_point = np.squeeze(A_p @ c)

            center_vec = np.array([norm[1], -norm[0]])
            down_length_half = np.linalg.norm((1 - t1) * down_plate + t1 * up_plate)
            up_length_half = np.linalg.norm((1 - t2) * down_plate + t2 * up_plate)

            new_ref_points.append(
                [
                    start_point - center_vec * median_vertebrae_height / 2 - norm * down_length_half,
                    start_point + center_vec * median_vertebrae_height / 2 - norm * up_length_half,
                    start_point + center_vec * median_vertebrae_height / 2 + norm * up_length_half,
                    start_point - center_vec * median_vertebrae_height / 2 + norm * down_length_half,
                ]
            )

        reference_points = np.delete(reference_points, v_ind, axis=0)
        reference_points = np.insert(
            reference_points,
            [v_ind for i in indexes],
            new_ref_points,
            axis=0
        )

        vertebrae_polynom_points = np.delete(vertebrae_polynom_points, v_ind, axis=0)
        vertebrae_polynom_points = np.insert(
            vertebrae_polynom_points,
            [v_ind for i in indexes],
            [
                [
                    (reference_points[v_ind_new][0] + reference_points[v_ind_new][3]) / 2,
                    (reference_points[v_ind_new][1] + reference_points[v_ind_new][2]) / 2
                ]
                for v_ind_new in indexes
            ],
            axis=0
        )

        vertebrae_heights = np.delete(vertebrae_heights, v_ind, axis=0)
        vertebrae_heights = np.insert(
            vertebrae_heights,
            [v_ind for i in indexes],
            [
                np.linalg.norm(vertebrae_polynom_points[v_ind_new][0] - vertebrae_polynom_points[v_ind_new][1])
                for v_ind_new in indexes
            ],
            axis=0
        )

        vertebrae_plates_lengths = np.delete(vertebrae_plates_lengths, v_ind, axis=0)
        vertebrae_plates_lengths = np.insert(
            vertebrae_plates_lengths,
            [v_ind for i in indexes],
            [
                [np.linalg.norm(reference_points[v_ind_new][0] - reference_points[v_ind_new][3]), np.linalg.norm(reference_points[v_ind_new][1] - reference_points[v_ind_new][2])]
                for v_ind_new in indexes
            ],
            axis=0
        )

        v_ind += len(indexes) - 1
    
    if v_ind < reference_points.shape[0] - 1:
        comb_points: np.ndarray = np.concatenate([reference_points[v_ind], reference_points[v_ind + 1]])

        max_v = np.max(comb_points, axis=0)
        min_v = np.min(comb_points, axis=0)
        
        tmp_canvas_fst: np.ndarray = np.zeros((max_v - min_v)[::-1], dtype=np.uint8)
        tmp_canvas_sec: np.ndarray = np.zeros((max_v - min_v)[::-1], dtype=np.uint8)

        cv2.fillConvexPoly(tmp_canvas_fst, reference_points[v_ind] - min_v, 255, 1)
        cv2.fillConvexPoly(tmp_canvas_sec, reference_points[v_ind + 1] - min_v, 255, 1)

        down_plate: np.ndarray = (reference_points[v_ind, 3] - reference_points[v_ind, 0]) / 2
        up_plate: np.ndarray = (reference_points[v_ind + 1, 2] - reference_points[v_ind + 1, 1]) / 2
        down_plate_norm: np.ndarray = down_plate / np.linalg.norm(down_plate)
        up_plate_norm: np.ndarray = up_plate / np.linalg.norm(up_plate)

        n: np.int32 = 4
        A_vand: np.ndarray = np.array(
            [
                [0 ** i for i in range(n)] + [0 for i in range(n)],
                [0 for i in range(n)] + [0 ** i for i in range(n)],
                [1 ** i for i in range(n)] + [0 for i in range(n)],
                [0 for i in range(n)] + [1 ** i for i in range(n)],
                [0] + [down_plate_norm[0]] + [0 for i in range(n - 2)] + [0] + [down_plate_norm[1]] + [0 for i in range(n - 2)],
                [0] + [up_plate_norm[0] * i for i in range(1, n)] + [0] + [up_plate_norm[1] * i for i in range(1, n)],
                [0, 0] + [2] + [0 for i in range(n - 3)] + [0 for i in range(n)],
                [0 for i in range(n)] + [0, 0] + [2] + [0 for i in range(n - 3)],
                [0, 0] + [2] + [i * (i - 1) for i in range(3, n)] + [0 for i in range(n)],
                [0 for i in range(n)] + [0, 0] + [2] + [i * (i - 1) for i in range(3, n)],
            ],
            dtype=np.float32
        )
        b: np.ndarray = np.array(
            [
                [vertebrae_polynom_points[v_ind, 0, 0]],
                [vertebrae_polynom_points[v_ind, 0, 1]],
                [vertebrae_polynom_points[v_ind + 1, 1, 0]],
                [vertebrae_polynom_points[v_ind + 1, 1, 1]],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
            ],
            dtype=np.float32
        )
        c: np.ndarray = np.linalg.solve(A_vand.T @ A_vand, A_vand.T @ b)

        path_length: np.float32 = scipy.integrate.quad(int_path_length, 0, 1, args=(c, n))[0]

        # spine_conv: np.ndarray = np.zeros_like(pixel_array, dtype=np.uint8)
        # for v in vertebraes:
        #     cv2.fillConvexPoly(spine_conv, v, 255, 1)
        # for r in reference_points:
        #     cv2.polylines(spine_conv, [r], True, 200, 3)
        # for t in np.linspace(0, 1, 1000):
        #     A_p: np.ndarray = np.array(
        #         [
        #             [t ** i for i in range(n)] + [0 for i in range(n)],
        #             [0 for i in range(n)] + [t ** i for i in range(n)]
        #         ],
        #         dtype=np.float32
        #     )
        #     p = np.squeeze(A_p @ c)
        #     spine_conv[int(p[1]), int(p[0])] = 150
            
        # sns.heatmap(spine_conv)
        # plt.show()

        insert_vertebraes: np.float32 = path_length / median_vertebrae_height

        if cv2.countNonZero(cv2.bitwise_and(tmp_canvas_fst, tmp_canvas_sec)):
            indexes: list[int] = [v_ind + v_ind_new for v_ind_new in range(np.int32(insert_vertebraes))]
            
            res: np.float32 = 1 / insert_vertebraes
            gap_t: np.float32 = res * (insert_vertebraes % 1) / (np.int32(insert_vertebraes) + 1)

            break_t: list[tuple[np.float32, np.float32]] = [((gap_t * i + res * i), (res * (i + 1) + gap_t * i)) for i in range(np.int32(insert_vertebraes))]
            norm_t: list[tuple[np.ndarray, np.float32]] = [((1 - t) * down_plate_norm + t * up_plate_norm, t) for t in np.average(break_t, axis=1)]
            
            new_ref_points: list = []
            for [[t1, t2], [norm, t]] in zip(break_t, norm_t):
                A_p: np.ndarray = np.array(
                    [
                        [t ** i for i in range(n)] + [0 for i in range(n)],
                        [0 for i in range(n)] + [t ** i for i in range(n)]
                    ],
                    dtype=np.float32
                )
                start_point = np.squeeze(A_p @ c)

                center_vec = np.array([norm[1], -norm[0]])
                down_length_half = np.linalg.norm((1 - t1) * down_plate + t1 * up_plate)
                up_length_half = np.linalg.norm((1 - t2) * down_plate + t2 * up_plate)

                new_ref_points.append(
                    [
                        start_point - center_vec * median_vertebrae_height / 2 - norm * down_length_half,
                        start_point + center_vec * median_vertebrae_height / 2 - norm * up_length_half,
                        start_point + center_vec * median_vertebrae_height / 2 + norm * up_length_half,
                        start_point - center_vec * median_vertebrae_height / 2 + norm * down_length_half,
                    ]
                )

            reference_points = np.delete(reference_points, [v_ind, v_ind + 1], axis=0)
            reference_points = np.insert(
                reference_points,
                [v_ind for i in indexes],
                new_ref_points,
                axis=0
            )

            vertebrae_polynom_points = np.delete(vertebrae_polynom_points, [v_ind, v_ind + 1], axis=0)
            vertebrae_polynom_points = np.insert(
                vertebrae_polynom_points,
                [v_ind for i in indexes],
                [
                    [
                        (reference_points[v_ind_new][0] + reference_points[v_ind_new][3]) / 2,
                        (reference_points[v_ind_new][1] + reference_points[v_ind_new][2]) / 2
                    ]
                    for v_ind_new in indexes
                ],
                axis=0
            )

            vertebrae_heights = np.delete(vertebrae_heights, [v_ind, v_ind + 1], axis=0)
            vertebrae_heights = np.insert(
                vertebrae_heights,
                [v_ind for i in indexes],
                [
                    np.linalg.norm(vertebrae_polynom_points[v_ind_new][0] - vertebrae_polynom_points[v_ind_new][1])
                    for v_ind_new in indexes
                ],
                axis=0
            )

            vertebrae_plates_lengths = np.delete(vertebrae_plates_lengths, [v_ind, v_ind + 1], axis=0)
            vertebrae_plates_lengths = np.insert(
                vertebrae_plates_lengths,
                [v_ind for i in indexes],
                [
                    [np.linalg.norm(reference_points[v_ind_new][0] - reference_points[v_ind_new][3]), np.linalg.norm(reference_points[v_ind_new][1] - reference_points[v_ind_new][2])]
                    for v_ind_new in indexes
                ],
                axis=0
            )

            v_ind += len(indexes) - 1

    if v_ind < reference_points.shape[0] - 1:
        down_plate: np.ndarray = (reference_points[v_ind][2] - reference_points[v_ind][1]) / 2
        up_plate: np.ndarray = (reference_points[v_ind + 1][3] - reference_points[v_ind + 1][0]) / 2
        down_plate_norm: np.ndarray = down_plate / np.linalg.norm(down_plate)
        up_plate_norm: np.ndarray = up_plate / np.linalg.norm(up_plate)

        n: np.int32 = 4
        A_vand: np.ndarray = np.array(
            [
                [0 ** i for i in range(n)] + [0 for i in range(n)],
                [0 for i in range(n)] + [0 ** i for i in range(n)],
                [1 ** i for i in range(n)] + [0 for i in range(n)],
                [0 for i in range(n)] + [1 ** i for i in range(n)],
                [0] + [down_plate_norm[0]] + [0 for i in range(n - 2)] + [0] + [down_plate_norm[1]] + [0 for i in range(n - 2)],
                [0] + [up_plate_norm[0] * i for i in range(1, n)] + [0] + [up_plate_norm[1] * i for i in range(1, n)],
                [0, 0] + [2] + [0 for i in range(n - 3)] + [0 for i in range(n)],
                [0 for i in range(n)] + [0, 0] + [2] + [0 for i in range(n - 3)],
                [0, 0] + [2] + [i * (i - 1) for i in range(3, n)] + [0 for i in range(n)],
                [0 for i in range(n)] + [0, 0] + [2] + [i * (i - 1) for i in range(3, n)],
            ],
            dtype=np.float32
        )
        b: np.ndarray = np.array(
            [
                [vertebrae_polynom_points[v_ind, 1, 0]],
                [vertebrae_polynom_points[v_ind, 1, 1]],
                [vertebrae_polynom_points[v_ind +  1, 0, 0]],
                [vertebrae_polynom_points[v_ind +  1, 0, 1]],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
            ],
            dtype=np.float32
        )
        c: np.ndarray = np.linalg.solve(A_vand.T @ A_vand, A_vand.T @ b)

        path_length: np.float32 = scipy.integrate.quad(int_path_length, 0, 1, args=(c, n))[0]

        insert_vertebraes: np.float32 = path_length / median_vertebrae_height
        if insert_vertebraes > 1:
            indexes: list[int] = [v_ind + v_ind_new for v_ind_new in range(np.int32(insert_vertebraes))]
            
            res: np.float32 = 1 / insert_vertebraes
            gap_t: np.float32 = res * (insert_vertebraes % 1) / (np.int32(insert_vertebraes) + 1)

            break_t: list[tuple[np.float32, np.float32]] = [((gap_t * (i + 1) + res * i), (res * (i + 1) + gap_t * (i + 1))) for i in range(np.int32(insert_vertebraes))]
            norm_t: list[tuple[np.ndarray, np.float32]] = [((1 - t) * down_plate_norm + t * up_plate_norm, t) for t in np.average(break_t, axis=1)]
            
            new_ref_points: list = []
            for [[t1, t2], [norm, t]] in zip(break_t, norm_t):
                A_p: np.ndarray = np.array(
                    [
                        [t ** i for i in range(n)] + [0 for i in range(n)],
                        [0 for i in range(n)] + [t ** i for i in range(n)]
                    ],
                    dtype=np.float32
                )
                start_point = np.squeeze(A_p @ c)

                center_vec = np.array([norm[1], -norm[0]])
                down_length_half = np.linalg.norm((1 - t1) * down_plate + t1 * up_plate)
                up_length_half = np.linalg.norm((1 - t2) * down_plate + t2 * up_plate)

                new_ref_points.append(
                    [
                        start_point - center_vec * median_vertebrae_height / 2 - norm * down_length_half,
                        start_point + center_vec * median_vertebrae_height / 2 - norm * up_length_half,
                        start_point + center_vec * median_vertebrae_height / 2 + norm * up_length_half,
                        start_point - center_vec * median_vertebrae_height / 2 + norm * down_length_half,
                    ]
                )

            reference_points = np.insert(
                reference_points,
                [v_ind for i in indexes],
                new_ref_points,
                axis=0
            )

            vertebrae_polynom_points = np.insert(
                vertebrae_polynom_points,
                [v_ind for i in indexes],
                [
                    [
                        (reference_points[v_ind_new][0] + reference_points[v_ind_new][3]) / 2,
                        (reference_points[v_ind_new][1] + reference_points[v_ind_new][2]) / 2
                    ]
                    for v_ind_new in indexes
                ],
                axis=0
            )

            vertebrae_heights = np.insert(
                vertebrae_heights,
                [v_ind for i in indexes],
                [
                    np.linalg.norm(vertebrae_polynom_points[v_ind_new][0] - vertebrae_polynom_points[v_ind_new][1])
                    for v_ind_new in indexes
                ],
                axis=0
            )

            vertebrae_plates_lengths = np.insert(
                vertebrae_plates_lengths,
                [v_ind for i in indexes],
                [
                    [np.linalg.norm(reference_points[v_ind_new][0] - reference_points[v_ind_new][3]), np.linalg.norm(reference_points[v_ind_new][1] - reference_points[v_ind_new][2])]
                    for v_ind_new in indexes
                ],
                axis=0
            )

            v_ind += len(indexes)

    v_ind += 1



n = vertebrae_central_points.shape[0]
A_vand: np.ndarray = np.vander(np.linspace(0, 1, vertebrae_polynom_points.shape[0]), n, increasing=True).astype(dtype=np.complex128)
c = np.linalg.solve(A_vand.conj().T @ A_vand, A_vand.conj().T @ (vertebrae_polynom_points[:,:, 0] + vertebrae_polynom_points[:,:, 1] * 1j))







spine_conv: np.ndarray = np.zeros_like(pixel_array, dtype=np.uint8)

for v in vertebraes:
    cv2.fillConvexPoly(spine_conv, v, 255, 1)
for r in reference_points:
    cv2.polylines(spine_conv, [r], True, 200, 3)

sns.heatmap(spine_conv)
plt.show()