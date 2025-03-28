from spine_segmentation.segmentation.utils import open_dcm_prjs, open_png_prjs

import os
import numpy as np
from itertools import product
from pathlib import Path
import cv2

import pydicom

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

dicom_base_dir = Path(__file__).parent.parent.resolve() / "Data/spine-segmentation/dicom"
png_base_dir = Path(__file__).parent.parent.resolve() / "Data/spine-segmentation/filled"

new_routes = {
    "dicom": {
        "side": [],
        "frontal": []
    },
    "converted": {
        "side": [],
        "frontal": []
    }
}

for [dc, png] in zip(sorted(os.listdir(dicom_base_dir)), sorted(os.listdir(png_base_dir))):
    side_png = cv2.imread(png_base_dir / png)
    lf, hf = [(10, 10, 10), (256, 256, 256)]
    side_png = cv2.inRange(side_png, lf, hf)

    side_dc = pydicom.pixel_array(dicom_base_dir / dc)

    dc_obj = pydicom.dcmread(dicom_base_dir / dc)

    new_routes["converted"]["side"].append(str(png_base_dir / png))
    new_routes["converted"]["frontal"].append(str(png_base_dir / png))

    new_routes["dicom"]["side"].append(str(dicom_base_dir / dc))
    new_routes["dicom"]["frontal"].append(str(dicom_base_dir / dc))
    for a in [-7, -5, -3, 3, 5, 7]:
        new_dc = rotate(side_dc, a * np.pi / 180)
        new_png = rotate(side_png, a * np.pi / 180)

        dc_obj.PixelData = new_dc.tobytes()
        
        pydicom.dcmwrite(str(dicom_base_dir / f"{dc.split('.')[0]}_{str(a)}.dcm"), dc_obj)
        cv2.imwrite(str(png_base_dir / f"{png.split('.')[0]}_{str(a)}.png"), new_png)
        print(f"done {dc} and {png} with {a}-angle")

        new_routes["converted"]["side"].append(str(png_base_dir / f"{png.split('.')[0]}_{str(a)}.png"))
        new_routes["converted"]["frontal"].append(str(png_base_dir / f"{png.split('.')[0]}_{str(a)}.png"))

        new_routes["dicom"]["side"].append(str(dicom_base_dir / f"{dc.split('.')[0]}_{str(a)}.dcm"))
        new_routes["dicom"]["frontal"].append(str(dicom_base_dir / f"{dc.split('.')[0]}_{str(a)}.dcm"))

import json

with open(Path(__file__).parent.parent.resolve() / 'Data/routes.json', 'w') as f:
    json.dump(new_routes, f)

print('file has been written')