import pydicom
from pathlib import Path
import os

from PIL import Image
import numpy as np

def rotate(arr: np.ndarray, angle_r) -> np.ndarray:
    new_img = np.zeros_like(arr)

    origin_y, origin_x = map(int, np.floor(np.array(arr.shape) / 2))

    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            new_coords = np.array([[np.cos(angle_r), -np.sin(angle_r)], [np.sin(angle_r), np.cos(angle_r)]]) @ np.array([[c - origin_x], [r - origin_y]])
            new_y, new_x = int(new_coords[1][0]) + origin_y, int(new_coords[0][0]) + origin_x
            if 0 <= new_y < arr.shape[0] and 0 <= new_x < arr.shape[1]:
                new_img[new_y, new_x] = arr[r, c]

    return new_img

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'Data/dicom'
Y_PATH = BASE_DIR / 'Data/filled'

list_d = os.listdir(DATA_PATH)
for x_path in list_d:
    name, res = x_path.split('.')
    
    dcm = pydicom.dcmread(DATA_PATH / x_path)

    for angle_r in [15, 20, 25, 30]:
        rotated_dcm_pixel_array = rotate(dcm.pixel_array, np.pi / angle_r)

        dcm.PixelData = rotated_dcm_pixel_array.tobytes()
        pydicom.dcmwrite(DATA_PATH / f'{name}_rotd{angle_r}.{res}', dcm)
        print(x_path, f'{angle_r} done')
    
    for angle_r in [-15, -20, -25, -30]:
        rotated_dcm_pixel_array = rotate(dcm.pixel_array, np.pi / angle_r)

        dcm.PixelData = rotated_dcm_pixel_array.tobytes()
        pydicom.dcmwrite(DATA_PATH / f'{name}_rotmd{angle_r * -1}.{res}', dcm)
        print(x_path, f'{angle_r} done')

list_d = os.listdir(Y_PATH)
for y_path in list_d:
    name, res = y_path.split('.')
    
    img = Image.open(Y_PATH / y_path)
    pixel_array = np.array(img)
    img.close()

    for angle_r in [15, 20, 25, 30]:
        rotated_pixel_array = rotate(pixel_array, np.pi / angle_r)

        data = Image.fromarray(rotated_pixel_array)
        data.save(Y_PATH / f'{name}_rotd{angle_r}.{res}')
        print(y_path, f'{angle_r} done')
    
    for angle_r in [-15, -20, -25, -30]:
        rotated_pixel_array = rotate(pixel_array, np.pi / angle_r)

        data = Image.fromarray(rotated_pixel_array)
        data.save(Y_PATH / f'{name}_rotmd{angle_r * -1}.{res}')
        print(y_path, f'{angle_r} done')