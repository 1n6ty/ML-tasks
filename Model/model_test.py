# Import Libs
import numpy as np

from random import randint
import cv2
import os
import pydicom
import json
import pickle
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence

# Init Global Variables
DATA_DIR = os.path.abspath('../DataSet')
RESULTS = os.path.abspath('../Results')

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--routes", type=str, required=True)
parser.add_argument("-w", "--weights", type=str, required=True)
args = parser.parse_args()

with open(os.path.join(DATA_DIR, args.routes), 'r') as f:
    FILE_DIRS = json.loads(f.read())

IMG_SHAPE = (1760, 768)
BATCH_SIZE = 4

WEIGHTS2LOAD = os.path.join(RESULTS, f'saved_weights/{args.weights}')

# Data Generator
class Data_train_generator(Sequence):
    def __init__(self, x_files_list: list, batch_size: int, new_image_size = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data = x_files_list
        self.batch_size = batch_size

        self.new_image_size = new_image_size

        self.l_d = len(self.data)

    def __len__(self):
        return int(np.ceil((self.l_d) / float(self.batch_size)))

    def __open_dcm_x(self, file_path):
        dcm = pydicom.dcmread(file_path)

        data = dcm.pixel_array.astype("float32") / 255

        if self.new_image_size != None:
            data = cv2.resize(data, self.new_image_size[::-1], interpolation=cv2.INTER_CUBIC)

        return np.reshape(data, (*self.new_image_size, 1))

    def __getitem__(self, index):
        batch_x = np.array(list(map(self.__open_dcm_x, self.data[index * self.batch_size: (index + 1) * self.batch_size])))

        return batch_x

data_gen = Data_train_generator(FILE_DIRS["dicom"], BATCH_SIZE, IMG_SHAPE)

from model_unet import make_unet2p

model_unet = make_unet2p((*IMG_SHAPE, 1), filters=[64, 128, 256, 512, 1024], deep_supervision=True)
model_unet.compile()
model_unet.load_weights(WEIGHTS2LOAD)

res = model_unet.predict(x=data_gen, batch_size=BATCH_SIZE)

np.save(os.path.join(RESULTS, 'model_tresult.npy'), res)