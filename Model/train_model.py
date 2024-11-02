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
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy, Dice

# Init Global Variables
DATA_DIR = os.path.abspath('../DataSet')
RESULTS = os.path.abspath('../Results')

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str, required=True)
args = parser.parse_args()

MODE = args.mode

with open(os.path.join(DATA_DIR, f'routes.json'), 'r') as f:
    FILE_DIRS = json.loads(f.read())

IMG_SHAPE = (1760, 768)
VALIDATION_NUM = 100
BATCH_SIZE = 8
EPOCHS = 200

WEIGHTS2LOAD = os.path.join(RESULTS, 'saved_weights/init_unetpp_' + MODE + '.weights.h5')

# Data Generator
class Data_generator(Sequence):
    """
        Generator of images for spine-segmentation problem

        Output `__get_item__`
        ---------------------
        list `[batch_x, {"output_1": batch_y, "output_2": batch_y, "output_3": batch_y, "output_4": batch_y}]`

        if mode == `side | frontal` then `batch_shape` = `(batch_size, height, width, channels)`\n
        if mode == `both` then `batch_shape` = `(batch_size, 2, height, width, channels)`\n
    """
    def __init__(self, x_files_list: dict, y_files_list: dict, batch_size: int, update_func: Callable, mode: Literal["side", "frontal", "both"], new_image_size = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data = x_files_list
        self.labels = y_files_list
        self.batch_size = batch_size

        self.new_image_size = new_image_size
        self.mode = mode

        self.l_d = min(len(self.data["side"]), len(self.data["frontal"]))
        self.update_func = update_func

    def __len__(self):
        return int(np.ceil((self.l_d) / float(self.batch_size)))

    def __open_png_y(self, file_path_side, file_path_frontal):
        img_side = cv2.imread(file_path_side)
        img_frontal = cv2.imread(file_path_frontal)

        if img_side.shape[0] < img_frontal.shape[0]:
            d = int((img_frontal.shape[0] - img_side.shape[0]) / 2)
            img_side = cv2.copyMakeBorder(img_side, d, d, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
        else:
            d = int((img_side.shape[0] - img_frontal.shape[0]) / 2)
            img_frontal = cv2.copyMakeBorder(img_frontal, d, d, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))

        if img_side.shape[1] < img_frontal.shape[1]:
            d = int((img_frontal.shape[1] - img_side.shape[1]) / 2)
            img_side = cv2.copyMakeBorder(img_side, 0, 0, d, d, cv2.BORDER_CONSTANT, (0, 0, 0))
        else:
            d = int((img_side.shape[1] - img_frontal.shape[1]) / 2)
            img_frontal = cv2.copyMakeBorder(img_frontal, 0, 0, d, d, cv2.BORDER_CONSTANT, (0, 0, 0))

        img_side = cv2.inRange(img_side, (0, 0, 210), (40, 40, 256))
        img_frontal = cv2.inRange(img_frontal, (0, 0, 210), (40, 40, 256))
        
        if self.new_image_size != None:
            img_side = cv2.resize(img_side, self.new_image_size[::-1], interpolation=cv2.INTER_CUBIC)
            img_frontal = cv2.resize(img_frontal, self.new_image_size[::-1], interpolation=cv2.INTER_CUBIC)

        return (img_side.astype(np.float32) / np.max(img_side), img_frontal.astype(np.float32) / np.max(img_frontal))

    def __open_dcm_x(self, file_path_side, file_path_frontal):
        img_side = pydicom.dcmread(file_path_side).pixel_array
        img_frontal = pydicom.dcmread(file_path_frontal).pixel_array

        if img_side.shape[0] < img_frontal.shape[0]:
            d = int((img_frontal.shape[0] - img_side.shape[0]) / 2)
            img_side = cv2.copyMakeBorder(img_side, d, d, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
        else:
            d = int((img_side.shape[0] - img_frontal.shape[0]) / 2)
            img_frontal = cv2.copyMakeBorder(img_frontal, d, d, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))

        if img_side.shape[1] < img_frontal.shape[1]:
            d = int((img_frontal.shape[1] - img_side.shape[1]) / 2)
            img_side = cv2.copyMakeBorder(img_side, 0, 0, d, d, cv2.BORDER_CONSTANT, (0, 0, 0))
        else:
            d = int((img_side.shape[1] - img_frontal.shape[1]) / 2)
            img_frontal = cv2.copyMakeBorder(img_frontal, 0, 0, d, d, cv2.BORDER_CONSTANT, (0, 0, 0))
        
        if self.new_image_size != None:
            img_side = cv2.resize(img_side, self.new_image_size[::-1], interpolation=cv2.INTER_CUBIC)
            img_frontal = cv2.resize(img_frontal, self.new_image_size[::-1], interpolation=cv2.INTER_CUBIC)

        return (img_side.astype(np.float32) / np.max(img_side), img_frontal.astype(np.float32) / np.max(img_frontal))

    def __getitem__(self, index):
        batch_x, batch_y = [], []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            if i < len(self.data["side"]):
                x = self.__open_dcm_x(self.data["side"][i], self.data["frontal"][i])
                y = self.__open_png_y(self.labels["side"][i], self.labels["frontal"][i])
            if self.mode == "both":
                batch_x.append(x)
                batch_y.append(y)
            else:
                mode_ind = 0 if self.mode == "side" else 1
                batch_x.append(x[mode_ind])
                batch_y.append(y[mode_ind])
        
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        return batch_x, {"output_1": batch_y, "output_2": batch_y, "output_3": batch_y, "output_4": batch_y}

    def on_epoch_end(self):
        if self.update_func != None:
            self.update_func()

class CrossValidation:
    """
        CrossValidation class for Data_generator
    """
    def __init__(self, x_files_list: dict, y_files_list: dict, batch_size: int, val_size: int, mode: Literal["side", "frontal", "both"], new_image_size = None) -> None:
        self.data = x_files_list
        self.labels = y_files_list
        self.batch_size = batch_size

        self.new_image_size = new_image_size
        self.mode = mode

        self.l_d = min(len(self.data["side"]), len(self.data["frontal"]))
        self.val_size = val_size
        self.update_count = 0

        x, y = self.__get_data()

        self.data_gen = Data_generator(x[0], y[0], self.batch_size, self.__update_gen, mode, self.new_image_size)
        self.val_gen = Data_generator(x[1], y[1], self.batch_size, self.__update_gen, mode, self.new_image_size)

    def __get_data(self):
        return ([
            {
                "side": self.data["side"][:-self.val_size],
                "frontal": self.data["frontal"][:-self.val_size]
            },
            {
                "side": self.data["side"][-self.val_size:],
                "frontal": self.data["frontal"][-self.val_size:]
            }
        ],
        [
            {
                "side": self.labels["side"][:-self.val_size],
                "frontal": self.labels["frontal"][:-self.val_size]
            },
            {
                "side": self.labels["side"][-self.val_size:],
                "frontal": self.labels["frontal"][-self.val_size:]
            }
        ])
    
    def __update_gen(self):
        if self.update_count < 2:
            self.update_count += 1
        else:
            self.update_count = 0

            for _ in range(self.l_d):
                ind_from, ind_to = randint(0, self.l_d - 1), randint(0, self.l_d - 1)

                self.data["side"][ind_from], self.data["side"][ind_to] = self.data["side"][ind_to], self.data["side"][ind_from]
                self.data["frontal"][ind_from], self.data["frontal"][ind_to] = self.data["frontal"][ind_to], self.data["frontal"][ind_from]

                self.labels["side"][ind_from], self.labels["side"][ind_to] = self.labels["side"][ind_to], self.labels["side"][ind_from]
                self.labels["frontal"][ind_from], self.labels["frontal"][ind_to] = self.labels["frontal"][ind_to], self.labels["frontal"][ind_from]

            x, y = self.__get_data()

            self.data_gen = Data_generator(x[0], y[0], self.batch_size, self.__update_gen, self.mode, self.new_image_size)
            self.val_gen = Data_generator(x[1], y[1], self.batch_size, self.__update_gen, self.mode, self.new_image_size)

cross_validation = CrossValidation(FILE_DIRS["dicom"], FILE_DIRS["converted"], BATCH_SIZE, VALIDATION_NUM, MODE, IMG_SHAPE)

# Training
def make_loss(smooth=1e-6):
    bce_func = BinaryCrossentropy(from_logits=False)
    dice = Dice()
    def loss_f(y_true, y_pred):
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, dtype=tf.float32)

        return bce_func(y_true, y_pred) + dice(y_true, y_pred)
    return loss_f

loss_func = make_loss()

model_checkpoint = ModelCheckpoint(
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    filepath=os.path.join(RESULTS, 'saved_weights/{epoch}_unetpp_' + MODE + '.weights.h5')
)

model_loss_checkpoint = ModelCheckpoint(
    save_best_only=True,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    filepath=os.path.join(RESULTS, 'saved_weights/loss_unetpp_' + MODE + '.weights.h5')
)

class HistoryWriter(tf.keras.callbacks.Callback):
    def __init__(self, file_path):
        super().__init__()

        self.file_path = file_path

    def on_train_begin(self, logs={}):
        self.history={'output_4_loss': [], 'output_3_loss': [], 'output_2_loss': [], 'output_1_loss': [],
                    'val_output_4_loss': [], 'val_output_3_loss': [], 'val_output_2_loss': [], 'val_output_1_loss': []}

    def on_epoch_end(self, epoch, logs={}):
        for num in ['1', '2', '3', '4']:
            if logs.get(f'val_output_{num}_loss', None) != None:
                self.history[f'val_output_{num}_loss'].append(logs.get(f'val_output_{num}_loss'))
            if logs.get(f'output_{num}_loss', None) != None:
                self.history[f'output_{num}_loss'].append(logs.get(f'output_{num}_loss'))
        
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.history, f)

historyWriter = HistoryWriter(os.path.join(RESULTS, f"model_history_{MODE}"))

from model_unet import make_unet2p

model_unet = make_unet2p((*IMG_SHAPE, 1), filters=[64, 128, 256, 512, 1024], deep_supervision=True)
model_unet.compile(optimizer="adam", loss=[loss_func, loss_func, loss_func, loss_func], loss_weights=[1.0, 1.0, 1.0, 1.0], metrics=[Dice(), Dice(), Dice(), Dice()])
if WEIGHTS2LOAD: model_unet.load_weights(WEIGHTS2LOAD)

history_unet = model_unet.fit(x=cross_validation.data_gen, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=cross_validation.val_gen, callbacks=[model_checkpoint, model_loss_checkpoint, historyWriter])
