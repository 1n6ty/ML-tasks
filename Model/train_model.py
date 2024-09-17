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

with open(os.path.join(DATA_DIR, f'routes_{MODE}.json'), 'r') as f:
    FILE_DIRS = json.loads(f.read())

IMG_SHAPE = (1760, 768)
VALIDATION_NUM = 50
BATCH_SIZE = 8
EPOCHS = 200

WEIGHTS2LOAD = os.path.join(RESULTS, 'saved_weights/init_unetpp_' + MODE + '.weights.h5')

# Data Generator
class Data_train_generator(Sequence):
    def __init__(self, x_files_list: list, y_files_list: list, batch_size: int, update_func, new_image_size = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data = x_files_list
        self.labels = y_files_list
        self.batch_size = batch_size

        self.new_image_size = new_image_size

        self.l_d = len(self.data)
        self.update_func = update_func

    def __len__(self):
        return int(np.ceil((self.l_d) / float(self.batch_size)))

    def __open_png_y(self, file_path):
        img = cv2.imread(file_path)
        img = cv2.inRange(img, (0, 0, 210), (40, 40, 256))
        
        data = np.array(img, dtype="float32") / 255

        if self.new_image_size != None:
            data = cv2.resize(data, self.new_image_size[::-1], interpolation=cv2.INTER_CUBIC)

        return np.reshape(data, (*self.new_image_size, 1))

    def __open_dcm_x(self, file_path):
        dcm = pydicom.dcmread(file_path)

        data = dcm.pixel_array.astype("float32") / 255

        if self.new_image_size != None:
            data = cv2.resize(data, self.new_image_size[::-1], interpolation=cv2.INTER_CUBIC)

        return np.reshape(data, (*self.new_image_size, 1))

    def __getitem__(self, index):
        batch_x = np.array(list(map(self.__open_dcm_x, self.data[index * self.batch_size: (index + 1) * self.batch_size])))
        batch_y = np.array(list(map(self.__open_png_y, self.labels[index * self.batch_size: (index + 1) * self.batch_size])))

        return batch_x, batch_y

    def on_epoch_end(self):
        self.update_func()

class CrossValidation:
    def __get_data(self):
        return [
                [self.data[:-self.val_size], self.data[-self.val_size:]],
                [self.labels[:-self.val_size], self.labels[-self.val_size:]]
            ]
    
    def __update_gen(self):
        if self.update_count < 2:
            self.update_count += 1
        else:
            self.update_count = 0

            for i in range(self.l_d):
                ind_from, ind_to = randint(0, self.l_d - 1), randint(0, self.l_d - 1)
                self.data[ind_from], self.data[ind_to] = self.data[ind_to], self.data[ind_from]
                self.labels[ind_from], self.labels[ind_to] = self.labels[ind_to], self.labels[ind_from]

            x, y = self.__get_data()

            self.data_gen = Data_train_generator(x[0], y[0], self.batch_size, self.__update_gen, self.new_image_size)
            self.val_gen = Data_train_generator(x[1], y[1], self.batch_size, self.__update_gen, self.new_image_size)
        

    def __init__(self, x_files_list: list, y_files_list: list, batch_size: int, val_size: int, new_image_size = None) -> None:
        self.data = x_files_list
        self.labels = y_files_list
        self.batch_size = batch_size

        self.new_image_size = new_image_size

        self.l_d = min(len(self.data), len(self.labels))
        self.val_size = val_size
        self.update_count = 0

        x, y = self.__get_data()

        self.data_gen = Data_train_generator(x[0], y[0], self.batch_size, self.__update_gen, self.new_image_size)
        self.val_gen = Data_train_generator(x[1], y[1], self.batch_size, self.__update_gen, self.new_image_size)

cross_validation = CrossValidation(FILE_DIRS["dicom"], FILE_DIRS["converted"], BATCH_SIZE, VALIDATION_NUM, IMG_SHAPE)

# Training
def make_loss(smooth=1e-6):
    bce_func = BinaryCrossentropy(from_logits=True)
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
model_unet.compile(optimizer="adam", loss={
    'output_1': loss_func,
    'output_2': loss_func,
    'output_3': loss_func,
    'output_4': loss_func
}, loss_weights=[1.0, 1.0, 1.0, 1.0])
if WEIGHTS2LOAD: model_unet.load_weights(WEIGHTS2LOAD)

history_unet = model_unet.fit(x=cross_validation.data_gen, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=cross_validation.val_gen, callbacks=[model_checkpoint, model_loss_checkpoint, historyWriter])
