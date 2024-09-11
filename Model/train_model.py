# Import Libs
import numpy as np

from random import randint
import cv2
import os
import pydicom
import json
import pickle

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.utils.all_utils import Sequence
from tensorflow.python.keras.callbacks import ModelCheckpoint

# Init Global Variables
DATA_DIR = os.path.abspath('../DataSet')
RESULTS = os.path.abspath('../Results')

MODE = "side"

with open(os.path.join(DATA_DIR, f'routes_{MODE}.json'), 'r') as f:
    FILE_DIRS = json.loads(f.read())

IMG_SHAPE = (3408, 1552)
VALIDATION_NUM = 
BATCH_SIZE = 1
EPOCHS = 200

WEIGHTS2LOAD = None #os.path.join(RESULTS, 'saved_weights/0-weights_unetpp.hdf5')

# Data Generator
class Data_train_generator(Sequence):
    def __init__(self, x_files_list: list, y_files_list: list, batch_size: int, update_func, new_image_size = None) -> None:
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
def make_combine_loss(smooth=1e-6, gama=2): # Actually bin_cross_entropy + dice_loss
    def comb_loss(y_true, y_pred):
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, dtype=tf.float32)

        nominator = 2 * tf.multiply(y_pred, y_true) + smooth
        denominator = (y_pred ** gama) + (y_true ** gama) + smooth
        dice_coef = tf.divide(nominator, denominator)

        cross_ent_coef = 0.5 * (tf.multiply(y_true, tf.math.log(y_pred + smooth)) + tf.multiply(1 - y_true, tf.math.log(1 - y_pred + smooth)))

        return tf.reduce_sum(1 - (cross_ent_coef + dice_coef))
    return comb_loss

combine_loss = make_combine_loss()

model_checkpoint = ModelCheckpoint(
    save_best_only=True,
    save_weights_only=True,
    monitor='val_output_4_loss',
    mode='min',
    filepath=os.path.join(RESULTS, 'saved_weights/{epoch}-weights_unetpp_' + MODE + '.hdf5')
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

from Model.model_unet import make_unet2p

model_unet = make_unet2p((*IMG_SHAPE, 1), filters=[64, 128, 256, 512, 1024], deep_supervision=True)

model_unet.compile(optimizer='Adam', loss={
    'output_1': combine_loss,
    'output_2': combine_loss,
    'output_3': combine_loss,
    'output_4': combine_loss
}, loss_weights=[1.0, 1.0, 1.0, 1.0])

if WEIGHTS2LOAD: model_unet.load_weights(WEIGHTS2LOAD)

history_unet = model_unet.fit(x=cross_validation.data_gen, epochs=EPOCHS, validation_data=cross_validation.val_gen, callbacks=[model_checkpoint, historyWriter])