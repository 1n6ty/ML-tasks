# Import Libs
import numpy as np

from random import randint
import cv2
import os
import pydicom
import json
import pickle

import tensorflow as tf
from tensorflow import keras
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint

# Init Global Variables
DATA_DIR = os.path.abspath('../DataSet')
RESULTS = os.path.abspath('../Results')

with open(os.path.join(DATA_DIR, 'files_routes.json'), 'r') as f:
    FILE_DIRS = json.loads(f.read())

IMG_SHAPE = (3408, 1552)
VALIDATION_NUM = 100
BATCH_SIZE = 1

WEIGHTS2LOAD = None #os.path.join(RESULTS, 'saved_weights/0-weights_unetpp.hdf5')

# Data Generator
class Data_train_generator(Sequence):
    def __init__(self, x_files_list: list, y_files_list: list, batch_size: int, n_parts: int, new_image_size = None, shuffle = True) -> None:
        self.data = x_files_list
        self.labels = y_files_list
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.new_image_size = new_image_size

        self.part_size = int(len(self.data) / n_parts)

    def __len__(self):
        return int(np.ceil(self.part_size / float(self.batch_size)))

    def __open_png_y(self, file_path):
        img = cv2.imread(file_path)
        img = cv2.inRange(img, (20, 20, 210), (40, 40, 240))
        
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
        l_d = len(self.data)
        if self.shuffle:
            for i in range(l_d):
                ind_from, ind_to = randint(0, l_d - 1), randint(0, l_d - 1)
                self.data[ind_from], self.data[ind_to] = self.data[ind_to], self.data[ind_from]
                self.labels[ind_from], self.labels[ind_to] = self.labels[ind_to], self.labels[ind_from]

# Training
def make_combine_loss(smooth=1e-6, gama=2): # Actually bin_cross_entropy + dice losses
    def comb_loss(y_true, y_pred):
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, dtype=tf.float32)

        nominator = 2 * tf.multiply(y_pred, y_true) + smooth
        denominator = (y_pred ** gama) + (y_true ** gama) + smooth
        dice_coef = tf.divide(nominator, denominator)

        cross_ent_coef = 0.5 * (tf.multiply(y_true, tf.math.log(y_pred + smooth)) + tf.multiply(1 - y_true, tf.math.log(1 - y_pred + smooth)))

        return tf.reduce_sum(1 - (cross_ent_coef + dice_coef))
    return comb_loss

combine_loss = make_combine_loss()

data_gen = Data_train_generator(FILE_DIRS["dicom"][:-VALIDATION_NUM], FILE_DIRS["converted"][:-VALIDATION_NUM], BATCH_SIZE, 7, IMG_SHAPE)
val_gen = Data_train_generator(FILE_DIRS["dicom"][-VALIDATION_NUM:], FILE_DIRS["converted"][-VALIDATION_NUM:], BATCH_SIZE, 1, IMG_SHAPE)

model_checkpoint = ModelCheckpoint(
    save_best_only=True,
    save_weights_only=True,
    monitor='val_output_4_loss',
    mode='min',
    filepath=os.path.join(RESULTS, 'saved_weights/{epoch}-weights_unetpp.hdf5')
)

from model import make_unet2p

model_unet = make_unet2p((*IMG_SHAPE, 1), filters=[64, 128, 256, 512, 1024], deep_supervision=True)

model_unet.compile(optimizer='Adam', loss={
    'output_1': combine_loss,
    'output_2': combine_loss,
    'output_3': combine_loss,
    'output_4': combine_loss
}, loss_weights=[1.0, 1.0, 1.0, 1.0])

if WEIGHTS2LOAD: model_unet.load_weights(WEIGHTS2LOAD)

history_unet = model_unet.fit(x=data_gen, epochs=5, validation_data=val_gen, callbacks=[model_checkpoint])

with open(os.path.join(RESULTS, 'model_history'), 'wb') as f:
    pickle.dump(history_unet.history, f)