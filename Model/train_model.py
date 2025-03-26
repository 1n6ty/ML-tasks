"""
    This file contains model training script
"""

import os
import sys
import json
import pickle
import tomllib

from typing import Literal

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.losses import BinaryCrossentropy, Dice

from spine_segmentation.segmentation.datagen import CrossValidation_train_generator
from spine_segmentation.segmentation.model import make_unet2p

# Parse config
if len(sys.argv) < 2:
    raise AttributeError("config.toml file must be provided")

with open(sys.argv[1], 'rb') as f:
    conf = tomllib.load(f)

RESULTS_DIR: str = conf["basic"]["RESULTS_DIR"]
with open(conf["basic"]["ROUTES_FILE"], 'rb') as f:
    FILE_DIRS: dict[Literal["dicom", "converted"], dict[Literal["side", "frontal"], str]] = json.load(f)

IMG_SHAPE: list[int] = conf["settings"]["IMG_SHAPE"]
VALIDATION_NUM: int = conf["settings"]["VALIDATION_NUM"]
BATCH_SIZE: int = conf["settings"]["BATCH_SIZE"]
EPOCHS: int = conf["settings"]["EPOCHS"]

MODE: Literal["side", "frontal"] = conf["settings"]['MODE']
WPART: Literal["spine", "hip"] = conf["settings"]['WPART']

WEIGHTS2LOAD: str | None = conf["basic"]['WEIGHTS2LOAD'] if len(conf["basic"]['WEIGHTS2LOAD']) > 0 else None

DEEP_SUPERVISION: bool = conf["model"]['DEEP_SUPERVISION']
FILTERS: list[int] = conf["model"]['FILTERS']
CONV_ACTIVATION_FUNC: str = conf["model"]['CONV_ACTIVATION_FUNC']
CONV_KERNEL_SIZE: int = conf["model"]['CONV_KERNEL_SIZE']
DROPOUT_RATE: float = conf["model"]['DROPOUT_RATE']

cross_validation_generator = CrossValidation_train_generator(FILE_DIRS["dicom"], FILE_DIRS["converted"], BATCH_SIZE, VALIDATION_NUM, MODE, WPART, DEEP_SUPERVISION, IMG_SHAPE)

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
    filepath=os.path.join(RESULTS_DIR, 'saved_weights/{epoch}_unetpp_' + MODE + '_' + WPART + '.weights.h5')
)

model_loss_checkpoint = ModelCheckpoint(
    save_best_only=True,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    filepath=os.path.join(RESULTS_DIR, 'saved_weights/loss_unetpp_' + MODE + '_' + WPART + '.weights.h5')
)

class HistoryWriter(Callback):
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

historyWriter = HistoryWriter(os.path.join(RESULTS_DIR, f"model_history_{MODE}_{WPART}"))

import seaborn as sns
import matplotlib.pyplot as plt

print(cross_validation_generator.data_gen[0])
sns.heatmap(cross_validation_generator.data_gen[0])

model_unet = make_unet2p((*IMG_SHAPE, 1), filters=FILTERS, conv_activation_func=CONV_ACTIVATION_FUNC, dropout_rate=DROPOUT_RATE, conv_kernel_size=CONV_KERNEL_SIZE, deep_supervision=DEEP_SUPERVISION)
model_unet.compile(optimizer="adam", loss=[loss_func, loss_func, loss_func, loss_func], loss_weights=[1.0, 1.0, 1.0, 1.0], metrics=[Dice(), Dice(), Dice(), Dice()])
if WEIGHTS2LOAD: model_unet.load_weights(WEIGHTS2LOAD)

history_unet = model_unet.fit(x=cross_validation_generator.data_gen, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=cross_validation_generator.val_gen, callbacks=[model_checkpoint, model_loss_checkpoint, historyWriter])
