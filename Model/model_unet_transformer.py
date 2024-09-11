import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Input, Dropout, concatenate, Layer, Softmax, Layer
from tensorflow.python.keras.models import Model

act = 'elu'
dropout_rate = 0.1

def get_PE_matrix(rows, cols, n=100000):
    """
    Generates Positional Encoding Matrix
    """
    common_col = np.arange(0, rows - 1, 1, dtype=np.float32)
    return np.array([np.cos((common_col / (n ** (d - 1))) if d % 2 else np.sin(common_col / (n ** d))) for d in range(cols)]).T

class BatchNormalization(Layer):
    def __init__(self, gamma = 1e-6, name=None, dtype="float32", dynamic=False, **kwargs):
        super().__init__(trainable=False, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

        self.gamma = gamma
        self.dtype = dtype

    def call(self, inputs, *args, **kwargs):
        inputs = tf.cast(inputs, dtype=self.dtype)
        d = tf.reduce_prod(tf.cast(inputs.shape, dtype=self.dtype))
        
        mean = tf.reduce_sum(inputs) / d
        std = tf.reduce_sum(tf.square(-inputs + mean)) / d

        return (inputs - mean) / tf.sqrt(std + self.gamma)

class MHSA(Layer): # Multi-Head-Cross-Attention Layer
    def __init__(self, filters, trainable=True, name=None, dtype="float32", dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

        self.filters = filters
    
    def build(self, input_shape):
        pass

    def call(self, inputs):
        pass

def standard_unit(input_tensor, filters, kernel_size=3, name=None):
    x = Conv2D(filters, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    return x

def make_unet_transformer(input_shape: tuple, filters: list[int, int, int, int, int]) -> Model:
    x = Input(shape=input_shape)
    
    conv_1 = standard_unit(x, filters[0])
    pool_1 = MaxPool2D((2, 2), (2, 2))(conv_1)

    conv_2 = standard_unit(pool_1, filters[1])
    pool_2 = MaxPool2D((2, 2), (2, 2))(conv_2)

    conv_3 = standard_unit(pool_2, filters[2])
    pool_3 = MaxPool2D((2, 2), (2, 2))(conv_3)

    conv_4 = standard_unit(pool_3, filters[3])
    pool_4 = MaxPool2D((2, 2), (2, 2))(conv_4)

    conv_5 = standard_unit(pool_4, filters[4])

