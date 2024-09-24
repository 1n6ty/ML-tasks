import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Dropout, concatenate, UpSampling2D, Layer, Softmax, Layer, BatchNormalization, Reshape, Activation, multiply, Dot, Permute
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

act = 'elu'
dropout_rate = 0.1

class Attention(Layer):
    """
        Basic Attention layer

        Softmax(QK^T)V
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.softmax_layer = Softmax(axis=-1)
        self.dot_layer = Dot(axes=(1, 2))
        self.permute_layer = Permute((2, 1))

    def build(self, input_shape: list) -> None:
        self.sigma = tf.math.sqrt(tf.cast(input_shape[0][2], dtype=tf.float32))
        
        self.q_weights = self.add_weight(
            shape=(input_shape[0][2], input_shape[0][2]),
            initializer="random_normal",
            trainable=True
        )
        self.q_bias = self.add_weight(
            shape=(input_shape[0][1], input_shape[0][2]),
            initializer="random_normal",
            trainable=True
        )

        self.k_weights = self.add_weight(
            shape=(input_shape[1][2], input_shape[1][2]),
            initializer="random_normal",
            trainable=True
        )
        self.k_bias = self.add_weight(
            shape=(input_shape[1][1], input_shape[1][2]),
            initializer="random_normal",
            trainable=True
        )

        self.v_weights = self.add_weight(
            shape=(input_shape[2][2], input_shape[2][2]),
            initializer="random_normal",
            trainable=True
        )
        self.v_bias = self.add_weight(
            shape=(input_shape[2][1], input_shape[2][2]),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs: list) -> KerasTensor:
        """
            Call Attention Layer

            Inputs
            -------

            `list`[query, key, value], `shape` of each should be (batch_size, features, dim)
        """
        Q_weighted = self.dot_layer(inputs[0], self.q_weights) + self.q_bias
        K_weighted = self.dot_layer(inputs[1], self.k_weights) + self.k_bias
        V_weighted = self.dot_layer(inputs[2], self.v_weights) + self.v_bias

        return self.dot_layer(
            self.softmax_layer(
                self.dot_layer(
                    Q_weighted, self.permute_layer(K_weighted)
                ) / self.sigma
            ), V_weighted
        )
    
    def compute_output_shape(self, input_shape: list) -> tf.Tensor:
        return input_shape[0]

def standard_unit(input_tensor: KerasTensor, filters: int, kernel_size=3, name=None) -> KerasTensor:
    """
        Standart Convolution Unit
        2xConv(kernel_size * kernel_size * filters + dropout + batchNorm)
    """
    x = Conv2D(filters, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization(name=name)(x)

    return x

def MHSA(x: KerasTensor, layer_num: int, input_shape: tuple[int, int], filter_x: int, PE: tf.Tensor, heads_num: int) -> KerasTensor: 
    """
        Multi_Head_Self_Attention
        x.shape = (BATCH_SIZE, rows, cols, filters)
        layer_num - layer on which MHSA stands [1, n]
        filter_x - filters count on layer
        PE - Position Encoding Matrix
        heads_num - number of attention heads
    """
    x = Reshape((int(input_shape[0] * input_shape[1] / (4 ** (layer_num - 1))), filter_x))(x) + PE
    x = Attention()([x, x, x])
    for i in range(heads_num - 1):
        x += Attention()([x, x, x])
    return Reshape((int(input_shape[0] / (2 ** (layer_num - 1))), int(input_shape[1] / (2 ** (layer_num - 1))), filter_x))(x)

def MHCA(x_inp: KerasTensor, y_inp: KerasTensor, layer_num: int, input_shape: tuple[int, int], filter_x: int, PE_x: tf.Tensor, heads_num: int) -> KerasTensor:
    """
        Multi_Head_Cross_Attention
        x_inp is input of more abstract layer than y_inp
        layer_num - layer on which MHSA stands [1, n]
        filter_x - filters count on layer
        PE_x - Position Encoding Matrix
        heads_num - number of attention heads
    """
    y_inp = UpSampling2D((2, 2))(y_inp)
    y_inp = standard_unit(y_inp, filter_x)

    map_shape = (int(input_shape[0] * input_shape[1] / (4 ** (layer_num - 1))), filter_x)
    x = Reshape(map_shape)(x_inp) + PE_x
    y = Reshape(map_shape)(y_inp) + PE_x

    ca = Attention()([y, y, x])
    for i in range(heads_num - 1):
        ca += Attention()([y, y, x])
    ca = Reshape((int(input_shape[0] / (2 ** (layer_num - 1))), int(input_shape[1] / (2 ** (layer_num - 1))), filter_x))(ca)
    ca = Conv2D(1, (1, 1), activation='relu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(ca)
    ca = BatchNormalization()(ca)
    ca = Activation("sigmoid")(ca)

    x_inp = multiply([x_inp, ca])

    return concatenate([x_inp, y_inp], axis=-1)


def make_unet_transformer(input_shape: tuple[int, int, int], filters: list[int, int, int, int, int], PE: list[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], heads_num=16) -> Model:
    """
        This function makes a unet_transformer model
        filters - list of filters for each layers
        PE - Position Encoding Matrixes for each layer
        heads_num - numbers of heads for multi_head_attention blocks
    """
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
    mhsa = MHSA(conv_5, 5, input_shape, filters[4], PE[4], heads_num)
    
    mhca_4 = MHCA(conv_4, mhsa, 4, input_shape, filters[3], PE[3], heads_num)
    mhca_4 = standard_unit(mhca_4, filters[3])

    mhca_3 = MHCA(conv_3, mhca_4, 3, input_shape, filters[2], PE[2], heads_num)
    mhca_3 = standard_unit(mhca_3, filters[2])

    mhca_2 = MHCA(conv_2, mhca_3, 2, input_shape, filters[1], PE[1], heads_num)
    mhca_2 = standard_unit(mhca_2, filters[1])

    mhca_1 = MHCA(conv_1, mhca_2, 1, input_shape, filters[0], PE[0], heads_num)
    mhca_1 = standard_unit(mhca_1, filters[0])

    out = Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal", activation="sigmoid", name="output")(mhca_1)

    return Model(x, out)