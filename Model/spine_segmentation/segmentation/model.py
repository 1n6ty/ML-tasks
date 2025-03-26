"""
    This file contains model for spine-segmentation processing
"""

from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Dropout, concatenate, Conv2DTranspose, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

def standard_unit(input_tensor: KerasTensor, filters: int, activation_func: str, dropout_rate: float, kernel_size=3, name=None) -> KerasTensor:
    """
        Standart Convolution Unit\n
        `2x(Conv + dropout + batchNorm)`

        Parameters:
        ----------
            input_tensor:
                Input tensor to be produced
            \n
            filters:
                Number of filters
            \n
            activation_func:
                Activation function name
            \n
            dropout_rate:
                Dropout rate in [0, 1]
            \n
            kernel_size:
                Shape factor of the kernel
            \n
            name:
                Name of the standart_unit layer
    """
    x = Conv2D(filters, 
               (kernel_size, kernel_size), 
               activation=activation_func, 
               kernel_initializer = 'he_normal', 
               padding='same', 
               kernel_regularizer=l2(1e-4)
        )(input_tensor)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters,
               (kernel_size, kernel_size), 
               activation=activation_func, 
               kernel_initializer = 'he_normal', 
               padding='same', 
               kernel_regularizer=l2(1e-4)
        )(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization(name=name)(x)

    return x

def make_unet2p(input_shape: tuple[int, int, int], filters: list[int, int, int, int, int], conv_activation_func: str, dropout_rate: float, conv_kernel_size: int, deep_supervision=True) -> Model:
    """
        Makes unet_pp model

        Parameters:
        -----------
            input_shape:
                Shape of the input `[height, width, channels]`
            \n
            filters:
                List of filter numbers for each layers
            \n
            conv_activation_func:
                Activation function used in standart_unit layer
            \n
            dropout_rate:
                Dropout rate in [0, 1]
            \n
            conv_kernel_size:
                Shape factor of the kernel in standart_unit layer
            \n
            deep_supervision:
                True (observes all 4 outputs), False (only the last one)
    """
    img_input = Input(input_shape)

    conv_1 = standard_unit(img_input, filters[0], conv_activation_func, dropout_rate, conv_kernel_size)
    pool_1 = MaxPool2D((2, 2), (2, 2))(conv_1)

    conv_2 = standard_unit(pool_1, filters[1], conv_activation_func, dropout_rate, conv_kernel_size)
    pool_2 = MaxPool2D((2, 2), (2, 2))(conv_2)

    conv_3 = standard_unit(pool_2, filters[2], conv_activation_func, dropout_rate, conv_kernel_size)
    pool_3 = MaxPool2D((2, 2), (2, 2))(conv_3)

    conv_4 = standard_unit(pool_3, filters[3], conv_activation_func, dropout_rate, conv_kernel_size)
    pool_4 = MaxPool2D((2, 2), (2, 2))(conv_4)

    conv_5 = standard_unit(pool_4, filters[4], conv_activation_func, dropout_rate, conv_kernel_size)   

    up_4 = standard_unit(
        concatenate([
            conv_4,
            Conv2DTranspose(filters[4], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(conv_5)
        ]),
        filters[3],
        conv_activation_func, 
        dropout_rate, 
        conv_kernel_size
    )

    hid_3_1 = standard_unit(
        concatenate([
            conv_3,
            Conv2DTranspose(filters[3], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(conv_4)
        ]),
        filters[2],
        conv_activation_func, 
        dropout_rate, 
        conv_kernel_size
    ) 
    up_3 = standard_unit(
        concatenate([
            conv_3,
            hid_3_1,
            Conv2DTranspose(filters[3], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(up_4)
        ]),
        filters[2],
        conv_activation_func, 
        dropout_rate, 
        conv_kernel_size
    )

    hid_2_1 = standard_unit(
        concatenate([
            conv_2,
            Conv2DTranspose(filters[2], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(conv_3)
        ]),
        filters[1],
        conv_activation_func, 
        dropout_rate, 
        conv_kernel_size
    )
    hid_2_2 = standard_unit(
        concatenate([
            conv_2,
            hid_2_1,
            Conv2DTranspose(filters[2], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(hid_3_1)
        ]),
        filters[1],
        conv_activation_func, 
        dropout_rate, 
        conv_kernel_size
    )
    up_2 = standard_unit(
        concatenate([
            conv_2,
            hid_2_1,
            hid_2_2,
            Conv2DTranspose(filters[2], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(up_3)
        ]),
        filters[1],
        conv_activation_func, 
        dropout_rate, 
        conv_kernel_size
    )

    hid_1_1 = standard_unit(
        concatenate([
            conv_1,
            Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(conv_2)
        ]),
        filters[0],
        conv_activation_func, 
        dropout_rate, 
        conv_kernel_size
    )
    hid_1_2 = standard_unit(
        concatenate([
            conv_1,
            hid_1_1,
            Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(hid_2_1)
        ]),
        filters[0],
        conv_activation_func, 
        dropout_rate, 
        conv_kernel_size
    )
    hid_1_3 = standard_unit(
        concatenate([
            conv_1,
            hid_1_1,
            hid_1_2,
            Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(hid_2_2)
        ]),
        filters[0],
        conv_activation_func, 
        dropout_rate, 
        conv_kernel_size
    )
    up_1 = standard_unit(
        concatenate([
            conv_1,
            hid_1_1,
            hid_1_2,
            hid_1_3,
            Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(up_2)
        ]),
        filters[0],
        conv_activation_func, 
        dropout_rate, 
        conv_kernel_size
    )

    output_4 = Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal", activation="sigmoid", name="output_4")(
        up_1
    )
    output_3 = Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal", activation="sigmoid", name="output_3")(
        hid_1_3
    )
    output_2 = Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal", activation="sigmoid", name="output_2")(
        hid_1_2
    )
    output_1 = Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal", activation="sigmoid", name="output_1")(
        hid_1_1
    )

    if deep_supervision:
        return Model(img_input, [output_1, output_2, output_3, output_4])
    else:
        return Model(img_input, output_4)