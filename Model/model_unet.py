import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Dropout, concatenate, UpSampling2D, Layer, Softmax, Conv2DTranspose, Layer, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

act = 'elu'
dropout_rate = 0.1

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

def make_unet2p(input_shape: tuple[int, int, int], filters: list[int, int, int, int, int], deep_supervision) -> Model:
    """
        This function makes unet_pp model
        filters - list of filters for each layers
        deep_supervision - True (observes all 4 outputs), False (only the last one)
    """
    img_input = Input(input_shape)

    conv_1 = standard_unit(img_input, filters[0])
    pool_1 = MaxPool2D((2, 2), (2, 2))(conv_1)

    conv_2 = standard_unit(pool_1, filters[1])
    pool_2 = MaxPool2D((2, 2), (2, 2))(conv_2)

    conv_3 = standard_unit(pool_2, filters[2])
    pool_3 = MaxPool2D((2, 2), (2, 2))(conv_3)

    conv_4 = standard_unit(pool_3, filters[3])
    pool_4 = MaxPool2D((2, 2), (2, 2))(conv_4)

    conv_5 = standard_unit(pool_4, filters[4])   

    up_4 = standard_unit(
        concatenate([
            conv_4,
            Conv2DTranspose(filters[4], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(conv_5)
        ]),
        filters[3]
    )

    hid_3_1 = standard_unit(
        concatenate([
            conv_3,
            Conv2DTranspose(filters[3], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(conv_4)
        ]),
        filters[2]
    ) 
    up_3 = standard_unit(
        concatenate([
            conv_3,
            hid_3_1,
            Conv2DTranspose(filters[3], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(up_4)
        ]),
        filters[2]
    )

    hid_2_1 = standard_unit(
        concatenate([
            conv_2,
            Conv2DTranspose(filters[2], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(conv_3)
        ]),
        filters[1]
    )
    hid_2_2 = standard_unit(
        concatenate([
            conv_2,
            hid_2_1,
            Conv2DTranspose(filters[2], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(hid_3_1)
        ]),
        filters[1]
    )
    up_2 = standard_unit(
        concatenate([
            conv_2,
            hid_2_1,
            hid_2_2,
            Conv2DTranspose(filters[2], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(up_3)
        ]),
        filters[1]
    )

    hid_1_1 = standard_unit(
        concatenate([
            conv_1,
            Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(conv_2)
        ]),
        filters[0]
    )
    hid_1_2 = standard_unit(
        concatenate([
            conv_1,
            hid_1_1,
            Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(hid_2_1)
        ]),
        filters[0]
    )
    hid_1_3 = standard_unit(
        concatenate([
            conv_1,
            hid_1_1,
            hid_1_2,
            Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(hid_2_2)
        ]),
        filters[0]
    )
    up_1 = standard_unit(
        concatenate([
            conv_1,
            hid_1_1,
            hid_1_2,
            hid_1_3,
            Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(up_2)
        ]),
        filters[0]
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