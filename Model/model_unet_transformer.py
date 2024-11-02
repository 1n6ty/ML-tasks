import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Dropout, concatenate, Conv2DTranspose, BatchNormalization, Lambda
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

def MHSA(x: KerasTensor, filters: int, kernel_size=3) -> KerasTensor: 
    """
        Multi_Head_Self_Attention\n
        x.shape = (BATCH_SIZE, height, width, channels)

        Parameters
        ----------

        x:
            KerasTensor input
        filters:
            Filters count for convolutional layers
        kernel_size:
            Size of the convolutional kernel
    """
    q = Conv2D(filters, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    k = Conv2D(filters, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    v = Conv2D(filters, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)

    attention = Conv2D(filters, (1, 1), activation="sigmoid", kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
        keras.ops.add(q, k)
    )

    return keras.ops.multiply(attention, v)

def MHCA(x: KerasTensor, y: KerasTensor, filters: int, kernel_size=3) -> KerasTensor:
    """
        Multi_Head_Cross_Attention\n
        x.shape = (BATCH_SIZE, height, width, channels_1)\n
        y.shape = (BATCH_SIZE, height / 2, width / 2, channels_2) <- higher level

        Parameters
        ----------

        x:
            KerasTensor input
        filters:
            Filters count for convolutional layers
        kernel_size:
            Size of the convolutional kernel
    """
    q = Conv2DTranspose(filters, (kernel_size, kernel_size), strides=(2, 2), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(y)
    k = Conv2D(filters, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    v = Conv2D(filters, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)

    attention = Conv2D(filters, (1, 1), activation="sigmoid", kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
        keras.ops.add(q, k)
    )
    return keras.ops.multiply(attention, v)

def make_unet_transformer(input_shape: tuple[int], filters: list[int], deep_supervision = True) -> Model:
    """
        This function makes unet_transformer model\n

        Parameters
        ----------

        input_shape:
            Shape of an input (height, width, channels)

        filters:
            list of filters for each layers (5 filters)

        deep_supervision:
            True (observes all 4 outputs), False (only the last one)
    """

    inp = Input(input_shape)

    conv_1 = standard_unit(inp, filters[0])
    MHSA_1 = MHSA(conv_1, filters[0])
    pool_1 = MaxPool2D((2, 2), (2, 2))(conv_1)

    conv_2 = standard_unit(pool_1, filters[1])
    MHSA_2 = MHSA(conv_2, filters[1])
    pool_2 = MaxPool2D((2, 2), (2, 2))(conv_2)

    conv_3 = standard_unit(pool_2, filters[2])
    MHSA_3 = MHSA(conv_3, filters[2])
    pool_3 = MaxPool2D((2, 2), (2, 2))(conv_3)

    conv_4 = standard_unit(pool_3, filters[3])
    MHSA_4 = MHSA(conv_4, filters[3])
    pool_4 = MaxPool2D((2, 2), (2, 2))(conv_4)

    conv_5 = standard_unit(pool_4, filters[4])
    MHSA_5 = MHSA(conv_5, filters[4])

    up_4 = standard_unit(
        concatenate([
            MHCA(MHSA_4, MHSA_5, filters[3]),
            Conv2DTranspose(filters[3], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(MHSA_5)
        ]),
        filters[3]
    )

    hid_3_1 = standard_unit(
        concatenate([
            MHCA(MHSA_3, MHSA_4, filters[2]),
            Conv2DTranspose(filters[2], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(MHSA_4)
        ]),
        filters[2]
    ) 
    up_3 = standard_unit(
        concatenate([
            MHCA(MHSA_3, up_4, filters[2]),
            MHCA(hid_3_1, up_4, filters[2]),
            Conv2DTranspose(filters[2], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(up_4)
        ]),
        filters[2]
    )

    hid_2_1 = standard_unit(
        concatenate([
            MHCA(MHSA_2, MHSA_3, filters[1]),
            Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(MHSA_3)
        ]),
        filters[1]
    )
    hid_2_2 = standard_unit(
        concatenate([
            MHCA(MHSA_2, hid_3_1, filters[1]),
            MHCA(hid_2_1, hid_3_1, filters[1]),
            Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(hid_3_1)
        ]),
        filters[1]
    )
    up_2 = standard_unit(
        concatenate([
            MHCA(MHSA_2, up_3, filters[1]),
            MHCA(hid_2_1, up_3, filters[1]),
            MHCA(hid_2_2, up_3, filters[1]),
            Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(up_3)
        ]),
        filters[1]
    )

    hid_1_1 = standard_unit(
        concatenate([
            MHCA(MHSA_1, MHSA_2, filters[0]),
            Conv2DTranspose(filters[0], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(MHSA_2)
        ]),
        filters[0]
    )
    hid_1_2 = standard_unit(
        concatenate([
            MHCA(MHSA_1, hid_2_1, filters[0]),
            MHCA(hid_1_1, hid_2_1, filters[0]),
            Conv2DTranspose(filters[0], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(hid_2_1)
        ]),
        filters[0]
    )
    hid_1_3 = standard_unit(
        concatenate([
            MHCA(MHSA_1, hid_2_2, filters[0]),
            MHCA(hid_1_1, hid_2_2, filters[0]),
            MHCA(hid_1_2, hid_2_2, filters[0]),
            Conv2DTranspose(filters[0], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(hid_2_2)
        ]),
        filters[0]
    )
    up_1 = standard_unit(
        concatenate([
            MHCA(MHSA_1, up_2, filters[0]),
            MHCA(hid_1_1, up_2, filters[0]),
            MHCA(hid_1_2, up_2, filters[0]),
            MHCA(hid_1_3, up_2, filters[0]),
            Conv2DTranspose(filters[0], (3, 3), strides=(2, 2), padding="same", activation='elu', kernel_initializer='he_normal')(up_2)
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
        return Model(inp, [output_1, output_2, output_3, output_4])
    else:
        return Model(inp, output_4)