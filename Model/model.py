import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPool2D, Input, Dropout, concatenate, UpSampling2D, BatchNormalization, Layer, Softmax, Conv2DTranspose
from keras.models import Model

dropout_rate = 0.1

def standard_unit(input_tensor, filters, kernel_size=3, name=None):
    act = 'elu'

    x = Conv2D(filters, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization(name=name)(x)

    return x

class Attention(Layer):
    def __init__(self, kernel_size = 1):
        super().__init__()

        self.kernel_size = kernel_size

        self.softmax_layer = Softmax(axis = [1, 2])
    
    def build(self, input_shape):
        self.conv_query = Conv2D(
            input_shape = input_shape[-1:],
            filters = input_shape[-1],
            kernel_size = self.kernel_size,
            padding = "same",
            kernel_initializer = "he_normal",
            activation = "linear"
        )
        self.conv_key = Conv2D(
            input_shape = input_shape[-1:],
            filters = input_shape[-1],
            kernel_size = self.kernel_size,
            padding = "same",
            kernel_initializer = "he_normal",
            activation = "linear"
        )
        self.conv_value = Conv2D(
            input_shape = input_shape[-1:],
            filters = input_shape[-1],
            kernel_size = self.kernel_size,
            padding = "same",
            kernel_initializer = "he_normal",
            activation = "linear"
        )

    def call(self, inputs):
        query_layer = self.conv_query(inputs)
        key_layer = self.conv_key(inputs)
        value_layer = self.conv_value(inputs)

        kv_layer = tf.multiply(query_layer, key_layer)

        sm_layer = self.softmax_layer(kv_layer)
        
        weighted_layer = tf.multiply(sm_layer, value_layer)
        
        return tf.add(weighted_layer, inputs)

def make_unet2p(input_shape, filters, deep_supervision):
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
        return Model(img_input, up_1)

def make_unet3p(input_shape, filters, skip_filters = 64, up_filters = 320, deep_supervision = True):
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

    up_4 = concatenate([
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            MaxPool2D((8, 8), (8, 8))(conv_1)
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            MaxPool2D((4, 4), (4, 4))(conv_2)
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            pool_3
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            conv_4
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            UpSampling2D((2, 2), interpolation="bilinear")(conv_5)
        )
    ], axis=-1)
    up_4 = standard_unit(up_4, up_filters)

    up_3 = concatenate([
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            MaxPool2D((4, 4), (4, 4))(conv_1)
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            pool_2
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            conv_3
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            UpSampling2D((2, 2), interpolation="bilinear")(conv_4)
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            UpSampling2D((4, 4), interpolation="bilinear")(conv_5)
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            UpSampling2D((2, 2), interpolation="bilinear")(up_4)
        )
    ], axis=-1)
    up_3 = standard_unit(up_3, up_filters)

    up_2 = concatenate([
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            pool_1
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            conv_2
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            UpSampling2D((2, 2), interpolation="bilinear")(conv_3)
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            UpSampling2D((4, 4), interpolation="bilinear")(conv_4)
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            UpSampling2D((8, 8), interpolation="bilinear")(conv_5)
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            UpSampling2D((2, 2), interpolation="bilinear")(up_3)
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            UpSampling2D((4, 4), interpolation="bilinear")(up_4)
        )
    ], axis=-1)
    up_2 = standard_unit(up_2, up_filters)

    upSample_2 = UpSampling2D((2, 2), interpolation="bilinear")(up_2)
    upSample_3 = UpSampling2D((4, 4), interpolation="bilinear")(up_3)
    upSample_4 = UpSampling2D((8, 8), interpolation="bilinear")(up_4)

    up_1 = concatenate([
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            conv_1
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            UpSampling2D((2, 2), interpolation="bilinear")(conv_2)
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            UpSampling2D((4, 4), interpolation="bilinear")(conv_3)
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            UpSampling2D((8, 8), interpolation="bilinear")(conv_4)
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            UpSampling2D((16, 16), interpolation="bilinear")(conv_5)
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            upSample_2
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            upSample_3
        ),
        Conv2D(skip_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            upSample_4
        )
    ], axis=-1)
    up_1 = standard_unit(up_1, up_filters)

    output_1 = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4), name="output_1")(
        Attention()(up_1)
    )

    if deep_supervision:
        output_2 = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4), name="output_2")(
            Attention()(
                Conv2D(up_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
                    upSample_2
                )
            )
        )
        output_3 = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4), name="output_3")(
            Attention()(
                Conv2D(up_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
                    upSample_3
                )
            )
        )
        output_4 = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4), name="output_4")(
            Attention()(
                Conv2D(up_filters, (3, 3), activation='elu', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
                    upSample_4
                )
            )
        )

        return Model(img_input, [output_1, output_2, output_3, output_4])
    else:
        return Model(img_input, output_1)