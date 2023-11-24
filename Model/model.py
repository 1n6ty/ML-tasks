import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l2
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Input, Dropout, concatenate, UpSampling2D, Activation, BatchNormalization
from keras.models import Model

dropout_rate = 0.1

def standard_unit(input_tensor, filters, kernel_size=3):
    act = 'elu'

    x = Conv2D(filters, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filters, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate)(x)

    return x

def make_UNet2p(img_rows, img_cols, filters: list, color_type=1, num_class=1, deep_supervision=False):
    img_input = Input(shape=(img_rows, img_cols, color_type))

    conv1_1 = standard_unit(img_input, filters[0])
    pool1 = MaxPool2D((2, 2), strides=(2, 2))(conv1_1)

    conv2_1 = standard_unit(pool1, filters[1])
    pool2 = MaxPool2D((2, 2), strides=(2, 2))(conv2_1)

    up1_2 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], axis=3)
    conv1_2 = standard_unit(conv1_2, filters[0])

    conv3_1 = standard_unit(pool2, filters[2])
    pool3 = MaxPool2D((2, 2), strides=(2, 2))(conv3_1)

    up2_2 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], axis=3)
    conv2_2 = standard_unit(conv2_2, filters[1])

    up1_3 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], axis=3)
    conv1_3 = standard_unit(conv1_3, filters[0])

    conv4_1 = standard_unit(pool3, filters[3])
    pool4 = MaxPool2D((2, 2), strides=(2, 2))(conv4_1)

    up3_2 = Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], axis=3)
    conv3_2 = standard_unit(conv3_2, filters[2])

    up2_3 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], axis=3)
    conv2_3 = standard_unit(conv2_3, filters[1])

    up1_4 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], axis=3)
    conv1_4 = standard_unit(conv1_4, filters[0])

    conv5_1 = standard_unit(pool4, filters[4])

    up4_2 = Conv2DTranspose(filters[3], (2, 2), strides=(2, 2), padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], axis=3)
    conv4_2 = standard_unit(conv4_2, filters[3])

    up3_3 = Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], axis=3)
    conv3_3 = standard_unit(conv3_3, filters[2])
    up2_4 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], axis=3)
    conv2_4 = standard_unit(conv2_4, filters[1])

    up1_5 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], axis=3)
    conv1_5 = standard_unit(conv1_5, filters[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(img_input, [nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4])
    else:
        model = Model(img_input, [nestnet_output_4])
    
    return model

def make_UNet3p(img_shape: tuple[int, int, int], filters: tuple, deep_supervision=False):
    inp = Input(img_shape)

    conv1_d = standard_unit(inp, filters=filters[0])
    pool1 = MaxPool2D((2, 2), strides=(2, 2))(conv1_d)

    conv2_d = standard_unit(pool1, filters=filters[1])
    pool2 = MaxPool2D((2, 2), strides=(2, 2))(conv2_d)

    conv3_d = standard_unit(pool2, filters=filters[2])
    pool3 = MaxPool2D((2, 2), strides=(2, 2))(conv3_d)

    conv4_d = standard_unit(pool3, filters=filters[3])
    pool4 = MaxPool2D((2, 2), strides=(2, 2))(conv4_d)

    conv5_d = standard_unit(pool4, filters=filters[4])

    conv4_up_inp = concatenate([
        MaxPool2D((8, 8), strides=(8, 8))(conv1_d),
        MaxPool2D((4, 4), strides=(4, 4))(conv2_d),
        MaxPool2D((2, 2), strides=(2, 2))(conv3_d),
        conv4_d,
        UpSampling2D((2, 2), interpolation="bilinear")(conv5_d)
    ], axis=3)
    conv4_up_inp = Activation('elu')(conv4_up_inp)
    conv4_up = standard_unit(conv4_up_inp, filters=filters[5])

    conv3_up_inp = concatenate([
        MaxPool2D((4, 4), strides=(4, 4))(conv1_d),
        MaxPool2D((2, 2), strides=(2, 2))(conv2_d),
        conv3_d,
        UpSampling2D((2, 2), interpolation="bilinear")(conv4_up),
        UpSampling2D((4, 4), interpolation="bilinear")(conv5_d)
    ], axis=3)
    conv3_up_inp = Activation('elu')(conv3_up_inp)
    conv3_up = standard_unit(conv3_up_inp, filters=filters[6])

    conv2_up_inp = concatenate([
        MaxPool2D((2, 2), strides=(2, 2))(conv1_d),
        conv2_d,
        UpSampling2D((2, 2), interpolation="bilinear")(conv3_up),
        UpSampling2D((4, 4), interpolation="bilinear")(conv4_d),
        UpSampling2D((8, 8), interpolation="bilinear")(conv5_d)
    ], axis=3)
    conv2_up_inp = Activation('elu')(conv2_up_inp)
    conv2_up = standard_unit(conv2_up_inp, filters=filters[7])

    conv1_up_inp = concatenate([
        conv1_d,
        UpSampling2D((2, 2), interpolation="bilinear")(conv2_up),
        UpSampling2D((4, 4), interpolation="bilinear")(conv3_d),
        UpSampling2D((8, 8), interpolation="bilinear")(conv4_d),
        UpSampling2D((16, 16), interpolation="bilinear")(conv5_d)
    ], axis=3)
    conv1_up_inp = Activation('elu')(conv1_up_inp)
    conv1_up = standard_unit(conv1_up_inp, filters=filters[8])

    output_1 = Activation(activation="sigmoid", name='output_1')(
        Conv2D(1, (3, 3), kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
            conv1_up
        )
    )
    output_2 = Activation(activation='sigmoid', name='output_2')(
        UpSampling2D((2, 2), interpolation="bilinear")(
            Conv2D(1, (3, 3), kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
                conv2_up
            )
        )
    )
    output_3 = Activation(activation='sigmoid', name='output_3')(
        UpSampling2D((4, 4), interpolation="bilinear")(
            Conv2D(1, (3, 3), kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
                conv3_up
            )
        )
    )
    output_4 = Activation(activation='sigmoid', name='output_4')(
        UpSampling2D((8, 8), interpolation="bilinear")(
            Conv2D(1, (3, 3), kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
                conv4_up
            )
        )
    )
    output_5 = Activation(activation='sigmoid', name='output_5')(
        UpSampling2D((16, 16), interpolation="bilinear")(
            Conv2D(1, (3, 3), kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(
                conv5_d
            )
        )
    )

    if deep_supervision:
        model = Model(inp, [output_1, output_2, output_3, output_4, output_5])
    else:
        model = Model(inp, output_1)
    
    return model