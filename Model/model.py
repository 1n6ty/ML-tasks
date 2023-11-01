import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, ConvLSTM2D, Conv2D, Dropout, Concatenate, Reshape, Flatten, Dense, MaxPooling2D, Conv2DTranspose

def conv_block(x, n_filter: int, kernel_size: tuple[int, int], dropout_rate = 0.1):
    x = Conv2D(n_filter, kernel_size=kernel_size, padding="same", data_format="channels_last", kernel_initializer="he_normal", activation='elu')(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(n_filter, kernel_size=kernel_size, padding="same", data_format="channels_last", kernel_initializer="he_normal", activation='elu')(x)
    x = Dropout(dropout_rate)(x)
    return x

def attention_block(x, x_shape: tuple[int, int, int]): # shape: ([Batch_size,] height, width, filters)
    inp = Flatten(data_format="channels_last")(x)

    weights = Dense(x_shape[2], activation="softmax")(inp)
    weights = Reshape((1, 1, x_shape[2]))(weights)
    weights = Concatenate(axis=2)([weights for i in range(x_shape[1])])
    weights = Concatenate(axis=1)([weights for i in range(x_shape[0])])

    return tf.multiply(x, weights)


def make_ConvLSTM(
            input_shape: tuple[int, int, int, int], n_filters: int, kernel_size: tuple[int, int], out_seq: int, dropout_rate = 0.1
        ):
    inp = Input(input_shape)
    
    # Downsampling
    down_inp = tf.split(inp, input_shape[0], axis=1)

    for i in range(input_shape[0]):
        down_inp[i] = Reshape(input_shape[1:])(down_inp[i])
    
    for step in range(0, 3):
        for i in range(input_shape[0]):
            down_inp[i] = conv_block(down_inp[i], n_filters * (2 ** step), (3, 3), dropout_rate)
            down_inp[i] = MaxPooling2D((2, 2), strides=(2, 2))(down_inp[i])
    
    for i in range(input_shape[0]):
        down_inp[i] = tf.expand_dims(down_inp[i], axis=1)
    
    # Encoder
    conv_inp = Concatenate(axis=1)(down_inp)
    conv_lstm = ConvLSTM2D(filters=n_filters * 4, kernel_size=(3, 3), padding="same")(conv_inp)
    
    # Decoder
    decoder_inp = Concatenate(axis=1)([tf.expand_dims(attention_block(conv_lstm, (int(input_shape[1] / 8), int(input_shape[2] / 8), n_filters * 4)), axis=1) for i in range(out_seq)])
    decoder_lstm = ConvLSTM2D(filters=n_filters * 4, kernel_size=(3, 3), padding="same", return_sequences=True)(decoder_inp)

    decoder_lstm = tf.split(decoder_lstm, out_seq, axis=1)

    for i in range(out_seq):
        decoder_lstm[i] = Reshape((int(input_shape[1] / 8), int(input_shape[2] / 8), n_filters * 4))(decoder_lstm[i])
    
    for step in range(2, -1, -1):
        for i in range(out_seq):
            decoder_lstm[i] = Conv2DTranspose(n_filters * (2 ** step), (3, 3), strides=(2, 2), padding="same")(decoder_lstm[i])
            decoder_lstm[i] = Dropout(dropout_rate)(decoder_lstm[i])

    # Output
    for i in range(out_seq):
        decoder_lstm[i] = Conv2D(1, (3, 3), padding="same", kernel_initializer="he_normal", activation="linear")(decoder_lstm[i])
        decoder_lstm[i] = tf.expand_dims(decoder_lstm[i], axis=1)
    
    output = Concatenate(axis=1)(decoder_lstm)

    return Model(inputs=[inp], outputs=[output])