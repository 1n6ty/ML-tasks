import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, ConvLSTM2D, Conv2D, Dropout, Concatenate, Reshape, Dense, MaxPooling2D, Conv2DTranspose

def conv_block(x, n_filter: int, kernel_size: tuple[int, int], dropout_rate = 0.1):
    x = Conv2D(n_filter, kernel_size=kernel_size, padding="same", data_format="channels_last", kernel_initializer="he_normal", activation='elu')(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(n_filter, kernel_size=kernel_size, padding="same", data_format="channels_last", kernel_initializer="he_normal", activation='elu')(x)
    x = Dropout(dropout_rate)(x)
    return x

def axile_attention(x, x_shape: tuple[int, int], mode="width"): # shape: ([Batch_size,] height, width)
    if mode == 'width':
        dense_layer = Dense(x_shape[0], activation="softmax")
        new_data = []
        for i in range(x_shape[0]):
            weights = dense_layer(x[:, i])

            l = tf.multiply(x[:, i], weights)
            l = tf.expand_dims(l, axis=1)

            new_data.append(l)
        
        return Concatenate(axis=1)(new_data)
    elif mode == 'height':
        x = tf.transpose(x, perm=[0, 2, 1])
        dense_layer = Dense(x_shape[1], activation="softmax")
        new_data = []
        for i in range(x_shape[1]):
            weights = dense_layer(x[:, i])

            l = tf.multiply(x[:, i], weights)
            l = tf.expand_dims(l, axis=1)

            new_data.append(l)
        return tf.transpose(Concatenate(axis=1)(new_data), perm=[0, 2, 1])

def make_unet_conv_LSTM_v2(input_shape: tuple[int, int, int, int], n_filters: int, kernel_size: tuple[int, int], out_seq: int, dropout_rate = 0.1):
    inp = Input(input_shape)

    # Downsampling
    down_conv_1 = [inp[:, i] for i in range(input_shape[0])]
    for i in range(input_shape[0]):
        down_conv_1[i] = conv_block(down_conv_1[i], n_filters, kernel_size, dropout_rate)
        down_conv_1[i] = MaxPooling2D((2, 2), strides=(2, 2))(down_conv_1[i])
    
    down_conv_2 = [down_conv_1[i] for i in range(input_shape[0])]
    for i in range(input_shape[0]):
        down_conv_2[i] = conv_block(down_conv_2[i], n_filters * 2, kernel_size, dropout_rate)
        down_conv_2[i] = MaxPooling2D((2, 2), strides=(2, 2))(down_conv_2[i])
    
    down_conv_3 = [down_conv_2[i] for i in range(input_shape[0])]
    for i in range(input_shape[0]):
        down_conv_3[i] = conv_block(down_conv_3[i], n_filters * 4, kernel_size, dropout_rate)
        down_conv_3[i] = MaxPooling2D((2, 2), strides=(2, 2))(down_conv_3[i])
    
    for i in range(input_shape[0]):
        down_conv_1[i] = tf.expand_dims(down_conv_1[i], axis=1)
        down_conv_2[i] = tf.expand_dims(down_conv_2[i], axis=1)
        down_conv_3[i] = tf.expand_dims(down_conv_3[i], axis=1)
    
    down_conv_1 = Concatenate(axis=1)(down_conv_1)
    down_conv_2 = Concatenate(axis=1)(down_conv_2)
    down_conv_3 = Concatenate(axis=1)(down_conv_3)

    # Encoder
    conv_lstm_1 = ConvLSTM2D(filters=n_filters, kernel_size=kernel_size, padding="same")(down_conv_1)
    conv_lstm_2 = ConvLSTM2D(filters=n_filters * 2, kernel_size=kernel_size, padding="same")(down_conv_2)
    conv_lstm_3 = ConvLSTM2D(filters=n_filters * 4, kernel_size=kernel_size, padding="same")(down_conv_3)
    conv_lstm_4 = ConvLSTM2D(filters=n_filters * 8, kernel_size=kernel_size, padding="same")(down_conv_3)

    # Decoder
    output = []
    for seq in range(out_seq):
        up_1 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding="same")(Concatenate(axis=-1)([conv_lstm_3, conv_lstm_4]))
        up_1 = Dropout(dropout_rate)(up_1)

        up_2 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding="same")(Concatenate(axis=-1)([conv_lstm_2, up_1]))
        up_2 = Dropout(dropout_rate)(up_2)

        up_3 = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding="same")(Concatenate(axis=-1)([conv_lstm_1, up_2]))
        up_3 = Dropout(dropout_rate)(up_3)

        out = Conv2D(1, (3, 3), padding="same", kernel_initializer="he_normal", activation="linear")(up_3)
        output.append(tf.expand_dims(out, axis=1))
    
    output = Concatenate(axis=1)(output)
    output = Reshape((out_seq, input_shape[1], input_shape[2]))(output)

    return Model(inputs = [inp], outputs = [output])