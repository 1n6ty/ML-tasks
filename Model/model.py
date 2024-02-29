import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, ConvLSTM2D, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, BatchNormalization, Layer, concatenate, Softmax

class AxileAttention(Layer):
    def __init__(self, input_shape, mode): # mode = "vertical" | "horizontal"
        super().__init__()

        self.mode = mode
        ind = 2 if mode == 'horizontal' else 1

        self.query = self.add_weight(
            initializer="random_normal",
            trainable=True,
            shape=[input_shape[0], input_shape[ind], input_shape[ind]]
        )
        self.key = self.add_weight(
            initializer="random_normal",
            trainable=True,
            shape=[input_shape[0], input_shape[ind], input_shape[ind]]
        )
        self.var = self.add_weight(
            initializer="random_normal",
            trainable=True,
            shape=[input_shape[0], input_shape[ind], input_shape[ind]]
        )

        self.query_b = self.add_weight(
            initializer="random_normal",
            trainable=True,
            shape=input_shape
        )
        self.key_b = self.add_weight(
            initializer="random_normal",
            trainable=True,
            shape=input_shape
        )
        self.var_b = self.add_weight(
            initializer="random_normal",
            trainable=True,
            shape=input_shape
        )

        self.softmax_l = Softmax(axis = -1 if mode == 'horizontal' else -2)

    def call(self, x):
        if self.mode == "horizontal":
            x = self.softmax_l(
                (tf.matmul(x, self.query) + self.query_b) * (tf.matmul(x, self.key) + self.key_b)
            ) * (tf.matmul(x, self.var) + self.var_b)
        else:
            x = self.softmax_l(
                (tf.matmul(self.query, x) + self.query_b) * (tf.matmul(self.key, x) + self.key_b)
            ) * (tf.matmul(self.var, x) + self.var_b)
        
        return x

def conv_block(x, n_filter: int, kernel_size: tuple[int, int], dropout_rate = 0.1):
    for _ in range(2):
        x = Conv2D(n_filter, kernel_size=kernel_size, padding="same", data_format="channels_last", kernel_initializer="he_normal", activation='elu')(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
    return x

def make_unet_conv_LSTM(input_shape: tuple[int, int, int, int], n_filters: list, kernel_size: tuple[int, int], out_seq: int, n_attention: int, dropout_rate = 0.1):
    inp = Input(input_shape)

    # Unet structure before LSTM
    unet_output = []
    for i in range(input_shape[0]):
        # Downsampling
        conv_1_1 = conv_block(inp[:, i], n_filters[0], kernel_size, dropout_rate)
        pool_1_1 = MaxPooling2D((2, 2))(conv_1_1)

        conv_2_1 = conv_block(pool_1_1, n_filters[1], kernel_size, dropout_rate)
        pool_2_1 = MaxPooling2D((2, 2))(conv_2_1)

        conv_3 = conv_block(pool_2_1, n_filters[2], kernel_size, dropout_rate)

        # Upsampling
        conv_2_2 = conv_block(
            concatenate([
                conv_2_1,
                Conv2DTranspose(n_filters[1], kernel_size, strides=(2, 2), padding="same", kernel_initializer="he_normal", activation="elu")(conv_3)
            ]), n_filters[1], kernel_size, dropout_rate
        )
        conv_2_1 = conv_block(
            concatenate([
                conv_1_1,
                Conv2DTranspose(n_filters[1], kernel_size, strides=(2, 2), padding="same", kernel_initializer="he_normal", activation="elu")(conv_2_2)
            ]), n_filters[0], kernel_size, dropout_rate
        )
        unet_output.append(tf.expand_dims(conv_2_1, axis=1))
    unet_output = concatenate(unet_output, axis=1)

    # LSTM
    LSTM_output = []
    for i in range(out_seq):
        LSTM_output.append(
            tf.squeeze(tf.expand_dims(ConvLSTM2D(1, kernel_size, padding="same", kernel_initializer="he_normal")(unet_output), axis=1), axis=[4])
        )
    LSTM_output = concatenate(LSTM_output, axis=1)
    
    # Axile Attention
    for i in range(n_attention):
        LSTM_output = AxileAttention((out_seq, input_shape[1], input_shape[2]), "horizontal")(LSTM_output)
        LSTM_output = AxileAttention((out_seq, input_shape[1], input_shape[2]), "vertical")(LSTM_output)

    return Model(inputs = [inp], outputs = [LSTM_output])

conv_LSTM = make_unet_conv_LSTM((4, 256, 256, 11), [64, 128, 256], (3, 3), 12, 4)
print(conv_LSTM.summary())