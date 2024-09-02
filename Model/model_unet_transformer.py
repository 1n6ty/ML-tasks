import tensorflow as tf
from tensorflow import keras

from keras.regularizers import l2
from keras.layers import Conv2D, Dropout, BatchNormalization, Input, Layer, Softmax, UpSampling2D, Activation, MaxPool2D, concatenate
from keras.models import Model

def get_positional_encoding_matrix(rows, cols, dims, n):
    return tf.reshape(concatenate([
        tf.math.cos([i / (n ** (2 * d / dims)) for i in range(rows * cols)]) if (d % 2) else tf.math.sin([i / (n ** (2 * d / dims)) for i in range(rows * cols)]) for d in range(dims)
    ]), [rows, cols, dims])

class SelfAttention(Layer): # inputs: [Query, Key, Value]
    def __init__(self, dk_q, dv):
        super().__init__()

        self.dk_q = dk_q
        self.dv = dv
        self.softmax_layer = Softmax(axis = -1)

    def build(self, input_shape):
        self.Q_b = self.add_weight(
            shape = (1, input_shape[0][1], self.dk_q),
            initializer = "zeros",
            trainable = True,
            dtype = tf.float32
        )
        self.K_b = self.add_weight(
            shape = (1, input_shape[1][1], self.dk_q),
            initializer = "zeros",
            trainable = True,
            dtype = tf.float32
        )
        self.V_b = self.add_weight(
            shape = (1, input_shape[2][1], self.dv),
            initializer = "zeros",
            trainable = True,
            dtype = tf.float32
        )

        self.Q_w = self.add_weight(
            shape = (1, input_shape[0][-1], self.dk_q),
            initializer = "random_normal",
            trainable = True,
            dtype = tf.float32
        )
        self.K_w = self.add_weight(
            shape = (1, input_shape[1][-1], self.dk_q),
            initializer = "random_normal",
            trainable = True,
            dtype = tf.float32
        )
        self.V_w = self.add_weight(
            shape = (1, input_shape[2][-1], self.dv),
            initializer = "random_normal",
            trainable = True,
            dtype = tf.float32
        )

        super().build(input_shape)

    def call(self, inputs): # List[Q, K, V]
        Q = tf.matmul(inputs[0], self.Q_w) + self.Q_b
        K = tf.matmul(inputs[1], self.K_w) + self.K_b
        V = tf.matmul(inputs[2], self.V_w) + self.V_b

        S = self.softmax_layer(tf.matmul(Q, K, transpose_b = True))
        return tf.matmul(S, V)
    
    def compute_output_shape(self, input_shape):
        return input_shape[2]
    
class MHSA(Layer):
    def __init__(self, dk_q, dv, n):
        super().__init__()

        self.n = n
        self.Attention = SelfAttention(dk_q, dv)
    
    def build(self, input_shape):
        self.pe = tf.expand_dims(get_positional_encoding_matrix(*(input_shape[1:]), self.n), axis=0)

        super().build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)

        inputs += self.pe
        inputs = tf.reshape(inputs, [input_shape[0], input_shape[1] * input_shape[2], input_shape[3]])
        
        weighted_inp = self.Attention([inputs, inputs, inputs])
        return tf.reshape(weighted_inp, input_shape[:])
    
    def compute_output_shape(self, input_shape):
        return input_shape

class MCHA(Layer): # input: [skip connection, high-feature map]
    def __init__(self, dk_q, dv, n):
        super().__init__()

        self.n = n
        self.Attention = SelfAttention(dk_q, dv)
        self.conv_s = Conv2D(dk_q, (1, 1), padding = "same", kernel_initializer = "he_normal", activation = "linear")
        self.conv_y = Conv2D(dv, (1, 1), padding = "same", kernel_initializer = "he_normal", activation = "linear")
        self.conv_mcha = Conv2D(dv, (1, 1), padding = "same", kernel_initializer = "he_normal", activation = "linear")
        self.upSampling = UpSampling2D(interpolation = "nearest")
        self.sigmoid = Activation("sigmoid")
        self.batchNorm = BatchNormalization()

    def build(self, input_shape):
        self.pe_s = tf.expand_dims(get_positional_encoding_matrix(*(input_shape[0][1:]), self.n), axis=0)
        self.pe_y = tf.expand_dims(get_positional_encoding_matrix(*(input_shape[1][1:]), self.n), axis=0)

        super().build(input_shape)

    def call(self, inputs):
        input_shape = [tf.shape(inputs[0]), tf.shape(inputs[1])]
        S = inputs[0] + self.pe_s
        Y = inputs[1] + self.pe_y

        S_c = self.conv_s(S)
        Y_c = self.conv_y(Y)

        S_inp = tf.reshape(S_c, [input_shape[0][0], input_shape[0][1] * input_shape[0][2], input_shape[0][3]])
        Y_inp = tf.reshape(Y_c, [input_shape[1][0], input_shape[1][1] * input_shape[1][2], input_shape[1][3]])

        weighted_S = self.Attention([Y_inp, Y_inp, S_inp])
        weighted_S_inp = tf.reshape(weighted_S, input_shape[0])

        weighted_S_activate = self.sigmoid(self.batchNorm(self.conv_mcha(weighted_S_inp)))
        
        return tf.multiply(weighted_S_activate, inputs[0])
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3])

def conv_block(x, filters, kernel_size = (3, 3), dp_rate = 0.1, activation = "relu"):
    x = Conv2D(filters, kernel_size, kernel_initializer = "he_normal", padding = "same", activation = "linear")(x)
    x = Dropout(dp_rate)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(filters, kernel_size, kernel_initializer = "he_normal", padding = "same", activation = "linear")(x)
    x = Dropout(dp_rate)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def up_conv_block(x, filters, kernel_size = (1, 1), dp_rate = 0.1):
    x = Conv2D(filters, kernel_size, kernel_initializer = "he_normal", padding = "same", activation = "linear")(x)
    x = Dropout(dp_rate)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def make_unet_transformer(input_shape, filters, n = 10000, dp_rate = 0.1):
    inp = Input(shape = input_shape)

    conv_1 = conv_block(inp, filters[0], (3, 3), dp_rate, activation = "relu")
    max_p_1 = MaxPool2D((2, 2))(conv_1)

    conv_2 = conv_block(max_p_1, filters[1], (3, 3), dp_rate, activation = "relu")
    max_p_2 = MaxPool2D((2, 2))(conv_2)

    conv_3 = conv_block(max_p_2, filters[2], (3, 3), dp_rate, activation = "relu")
    max_p_3 = MaxPool2D((2, 2))(conv_3)

    conv_4 = conv_block(max_p_3, filters[3], (3, 3), dp_rate, activation = "relu")

    mhsa_out = MHSA(filters[3], filters[3], n)(conv_4)

    up_3_y = UpSampling2D((2, 2))(mhsa_out)
    up_3 = Conv2D(filters[2], (3, 3), padding = "same", kernel_initializer = "he_normal", activation = "linear")(up_3_y)
    mcha_3_out = MCHA(filters[3], filters[2], n)([conv_3, up_3_y])

    up_2_inp = conv_block(concatenate([mcha_3_out, up_3]), filters[2], (3, 3), dp_rate, activation = "relu")

    up_2_y = UpSampling2D((2, 2))(up_2_inp)
    up_2 = Conv2D(filters[1], (3, 3), padding = "same", kernel_initializer = "he_normal", activation = "linear")(up_2_y)
    mcha_2_out = MCHA(filters[2], filters[1], n)([conv_2, up_2_y])

    up_1_inp = conv_block(concatenate([mcha_2_out, up_2]), filters[1], (3, 3), dp_rate, activation = "relu")

    up_1_y = UpSampling2D((2, 2))(up_1_inp)
    up_1 = Conv2D(filters[0], (3, 3), padding = "same", kernel_initializer = "he_normal", activation = "linear")(up_1_y)
    mcha_1_out = MCHA(filters[1], filters[0], n)([conv_1, up_1_y])

    out_inp = conv_block(concatenate([mcha_1_out, up_1]), filters[0], (3, 3), dp_rate, activation = "relu")
    out = Conv2D(1, (1, 1), kernel_initializer = "he_normal", padding = "same", activation = "sigmoid")(out_inp)

    return Model(inp, out)

m = make_unet_transformer((256, 256, 1), [64, 128, 256, 512])
print(m.summary())
