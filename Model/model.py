import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Layer, ConvLSTM2D, Conv2D, Softmax, Lambda, BatchNormalization, Dropout, MaxPool2D, Conv2DTranspose, Reshape, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import KerasTensor

def get_PE_matrix(rows: int, cols: int, n=100000) -> tf.Tensor:
    """
    Generates Positional Encoding Matrix
    """
    common_col = tf.range(0, rows, 1, dtype=tf.float32)
    return tf.constant(
            tf.transpose(
                tf.concat([tf.expand_dims(tf.math.cos(common_col / (n ** ((d - 1) / cols))), axis=0) if d % 2 else tf.expand_dims(tf.math.sin(common_col / (n ** (d / cols))), axis=0) for d in range(cols)], axis=0),
                perm=[1, 0]
            )
        )

class CrossAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.softmax_layer = Softmax(axis=-1)

    def build(self, input_shape: list) -> None:
        self.sigma = tf.constant(tf.math.sqrt(tf.cast(input_shape[3], dtype=tf.float32)))
        
        self.q_weights = self.add_weight(
            shape=(input_shape[3], input_shape[3]),
            initializer="random_normal",
            trainable=True
        )

        self.k_weights = self.add_weight(
            shape=(input_shape[3], input_shape[3]),
            initializer="random_normal",
            trainable=True
        )

        self.v_weights = self.add_weight(
            shape=(input_shape[3], input_shape[3]),
            initializer="random_normal",
            trainable=True
        )

    @tf.function
    def apply_attention(self, inputs: list) -> tf.Tensor:
        """
            Apply Attention Layer

            Inputs
            -------

            `list`[query, key, value], `shape` of each should be (batch_size, features, dim)
        """
        Q_weighted = tf.matmul(inputs[0], self.q_weights)
        K_weighted = tf.matmul(inputs[1], self.k_weights)
        V_weighted = tf.matmul(inputs[2], self.v_weights)
        
        return tf.matmul(
            self.softmax_layer(
                tf.matmul(
                    Q_weighted, tf.transpose(K_weighted, perm=(0, 2, 1))
                )
            ) / self.sigma, V_weighted
        )

    @tf.function
    def call(self, inputs: KerasTensor) -> tf.Tensor:
        return tf.map_fn(
            lambda batch: tf.map_fn(
                lambda seq_1: tf.reduce_sum(
                    tf.map_fn(
                        lambda seq_2: self.apply_attention([seq_2, seq_2, seq_1]), elems=batch
                    ), axis=0
                ), elems=batch
            ), elems=inputs
        )
    
    def compute_output_shape(self, input_shape: list) -> tuple:
        return input_shape

def make_lstm_attention_text_model(input_shape: tuple[int], filters: list[int], lstm_filter: int, attention_heads = 1, predict=False) -> Model:
    """
        Makes LSTM encoder with cross attention for text model

        Parameters
        ----------

        input_shape:
            `shape` of the input (seq_len, encoded sentence dim)
        
        vector_len:
            length of the each feature vector on prediction
    """
    pe = get_PE_matrix(input_shape[1] * input_shape[2] / 64, lstm_filter)

    inp = Input(input_shape)
    
    conv_1 = [
        Conv2D(i, (3, 3), activation="elu", padding="same", kernel_regularizer=l2(1e-4)) for i in filters
    ]
    conv_2 = [
        Conv2D(i, (3, 3), activation="elu", padding="same", kernel_regularizer=l2(1e-4)) for i in filters
    ]
    max_pool = [
        MaxPool2D((2, 2)) for i in filters
    ]
    dropout_1 = [
        Dropout(0.1) for i in filters
    ]
    dropout_2 = [
        Dropout(0.1) for i in filters
    ]
    batch_normalization_1 = [
        BatchNormalization() for i in filters
    ]
    batch_normalization_2 = [
        BatchNormalization() for i in filters
    ]
    

    seq_inp = []
    for i in range(input_shape[0]):
        enc = Lambda(lambda x: x[i], output_shape=(1, *input_shape[1:]))(inp)
        enc = keras.ops.squeeze(enc)
        for i in range(len(filters)):
            enc = conv_1[i](enc)
            enc = dropout_2[i](enc)
            enc = batch_normalization_2[i](enc)

            enc = conv_2[i](enc)
            enc = dropout_2[i](enc)
            enc = batch_normalization_2[i](enc)

            enc = max_pool[i](enc)
        seq_inp.append(keras.ops.expand_dims(enc, axis=1))

    seq_inp = concatenate(seq_inp, axis=1)

    enc_lstm_seq = ConvLSTM2D(lstm_filter, (3, 3), padding="same", return_sequences=True)(seq_inp)

    seq_reshaped = Reshape((-1, int(input_shape[1] * input_shape[2] / 64), lstm_filter))(enc_lstm_seq)
    seq_reshaped = keras.ops.add(seq_reshaped, pe)

    seq_att = CrossAttention()(seq_reshaped)
    for h in range(attention_heads - 1):
        seq_att += CrossAttention()(seq_att)

    output_code = seq_att

    seq_out = Reshape((-1, int(input_shape[1] / 8), int(input_shape[2] / 8), lstm_filter))(seq_att)
    seq_out = Lambda(lambda x: x[::-1])(seq_out)
    
    seq_out = ConvLSTM2D(filters[-1], (3, 3), padding="same", return_sequences=True)(seq_out)
    seq_out = Lambda(lambda x: x[::-1])(seq_out)

    filters = filters[:-1][::-1] + [input_shape[-1]]

    transpose_1 = [
        Conv2DTranspose(i, (3, 3), strides=(2, 2), activation="elu", padding="same", kernel_regularizer=l2(1e-4)) for i in filters
    ]
    dropout_3 = [
        Dropout(0.1) for i in filters
    ]
    batch_normalization_3 = [
        BatchNormalization() for i in filters
    ]
    seq_out_list = []
    for i in range(input_shape[0]):
        dec = Lambda(lambda x: x[i], output_shape=(1, int(input_shape[1] / 8), int(input_shape[2] / 8), lstm_filter))(seq_out)
        dec = keras.ops.squeeze(dec)
        for i in range(len(filters)):
            dec = transpose_1[i](dec)
            dec = dropout_3[i](dec)
            dec = batch_normalization_3[i](dec)

        seq_out_list.append(keras.ops.expand_dims(dec, axis=1))

    seq_out_list = concatenate(seq_out_list, axis=1)

    if predict:
        return Model(inp, output_code)
    else:
        return Model(inp, seq_out_list)