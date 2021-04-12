from typing import Tuple

import gin
import tensorflow as tf
from tensorflow import TensorShape
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import RNN, TimeDistributed, Dense, Dropout
from tensorflow.python.keras.layers.ops.core import dense
from tensorflow.python.training.tracking.data_structures import NoDependency

from models import SimpleLSTMModel


class ExtendedMemoryRNNCell(tf.keras.layers.Layer):

    def __init__(self, f_block_num_layers: int, f_block_units: int, s_block_num_layers: int,
                 s_block_units: int, state_shape, batch_size: int, **kwargs):
        self.f_block_units = f_block_units
        self.s_block_units = s_block_units
        self.f_block_num_layers = f_block_num_layers
        self.s_block_num_layers = s_block_num_layers
        self.batch_size = batch_size

        self.state_shape = state_shape
        self.state_size = NoDependency([f_block_units, TensorShape(state_shape)])
        super().__init__(**kwargs)

    def _set_weights(self, name: str, first_layer_dim: int, layers_dim: int, num_of_params: int, res_dim):
        last_dim = first_layer_dim
        for i in range(num_of_params-1):
            cur_name_base = f'{name}_{i}'
            kernel_name = f"{cur_name_base}_kernel"
            bias_name = f"{cur_name_base}_bias"
            vars(self)[kernel_name] = self.add_weight(kernel_name,
                                                      shape=[last_dim, layers_dim],
                                                      initializer='glorot_uniform',
                                                      trainable=True)
            vars(self)[bias_name] = self.add_weight(bias_name,
                                                    shape=[layers_dim, ],
                                                    initializer='zeros',
                                                    trainable=True)
            last_dim = layers_dim

        i = num_of_params - 1
        cur_name_base = f'{name}_{i}'
        kernel_name = f"{cur_name_base}_kernel"
        bias_name = f"{cur_name_base}_bias"
        vars(self)[kernel_name] = self.add_weight(kernel_name,
                                                  shape=[last_dim, res_dim],
                                                  initializer='glorot_uniform',
                                                  trainable=True)
        vars(self)[bias_name] = self.add_weight(bias_name,
                                                shape=[res_dim, ],
                                                initializer='zeros',
                                                trainable=True)

    def build(self, input_shape: Tuple):
        self._set_weights(name="first_block",
                          first_layer_dim=input_shape[-1] + self.state_shape[0] * self.state_shape[1],
                          layers_dim=self.f_block_units, num_of_params=self.f_block_num_layers,
                          res_dim=self.f_block_units)
        self._set_weights(name="second_block",
                          first_layer_dim=input_shape[-1]
                                          + self.state_shape[0] * self.state_shape[1]
                                          + self.f_block_units,
                          layers_dim=self.s_block_units, num_of_params=self.s_block_num_layers,
                          res_dim=self.state_shape[0] * self.state_shape[1])
        self.built = True

    def _call_block(self, inputs: tf.Tensor, state: tf.Tensor, name: str, num_of_params: int) -> tf.Tensor:
        x = tf.concat([inputs, tf.reshape(state, [self.batch_size, -1])], axis=1)

        for i in range(num_of_params):
            cur_name_base = f'{name}_{i}'
            kernel = vars(self)[f"{cur_name_base}_kernel"]
            bias = vars(self)[f"{cur_name_base}_bias"]
            x = dense(x, kernel, bias, tf.keras.activations.relu)
        return x

    def call(self, inputs, states):
        prev_state = states[1]

        f_block_result = self._call_block(inputs, prev_state, name="first_block", num_of_params=self.f_block_num_layers)
        f_block_result = tf.keras.activations.sigmoid(f_block_result)

        s_block_input = tf.concat([inputs, f_block_result], axis=1)
        s_block_result = self._call_block(s_block_input, prev_state, name="second_block",
                                          num_of_params=self.s_block_num_layers)
        s_block_result = tf.reshape(s_block_result, [s_block_result.shape[0], self.state_shape[0], -1])

        return f_block_result, [f_block_result, s_block_result]


@gin.configurable
class ExtendedMemoryModel(SimpleLSTMModel):
    def __init__(self, f_block_num_layers: int, f_block_units: int, s_block_num_layers: int,
                 s_block_units: int, state_shape, *args, **kwargs):
        self.f_block_units = f_block_units
        self.f_block_num_layers = f_block_num_layers
        self.s_block_units = s_block_units
        self.s_block_num_layers = s_block_num_layers
        self.state_shape = state_shape
        super().__init__(*args, **kwargs)

    def set_model(self):
        input_layer = Input(shape=(self.seq_len, self.items_len,))

        rnn_part = RNN(ExtendedMemoryRNNCell(f_block_num_layers=self.f_block_num_layers,
                                             f_block_units=self.f_block_units,
                                             s_block_num_layers=self.s_block_num_layers,
                                             s_block_units=self.s_block_units,
                                             state_shape=self.state_shape,
                                             batch_size=self.batch_size), return_sequences=True)(
            input_layer
        )

        layer1_1 = TimeDistributed(Dense(units=64, activation='relu'))(rnn_part)
        layer1_2 = TimeDistributed(Dense(units=64, activation='sigmoid'))(layer1_1)
        y1_output = TimeDistributed(Dense(units=1, activation='sigmoid'), name='y1_output')(layer1_2)

        layer2_1 = TimeDistributed(Dense(units=64, activation='relu'))(rnn_part)
        dropout_1 = Dropout(0.2)(layer2_1)
        layer2_2 = TimeDistributed(Dense(units=64, activation='relu'))(dropout_1)
        layer2_3 = TimeDistributed(Dense(units=64, activation='relu'))(layer2_2)
        y2_output = TimeDistributed(Dense(units=self.features_range, activation='softmax'), name='y2_output')(layer2_3)

        # Define the model with the input layer and a list of output layers
        model = tf.keras.models.Model(inputs=input_layer, outputs=[y1_output, y2_output])

        # optimizer = tf.keras.optimizers.SGD(lr=0.001)
        optimizer = tf.keras.optimizers.Adam(clipvalue=0.5)
        model.compile(optimizer=optimizer,
                      loss={'y1_output': 'binary_crossentropy', 'y2_output': self.missed_value_loss},
                      metrics={'y1_output': 'accuracy', 'y2_output': self.missed_value_acc})
        print(model.summary())
        return model
