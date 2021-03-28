import os
from datetime import datetime
from typing import Tuple, Optional

import gin
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, LSTM, TimeDistributed, Dropout

from callbacks.samples_visualizer import SampleVisCallback
from dataset_generators.simple_memory_data_generator import SimpleMemoryDataGenerator


@gin.configurable
class SimpleLSTMModel:
    def __init__(self, seq_len: int, items_len: int, features_range: int, batch_size: int,  num_of_epochs: int,
                 train_test_split: float, train_val_split: float, path_to_weights: Optional[str] = None):
        self.items_len = items_len
        self.seq_len = seq_len
        self.features_range = features_range
        self.train_test_split = train_test_split
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.model = self.set_model()
        if path_to_weights:
            self.model.load_weights(path_to_weights)
            print(f"loaded weights from {path_to_weights}")

        logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.tb_callback = tf.keras.callbacks.TensorBoard(logdir)

    def set_model(self):
        input_layer = Input(shape=(self.seq_len, self.items_len,))
        lstm1 = LSTM(units=64,
                     activation='relu',
                     return_sequences=True,
                     input_shape=(self.seq_len, self.items_len))(input_layer)

        lstm2 = LSTM(units=64, activation='relu', return_sequences=True)(lstm1)
        lstm3 = LSTM(units=64, activation='relu', return_sequences=True)(lstm2)

        layer1_1 = TimeDistributed(Dense(units=64, activation='relu'))(lstm3)
        layer1_2 = TimeDistributed(Dense(units=64, activation='sigmoid'))(layer1_1)
        y1_output = TimeDistributed(Dense(units=1, activation='sigmoid'), name='y1_output')(layer1_2)

        layer2_1 = TimeDistributed(Dense(units=64, activation='relu'))(lstm3)
        dropout_1 = Dropout(0.2)(layer2_1)
        layer2_2 = TimeDistributed(Dense(units=64, activation='relu'))(dropout_1)
        layer2_3 = TimeDistributed(Dense(units=64, activation='relu'))(layer2_2)
        y2_output = TimeDistributed(Dense(units=self.features_range, activation='softmax'), name='y2_output')(layer2_3)

        # Define the model with the input layer and a list of output layers
        model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

        # optimizer = tf.keras.optimizers.SGD(lr=0.001)
        optimizer = tf.keras.optimizers.Adam(clipvalue=0.5)
        model.compile(optimizer=optimizer,
                      loss={'y1_output': 'binary_crossentropy', 'y2_output': self.missed_value_loss},
                      metrics={'y1_output': 'accuracy', 'y2_output': self.missed_value_acc})
        print(model.summary())
        return model

    def get_train_val_test(self) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
        generator = SimpleMemoryDataGenerator()
        full_dataset_X = []
        full_dataset_Y_1 = []
        full_dataset_Y_2 = []

        for X, Y in generator:
            full_dataset_X.append(X)
            full_dataset_Y_1.append(Y[0])
            full_dataset_Y_2.append(Y[1])

        full_dataset_X = np.concatenate(full_dataset_X)
        full_dataset_Y_1 = np.concatenate(full_dataset_Y_1)
        full_dataset_Y_2 = np.concatenate(full_dataset_Y_2)

        full_train_X, test_X, full_train_Y_1, test_Y_1, full_train_Y_2, test_Y_2 = train_test_split(
            full_dataset_X,
            full_dataset_Y_1,
            full_dataset_Y_2,
            test_size=self.train_test_split)

        train_X, val_X, train_Y_1, val_Y_1, train_Y_2, val_Y_2 = train_test_split(
            full_train_X,
            full_train_Y_1,
            full_train_Y_2,
            test_size=self.train_val_split)

        return train_X, (train_Y_1, train_Y_2), val_X, (val_Y_1, val_Y_2), test_X, (test_Y_1, test_Y_2)

    def train(self,
              train_X: np.array, train_Y: np.array,
              val_X: np.array, val_Y: np.array,
              test_X: np.array, test_Y: np.array):
        history = self.model.fit(train_X, train_Y, epochs=self.num_of_epochs, batch_size=self.batch_size,
                                 validation_data=(val_X, val_Y),
                                 callbacks=[self.tb_callback, SampleVisCallback(test_X, test_Y)])
        return history

    def missed_value_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        y_true_one_hot = tf.reshape(tf.one_hot(y_true, self.features_range),
                                    shape=(self.batch_size, self.seq_len, self.features_range))
        cross_entropy_res = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)

        y_true_labels = tf.reshape(y_true != -1, shape=(self.batch_size, -1))
        correct_cross_entropy_res = tf.math.reduce_sum(cross_entropy_res[y_true_labels])
        return correct_cross_entropy_res

    def missed_value_acc(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_true_flags = tf.cast(tf.reshape(y_true != -1, shape=(self.batch_size, -1)), tf.int32)
        res_diff = tf.dtypes.cast(y_pred_labels == tf.cast(y_true, tf.int64), tf.int32)
        res = y_true_flags * res_diff
        return tf.reduce_sum(res) / tf.reduce_sum(y_true_flags)

    def evaluate(self, test_X: np.array, test_Y: np.array):
        print("Evaluating test results:")
        self.model.evaluate(test_X, test_Y, batch_size=self.batch_size, callbacks=[self.tb_callback])
