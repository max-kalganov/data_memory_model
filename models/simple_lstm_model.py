from typing import Tuple

import gin
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Bidirectional, LSTM, TimeDistributed
from dataset_generators.simple_memory_data_generator import SimpleMemoryDataGenerator


@gin.configurable
class SimpleLSTMModel:
    def __init__(self, seq_len: int, items_len: int, features_range: int, batch_size, train_test_split, train_val_split):
        self.items_len = items_len
        self.seq_len = seq_len
        self.features_range = features_range
        self.train_test_split = train_test_split
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.model = self.set_model()

    def set_model(self):
        input_layer = Input(shape=(self.seq_len, self.items_len, ))
        lstm1 = LSTM(units=64, activation='relu', return_sequences=True, input_shape=(self.seq_len, self.items_len))(input_layer)
        lstm2 = LSTM(units=64, activation='relu', return_sequences=True)(lstm1)
        lstm3 = LSTM(units=64, activation='relu', return_sequences=True)(lstm2)

        layer1_1 = TimeDistributed(Dense(units=64, activation='relu'))(lstm3)
        layer1_2 = TimeDistributed(Dense(units=64, activation='relu'))(layer1_1)
        y1_output = TimeDistributed(Dense(units=1, activation='sigmoid'), name='y1_output')(layer1_2)

        layer2_1 = TimeDistributed(Dense(units=64, activation='relu'))(lstm3)
        layer2_2 = TimeDistributed(Dense(units=64, activation='relu'))(layer2_1)
        layer2_3 = TimeDistributed(Dense(units=64, activation='relu'))(layer2_2)
        y2_output = TimeDistributed(Dense(units=self.features_range, activation='softmax'), name='y2_output')(layer2_3)

        # Define the model with the input layer and a list of output layers
        model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

        optimizer = tf.keras.optimizers.SGD(lr=0.001)
        model.compile(optimizer=optimizer,
                      loss={'y1_output': 'binary_crossentropy', 'y2_output': self.missed_value_loss},
                      metrics=['accuracy'])
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

    def train(self, train_X: np.array, train_Y: np.array, val_X: np.array, val_Y: np.array):
        history = self.model.fit(train_X, train_Y, epochs=50, batch_size=self.batch_size, validation_data=(val_X, val_Y))
        return history

    def missed_value_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        y_true_one_hot = tf.reshape(tf.one_hot(y_true, self.features_range),
                                    shape=(self.batch_size, self.seq_len, self.features_range))
        cross_entropy_res = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)

        y_true_labels = tf.reshape(y_true != -1, shape=(self.batch_size, -1))
        correct_cross_entropy_res = tf.math.reduce_sum(cross_entropy_res[y_true_labels])
        return correct_cross_entropy_res

    def evaluate(self, test_X: np.array, test_Y: np.array):
        print("Evaluating test results:")
        self.model.evaluate(test_X, test_Y, batch_size=self.batch_size)
