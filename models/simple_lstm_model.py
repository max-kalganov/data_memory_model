from typing import Tuple

import gin
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Bidirectional, LSTM
from dataset_generators.simple_memory_data_generator import SimpleMemoryDataGenerator


@gin.configurable
class SimpleLSTMModel:
    def __init__(self, input_shape: Tuple, features_range: int, batch_size, train_test_split, train_val_split):
        self.input_shape = input_shape
        self.features_range = features_range
        self.train_test_split = train_test_split
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.model = self.set_model()

    def set_model(self):
        input_layer = Input(shape=self.input_shape)
        lstm1 = LSTM(units=64, activation='relu', return_sequences=True)(input_layer)
        lstm2 = LSTM(units=64, activation='relu', return_sequences=True)(lstm1)
        lstm3 = LSTM(units=64, activation='relu', return_sequences=True)(lstm2)

        layer1_1 = Dense(units=64, activation='relu')(lstm3)
        layer1_2 = Dense(units=64, activation='relu')(layer1_1)
        y1_output = Dense(units=1, activation='sigmoid', name='y1_output')(layer1_2)

        layer2_1 = Dense(units=64, activation='relu')(lstm3)
        layer2_2 = Dense(units=64, activation='relu')(layer2_1)
        layer2_3 = Dense(units=64, activation='relu')(layer2_2)
        y2_output = Dense(units=self.features_range, activation='softmax', name='y2_output')(layer2_3)

        # Define the model with the input layer and a list of output layers
        model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

        optimizer = tf.keras.optimizers.SGD(lr=0.001)
        model.compile(optimizer=optimizer,
                      loss={'y1_output': 'binary_crossentropy', 'y2_output': self.missed_value_loss})
        print(model.summary())
        return model

    def get_train_val_test(self) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
        generator = SimpleMemoryDataGenerator()
        full_dataset_X = []
        full_dataset_Y = []
        for X, Y in generator:
            full_dataset_X.append(X)
            full_dataset_Y.append(Y)

        full_train_X, test_X, full_train_Y, test_Y = train_test_split(full_dataset_X, full_dataset_Y,
                                                                      test_size=self.train_test_split)

        train_X, val_X, train_Y, val_Y = train_test_split(full_train_X, full_train_Y, test_size=self.train_val_split)

        return train_X, train_Y, val_X, val_Y, test_X, test_Y

    def train(self, train_X: np.array, train_Y: np.array, val_X: np.array, val_Y: np.array):

        history = self.model.fit(train_X, train_Y,
                                 epochs=50, batch_size=self.batch_size, validation_data=(val_X, val_Y))
        return history

    def missed_value_loss(self, y_true, y_pred):
        loss = 0
        if y_pred is not None:
            loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return loss

    def evaluate(self, test_X: np.array, test_Y: np.array):
        print(self.model.evaluate(test_X, test_Y))
