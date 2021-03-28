import logging

import gin
import tensorflow as tf
import numpy as np


@gin.configurable
class SampleVisCallback(tf.keras.callbacks.Callback):
    def __init__(self, inputs, ground_truth, n_samples):
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.n_samples = n_samples

    def process_y_pred(self, y_pred_1: np.array, y_pred_2: np.array, ground_truth_2: np.array):
        y_pred_labels = np.argmax(y_pred_2, axis=-1)
        y_pred_labels[ground_truth_2 == -1] = -1
        return (y_pred_1.reshape(-1) > 0.5).astype(int), y_pred_labels

    def on_epoch_end(self, epoch, logs=None):
        # Randomly sample data
        x_test = self.inputs[-self.n_samples:, :, :]
        y_test = self.ground_truth[0][-self.n_samples:, :], self.ground_truth[1][-self.n_samples:, :]
        predictions = self.model.predict(x_test)
        for i in range(self.n_samples):
            print(f"x_test: {x_test[i]}")
            cur_pred = self.process_y_pred(predictions[0][i], predictions[1][i], y_test[1][i])
            print(f"y_true_output1: {y_test[0][i]}")
            print(f"y_pred_output1: {cur_pred[0]}")
            print(f"y_true_output2: {y_test[1][i]}")
            print(f"y_pred_output2: {cur_pred[1]}")
