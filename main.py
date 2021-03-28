import gin
from tqdm import tqdm

import models as m
import dataset_generators as dg


def test_simple_dataset():
    gin.parse_config_file('configs/default.gin')
    a = dg.SimpleMemoryDataGenerator()
    for batch_x, batch_y in tqdm(a):
        print(f"\nbatch x = {batch_x.shape}, batch y = {batch_y.shape}")


def run_model():
    if __name__ == '__main__':
        gin.parse_config_file('configs/default.gin')

        model = m.SimpleLSTMModel(path_to_weights='data/model_weights.h5')
        train_X, train_Y, val_X, val_Y, test_X, test_Y = model.get_train_val_test()
        model.train(train_X, train_Y, val_X, val_Y, test_X, test_Y)
        model.evaluate(test_X, test_Y)
        model.model.save_weights('data/model_weights.h5')


if __name__ == '__main__':
    run_model()
