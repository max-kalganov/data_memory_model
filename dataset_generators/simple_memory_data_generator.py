import random

import gin
import tensorflow as tf
import numpy as np
import tqdm as tqdm

from dataset_generators.data_generator import InputOutput_T
from dataset_generators.full_memory_data_generator import FullMemoryDataGenerator


class SimpleMemoryDataGenerator(FullMemoryDataGenerator):
    """All values except last is key, last value is value (like in dict)"""

    def _is_seen(self, v: np.array) -> bool:
        return any([(v[:-1] == seen_item[:-1]).all() for seen_item in self.seen_vectors])

    def _gen_seen_with_missed_item(self) -> InputOutput_T:
        missed_position = self.items_len - 1
        seen_item, label = self._gen_seen_item()
        label[1] = tf.one_hot(seen_item[missed_position], self.features_range).numpy()
        seen_item[missed_position] = self.missed_value
        return seen_item, label

    def _gen_single_changed_item(self):
        seen_item, label = self._gen_seen_item()
        change_position = np.random.randint(0, self.items_len-1)

        not_found = True
        select_step = random.choice([-1, 1])
        for i in range(self.features_range):
            seen_item[change_position] += select_step
            seen_item[change_position] %= self.features_range
            if not self._is_seen(seen_item):
                not_found = False
                break
        seen_item[-1] = (seen_item[-1] + select_step) % self.features_range
        return seen_item, not_found, change_position

    def _gen_changed_with_missed_item(self) -> InputOutput_T:
        not_found = True
        changed_item = np.array([])
        while not_found:
            changed_item, not_found, changed_index = self._gen_single_changed_item()

        missed_position = self.items_len - 1
        changed_item[missed_position] = self.missed_value
        return changed_item, np.array([0, None])


if __name__ == '__main__':
    gin.parse_config_file('configs/default.gin')
    a = SimpleMemoryDataGenerator(batch_size=20, num_of_batches=10)
    for i in tqdm.tqdm(a):
        print(i)
