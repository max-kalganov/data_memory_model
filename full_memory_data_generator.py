import random
from typing import Tuple, Optional, List

import gin
import tensorflow as tf
import numpy as np
import tqdm as tqdm

MutProb_T = Tuple[int, int, int, int, int]
InputOutput_T = Tuple[np.array, np.array]

# tf.keras.utils.Sequence


@gin.configurable()
class FullMemoryDataGenerator:
    def __init__(self, batch_size: int, num_of_batches: Optional[int], items_len: int, features_range: int,
                 mut_prob: MutProb_T, missed_value: int, seed: int):
        """ Generates dataset

        self.seen_values creates one item by default

        Args:
            batch_size: int - if 1 seen_vectors not overwritten, else seen_vectors are used only for a batch
            num_of_batches: Optional[int] - if None - no limit generator, else seq_len batches
            mut_prob: Tuple[int, int, int, int, int] - Probabilities for running methods:
                        self._gen_new_item,
                        self._gen_seen_item,
                        self._gen_seen_with_missed_item,
                        self._gen_changed_item,
                        self._gen_changed_with_missed_item
        """

        assert sum(mut_prob) == 1, "wrong mut prob"
        assert batch_size >= 1, "wrong batch size"
        self.batch_size = batch_size
        self.items_len = items_len
        self.features_range = features_range
        self.mut_prob = mut_prob
        self.num_of_batches = num_of_batches
        self.missed_value = missed_value
        np.random.seed(seed)

        self.seen_vectors: List[np.array] = []

    def _is_seen(self, v: np.array) -> bool:
        return any([(v == seen_item).all() for seen_item in self.seen_vectors])

    def _gen_new_item(self) -> InputOutput_T:
        not_found = True
        new_item = np.array([])
        while not_found:
            new_item = np.random.randint(0, self.features_range, self.items_len)
            not_found = self._is_seen(new_item)
        self.seen_vectors.append(new_item)
        return new_item, np.array([0, None])

    def _gen_seen_item(self) -> InputOutput_T:
        seen_item = np.copy(random.choice(self.seen_vectors))
        return seen_item, np.array([1, None])

    def _gen_seen_with_missed_item(self) -> InputOutput_T:
        # TODO: check if there is only one correct answer
        seen_item, label = self._gen_seen_item()
        missed_position = np.random.randint(0, len(seen_item))
        label[1] = tf.one_hot(seen_item[missed_position], self.features_range)
        seen_item[missed_position] = self.missed_value
        return seen_item, label

    def _gen_single_changed_item(self):
        seen_item, label = self._gen_seen_item()
        change_position = np.random.randint(0, len(seen_item))

        not_found = True
        select_step = random.choice([-1, 1])
        for i in range(self.features_range):
            seen_item[change_position] += select_step
            seen_item[change_position] %= self.features_range
            if not self._is_seen(seen_item):
                not_found = False
                break
        return seen_item, not_found, change_position

    def _gen_changed_item(self) -> InputOutput_T:
        not_found = True
        seen_item = np.array([])
        while not_found:
            seen_item, not_found, _ = self._gen_single_changed_item()
        self.seen_vectors.append(seen_item)
        return seen_item, np.array([0, None])

    def _gen_changed_with_missed_item(self) -> InputOutput_T:
        # TODO: check if there is only one correct answer
        not_found = True
        seen_item = np.array([])
        changed_index = None
        while not_found:
            seen_item, not_found, changed_index = self._gen_single_changed_item()
        indexes = list(range(self.items_len))
        indexes.pop(changed_index)
        missed_position = random.choice(indexes)
        seen_item[missed_position] = self.missed_value
        return seen_item, np.array([0, None])

    def __iter__(self):
        while self.num_of_batches is None or self.num_of_batches != 0:
            yield self.__next__()
            self.num_of_batches = self.num_of_batches if self.num_of_batches is None else self.num_of_batches - 1

    def __next__(self):
        batch_inputs = []
        batch_labels = []

        num_of_generated_items = self.batch_size
        if self.batch_size != 1:
            self.seen_vectors = []
            first_input, first_label = self._gen_new_item()
            batch_inputs.append(first_input)
            batch_labels.append(first_label)
            num_of_generated_items -= 1

        for batch in range(num_of_generated_items):
            input, label = np.random.choice([
                self._gen_new_item,
                self._gen_seen_item,
                self._gen_seen_with_missed_item,
                self._gen_changed_item,
                self._gen_changed_with_missed_item
            ], p=self.mut_prob)()
            batch_inputs.append(input)
            batch_labels.append(label)
        return np.stack(batch_inputs), np.stack(batch_labels)


if __name__ == '__main__':
    gin.parse_config_file('configs/default_config.gin')
    a = FullMemoryDataGenerator(batch_size=50, num_of_batches=2000)
    for i in tqdm.tqdm(a):
        pass
        # print(i)