import random
from typing import Optional, List

import gin
import numpy as np
import tqdm as tqdm

from dataset_generators.data_generator import DataGenerator, InputOutput_T


@gin.configurable
class FullMemoryDataGenerator(DataGenerator):
    def __init__(self, items_len: int, features_range: int, *args, **kwargs):
        """ Generates dataset with random values"""

        super().__init__(*args, **kwargs)
        self.items_len = items_len
        self.features_range = features_range

    def _init_seen_vectors(self) -> List[np.array]:
        return []

    def _is_seen(self, v: np.array) -> bool:
        return any([(v == seen_item).all() for seen_item in self.seen_vectors])

    def _gen_new_item(self) -> InputOutput_T:
        # TODO: check _check_if_only_one_correct when create new item
        not_found = True
        new_item = np.array([])
        while not_found:
            new_item = np.random.randint(0, self.features_range, self.items_len)
            not_found = self._is_seen(new_item)
        self.seen_vectors.append(new_item)
        return new_item, np.array([0, -1])

    def _gen_seen_item(self) -> InputOutput_T:
        seen_item = np.copy(random.choice(self.seen_vectors))
        return seen_item, np.array([1, -1])

    def _has_only_one_correct(self, missed_position: int, item: Optional[np.array] = None) -> bool:
        """Reads all seen items and checks if there are no duplicates without missed position"""
        indices_to_check = list(range(missed_position)) + list(range(missed_position + 1, self.items_len))
        clipped_arrays = [tuple(seen_item[indices_to_check]) for seen_item in self.seen_vectors]
        if item is not None:
            clipped_arrays += [tuple(item[indices_to_check])]
        unique_arrays = np.unique(clipped_arrays, axis=0)
        return len(clipped_arrays) == len(unique_arrays)

    def _gen_seen_with_missed_item(self) -> InputOutput_T:
        all_positions = list(range(self.items_len))
        missed_position = np.random.choice(all_positions)

        while not self._has_only_one_correct(missed_position) and len(all_positions) > 0:
            all_positions.remove(missed_position)
            missed_position = np.random.choice(all_positions)

        assert len(all_positions) != 0, f"not found position for missed_position"

        seen_item, label = self._gen_seen_item()
        label[1] = seen_item[missed_position]
        seen_item[missed_position] = self.missed_value
        return seen_item, label

    def _gen_single_changed_item(self):
        seen_item, label = self._gen_seen_item()
        change_position = np.random.randint(0, self.items_len)

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
        return seen_item, np.array([0, -1])

    def _gen_changed_with_missed_item(self) -> InputOutput_T:
        not_found = True
        changed_item = np.array([])
        changed_index = None
        while not_found:
            changed_item, not_found, changed_index = self._gen_single_changed_item()
        indexes = list(range(self.items_len))
        indexes.pop(changed_index)

        missed_position = np.random.choice(indexes)
        while not self._has_only_one_correct(missed_position, item=changed_item) and len(indexes) > 0:
            missed_position = np.random.choice(indexes)
            indexes.remove(missed_position)

        assert len(indexes) != 0, f"not found position for missed_position"
        changed_item[missed_position] = self.missed_value
        return changed_item, np.array([0, -1])


if __name__ == '__main__':
    gin.parse_config_file('configs/default.gin')
    a = FullMemoryDataGenerator(batch_size=20, num_of_batches=10)
    for i in tqdm.tqdm(a):
        print(i)
