import random
from typing import Tuple, Optional, List, Set
import gin

import numpy as np

from dataset_generators.data_generator import DataGenerator, InputOutput_T

InputVectorConfig_T = Tuple[int, int, int, bool] # y = kx + b. tuple[k, b, starting x, increase values (true/false)


@gin.configurable
class SequenceMemoryDataGenerator(DataGenerator):
    def __init__(self, items_len: int, features_range: int, *args, **kwargs):
        """ Generates dataset with random values"""

        super().__init__(*args, **kwargs)
        self.items_len = items_len
        self.features_range = features_range

    def _init_seen_vectors(self) -> Set[InputVectorConfig_T]:
        return set()

    @staticmethod
    def _is_seen(v: np.array, all_seen: List[np.array]) -> bool:
        return any([(v == seen_item).all() for seen_item in all_seen])

    def _is_seen_config(self, v_config: InputVectorConfig_T) -> bool:
        return v_config in self.seen_configs

    def _get_vector_by_config(self, config: InputVectorConfig_T) -> np.array:
        raise NotImplemented("not implemented _get_vector_by_config")

    def _gen_config(self) -> InputVectorConfig_T:
        raise NotImplemented("not implemented _gen_config")

    def _gen_new_item(self) -> InputOutput_T:
        # TODO: check _check_if_only_one_correct when create new item
        not_found = True
        new_config = None
        while not_found:
            new_config = self._gen_config()
            not_found = self._is_seen_config(new_config)

        self.seen_vectors.add(new_config)
        new_item = self._get_vector_by_config(new_config)
        return new_item, np.array([0, -1])

    def _gen_seen_item(self) -> InputOutput_T:
        seen_config_ind = random.randint(0, len(self.seen_vectors))
        seen_config = list(self.seen_vectors)[seen_config_ind]
        seen_item = self._get_vector_by_config(seen_config)
        return seen_item, np.array([1, -1])

    def _has_only_one_correct(self, missed_position: int, item: Optional[np.array] = None) -> bool:
        """Reads all seen items and checks if there are no duplicates without missed position"""
        seen_items = [self._get_vector_by_config(seen_config) for seen_config in self.seen_vectors]

        indices_to_check = list(range(missed_position)) + list(range(missed_position + 1, self.items_len))
        clipped_arrays = [tuple(seen_item[indices_to_check]) for seen_item in seen_items]
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

    def _gen_unseen_with_missed_item(self) -> InputOutput_T:
        raise NotImplemented("not implemented _gen_unseen_with_missed_item")

    def _gen_single_changed_item(self, all_seen_items: List[np.array]):
        seen_item, label = self._gen_seen_item()
        change_position = np.random.randint(0, self.items_len)

        select_step = random.choice([-1, 1])
        seen_item[change_position] += select_step
        not_found = self._is_seen(seen_item, all_seen_items)
        return seen_item, not_found, change_position
