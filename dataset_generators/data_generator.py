from abc import abstractmethod
from typing import Tuple, Optional, Any, List

import gin
import numpy as np

MutProb_T = Tuple[int, int, int, int, int]
InputOutput_T = Tuple[np.array, np.array]


@gin.configurable
class DataGenerator:
    def __init__(self, batch_size: int, num_of_batches: Optional[int], mut_prob: MutProb_T, missed_value: int,
                 seed: int):
        """ Implements data generator interface and some base functionality

        Args:
            batch_size: int - if 1 seen_vectors not overwritten, else seen_vectors are used only for a batch
            num_of_batches: Optional[int] - if None - no limit generator, else num_of_batches batches
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
        self.mut_prob = mut_prob
        self.num_of_batches = num_of_batches
        self.missed_value = missed_value
        np.random.seed(seed)

        self.seen_vectors: List[Any] = []

    @abstractmethod
    def _gen_new_item(self) -> InputOutput_T:
        pass

    @abstractmethod
    def _gen_seen_item(self) -> InputOutput_T:
        pass

    @abstractmethod
    def _gen_seen_with_missed_item(self) -> InputOutput_T:
        pass

    @abstractmethod
    def _gen_changed_item(self) -> InputOutput_T:
        pass

    @abstractmethod
    def _gen_changed_with_missed_item(self) -> InputOutput_T:
        pass

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
