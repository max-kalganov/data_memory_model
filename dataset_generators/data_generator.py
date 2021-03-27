from abc import abstractmethod
from typing import Tuple, Optional, Any, List

import gin
import numpy as np

MutProb_T = Tuple[float, float, float, float, float, float]
InputOutput_T = Tuple[np.array, np.array]


@gin.configurable
class DataGenerator:
    def __init__(self, batch_size: int, seq_len: int, num_of_batches: Optional[int], mut_prob: MutProb_T, missed_value: int,
                 seed: int):
        """ Implements data generator interface and some base functionality
        Args:
            seq_len: int - if 1 seen_vectors not overwritten, else seen_vectors are used only for a batch
            batch_size: int - batch size
            num_of_batches: Optional[int] - if None - no limit generator, else num of batches
            mut_prob: Tuple[float, float, float, float, float, float] - Probabilities for running methods:
                        self._gen_new_item,
                        self._gen_seen_item,
                        self._gen_seen_with_missed_item,
                        self._gen_unseen_with_missed_item,
                        self._gen_changed_item,
                        self._gen_changed_with_missed_item
        """

        assert sum(mut_prob) == 1, "wrong mut prob"
        assert batch_size >= 1, "wrong batch size"
        assert num_of_batches >= 1 or num_of_batches is None, "wrong num of batches"
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.mut_prob = mut_prob
        self.num_of_batches = num_of_batches
        self.missed_value = missed_value
        np.random.seed(seed)

        self.seen_vectors: Any = self._init_seen_vectors()

    @abstractmethod
    def _init_seen_vectors(self) -> Any:
        pass

    @abstractmethod
    def _gen_new_item(self) -> InputOutput_T:
        pass

    @abstractmethod
    def _gen_seen_item(self) -> InputOutput_T:
        pass

    @abstractmethod
    def _gen_seen_with_missed_item(self) -> InputOutput_T:
        pass

    def _gen_unseen_with_missed_item(self) -> InputOutput_T:
        return None, None

    def _gen_changed_item(self) -> InputOutput_T:
        return None, None

    def _gen_changed_with_missed_item(self) -> InputOutput_T:
        return None, None

    def __iter__(self):
        while self.num_of_batches != 0:
            yield self.__next__()
            if self.num_of_batches is not None:
                self.num_of_batches -= 1

    def _next_seq(self):
        seq_inputs = []
        seq_labels = []

        num_of_generated_items = self.seq_len
        if self.seq_len != 1 or self.seen_vectors == self._init_seen_vectors():
            self.seen_vectors = self._init_seen_vectors()
            first_input, first_label = self._gen_new_item()
            seq_inputs.append(first_input)
            seq_labels.append(first_label)
            num_of_generated_items -= 1

        already_generated = 0
        while already_generated < num_of_generated_items:
            input, label = np.random.choice([
                self._gen_new_item,
                self._gen_seen_item,
                self._gen_seen_with_missed_item,
                self._gen_unseen_with_missed_item,
                self._gen_changed_item,
                self._gen_changed_with_missed_item
            ], p=self.mut_prob)()
            if input is None or label is None:
                continue
            seq_inputs.append(input)
            seq_labels.append(label)
            already_generated += 1
        return np.stack(seq_inputs).astype('int64'), np.stack(seq_labels).astype('int64')

    def __next__(self):
        batch_x = []
        batch_y = []
        for i in range(self.batch_size):
            seq_x, seq_y = self._next_seq()
            batch_x.append(seq_x)
            batch_y.append(seq_y)
        batch_y_res = np.stack(batch_y).astype('int64')
        return np.stack(batch_x).astype('int64'), (batch_y_res[:, :, 0], batch_y_res[:, :, 1])
