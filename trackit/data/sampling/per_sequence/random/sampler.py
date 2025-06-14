from typing import Optional
import numpy as np
import copy


class RandomSequencePicker:
    def __init__(self, dataset_lengths, datasets_sampling_weights: np.ndarray, total_size: int, seed: Optional[int], init: bool = True):
        self.dataset_lengths = dataset_lengths
        dataset_sampled_size_list = datasets_sampling_weights * total_size
        dataset_sampled_size_list = dataset_sampled_size_list.astype(np.int64)
        dataset_sampled_size_list[-1] = total_size - sum(dataset_sampled_size_list[:len(dataset_sampled_size_list) - 1])
        self.dataset_sampled_size_list = dataset_sampled_size_list
        self.seed = seed
        self.total_size = total_size
        self.rng_engine = np.random.default_rng(self.seed)
        if init:
            self.shuffle()

    def __len__(self):
        return self.total_size

    def reset(self):
        self.rng_engine = np.random.default_rng(self.seed)
        self.shuffle()

    def shuffle(self):
        self._last_rng_state = self.rng_engine.bit_generator.state
        dataset_indices = []
        sequence_indices = []
        for index, (dataset_num_sequences, dataset_sampled_size) in enumerate(zip(self.dataset_lengths, self.dataset_sampled_size_list)):
            current_dataset_indices = []
            indices = np.arange(dataset_num_sequences, dtype=np.int64)
            current_size = 0
            while current_size < dataset_sampled_size:
                current_indices = copy.copy(indices)
                self.rng_engine.shuffle(current_indices)
                current_dataset_indices.append(current_indices)
                current_size += len(indices)
            current_dataset_indices = np.concatenate(current_dataset_indices)
            current_dataset_indices = current_dataset_indices[: dataset_sampled_size]
            dataset_indices.append(np.full((dataset_sampled_size,), index, dtype=np.int64))
            sequence_indices.append(current_dataset_indices)
        dataset_indices = np.concatenate(dataset_indices)
        sequence_indices = np.concatenate(sequence_indices)
        shuffle_indices = self.rng_engine.permutation(len(dataset_indices))
        dataset_indices = dataset_indices[shuffle_indices]
        sequence_indices = sequence_indices[shuffle_indices]
        self.dataset_indices = dataset_indices
        self.sequence_indices = sequence_indices

    def __getitem__(self, index: int):
        return self.dataset_indices[index], self.sequence_indices[index]

    def get_state(self):
        assert isinstance(self._last_rng_state, dict), f"Expected dict, got {type(self._last_rng_state)}. value:\n{str(self._last_rng_state)}"
        return self._last_rng_state

    def set_state(self, rng_state):
        assert isinstance(rng_state, dict), f"Expected dict, got {type(rng_state)}. value:\n{str(rng_state)}"
        self.rng_engine.bit_generator.state = rng_state
        self.shuffle()
