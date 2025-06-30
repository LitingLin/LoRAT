import numpy as np
from typing import Tuple
from trackit.data.source import TrackingDataset


class DistractorGenerator:
    def __init__(self, dataset: TrackingDataset):
        category_ids = dataset.get_all_category_id_name_map()
        assert category_ids is not None, "distractor negative sample generating require all dataset has object category annotated"
        category_ids = category_ids.keys()
        category_id_track_map = {category_id: [] for category_id in category_ids}

        for index_of_sequence, sequence in enumerate(dataset):
            for index_of_track in range(sequence.get_number_of_tracks()):
                track = sequence.get_object_track_by_index(index_of_track)
                category_id_track_map[track.get_category_id()].append((index_of_sequence, index_of_track))

        for category_id in list(category_id_track_map.keys()):
            category_id_track_map[category_id] = tuple(category_id_track_map[category_id])
            if len(category_id_track_map[category_id]) == 0:
                del category_id_track_map[category_id]

        self.category_id_track_map = category_id_track_map

    def __call__(self, category_id: int, rng_engine: np.random.Generator) -> Tuple[int, int]:
        return self.category_id_track_map[rng_engine.integers(len(self.category_id_track_map[category_id]))]
