from typing import Sequence

import numpy as np

from trackit.data.source import TrackingDataset
from . import AuxFrameSampling
from .. import SamplingResult_Element, SiameseTrainingPairSamplingResult


class TubeletAuxFrameSampling(AuxFrameSampling):
    def __init__(self,
                 datasets: Sequence[TrackingDataset],
                 tubelet_size: int,
                 interval: int,
                 causal: bool):
        self.datasets = datasets
        self.tubelet_size = tubelet_size
        self.interval = interval
        self.causal = causal

    def __call__(self, siamese_training_pair: SiameseTrainingPairSamplingResult, rng_engine: np.random.Generator):
        auxiliary_frame_indices = []
        x = siamese_training_pair.x
        dataset = self.datasets[x.dataset_index]
        sequence = dataset[x.sequence_index]
        track = sequence.get_track_by_id(x.track_id)

        interval = self.interval
        if interval <= 0:
            interval = 1
        fps = sequence.get_fps()
        if fps is not None:
            standard_fps = 30
            interval = int(round(interval / fps * standard_fps))
            if interval < 1:
                interval = 1

        backward_pick = True
        if self.causal and x.frame_index < siamese_training_pair.z.frame_index:
            backward_pick = False

        for i in range(1, self.tubelet_size + 1):
            i = i * interval
            if backward_pick:
                aux_frame_index = x.frame_index - i
                if aux_frame_index < 0:
                    aux_frame_index = 0
            else:
                aux_frame_index = x.frame_index + i
                if aux_frame_index >= len(track):
                    aux_frame_index = len(track) - 1
            auxiliary_frame_indices.append(aux_frame_index)

        auxiliary_frame_indices.reverse()

        return SiameseTrainingPairSamplingResult(siamese_training_pair.z, siamese_training_pair.x,
                                                 siamese_training_pair.is_positive,
                                                 tuple(SamplingResult_Element(
                                                       x.dataset_index, x.sequence_index, x.track_id, frame_index)
                                                       for frame_index in auxiliary_frame_indices))
