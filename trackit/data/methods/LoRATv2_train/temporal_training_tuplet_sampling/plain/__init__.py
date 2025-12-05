from typing import Sequence, Optional, Tuple
import numpy as np

from ....siamese_tracker_train.siamese_training_pair_sampling._algos import _get_random_track
from trackit.data.source import TrackingDataset
from trackit.data.sampling.per_sequence import RandomAccessiblePerSequenceSampler
from .. import TemporalTrackerTrainingSamples_SamplingResult, SamplingResult_Element


class PlainTemporalTrainingSamplesSampler:
    def __init__(self, num_templates: int, num_search_regions: int,
                 datasets: Sequence[TrackingDataset],
                 dataset_weights: np.ndarray,
                 sequence_picker: Optional[RandomAccessiblePerSequenceSampler],
                 sampling_frame_range: int,
                 sampling_frame_range_adjust_according_to_sequence_fps: bool,
                 sampling_frame_range_auto_extend_step: int,
                 sampling_frame_range_auto_extend_max_retry_count: int,
                 sampling_disable_frame_range_constraint_if_search_frame_not_found: bool):
        self.num_templates = num_templates
        self.num_search_regions = num_search_regions
        self.datasets = datasets

        self.dataset_weights = self._normalize_weights(dataset_weights)

        self.sequence_picker = sequence_picker
        self.siamese_sampling_frame_range = sampling_frame_range
        self.siamese_sampling_frame_range_adjust_according_to_sequence_fps = sampling_frame_range_adjust_according_to_sequence_fps
        self.siamese_sampling_frame_range_auto_extend_step = sampling_frame_range_auto_extend_step
        self.siamese_sampling_frame_range_auto_extend_max_retry_count = sampling_frame_range_auto_extend_max_retry_count
        self.siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found = sampling_disable_frame_range_constraint_if_search_frame_not_found

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1."""
        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()
        return weights

    def __call__(self, index: Optional[int], rng_engine: np.random.Generator) -> TemporalTrackerTrainingSamples_SamplingResult:
        if index is not None:
            assert self.sequence_picker is not None, 'Sequence picker is required for indexed sampling'
            dataset_index, sequence_index = self.sequence_picker[index]
            dataset = self.datasets[dataset_index]
            sequence = dataset[sequence_index]
        else:
            dataset_index = rng_engine.choice(np.arange(len(self.datasets)), p=self.dataset_weights)
            dataset = self.datasets[dataset_index]
            sequence_index = rng_engine.integers(0, len(dataset))
            sequence = dataset[sequence_index]

        sequence_fps = sequence.get_fps()
        reference_sequence_fps = 30

        track = _get_random_track(sequence, rng_engine)
        siamese_sampling_frame_range = self.siamese_sampling_frame_range
        if self.siamese_sampling_frame_range_adjust_according_to_sequence_fps and sequence_fps is not None:
            siamese_sampling_frame_range = int(siamese_sampling_frame_range * sequence_fps / reference_sequence_fps)
            if siamese_sampling_frame_range < 1:
                siamese_sampling_frame_range = 1
        frame_indices = _do_training_sample_sampling(self.num_templates + self.num_search_regions,
                                                     len(track), siamese_sampling_frame_range,
                                                     track.get_all_object_existence_flag(),
                                                     rng_engine,
                                                     self.siamese_sampling_frame_range_auto_extend_step,
                                                     self.siamese_sampling_frame_range_auto_extend_max_retry_count,
                                                     self.siamese_sampling_disable_frame_range_constraint_if_search_frame_not_found)
        assert frame_indices is not None, "sequence should has at least one valid (object visible) frame."
        frame_indices = np.sort(frame_indices)
        if rng_engine.random() < 0.5:
            frame_indices = frame_indices[::-1]
        template_frame_indices = frame_indices[: self.num_templates]
        search_region_frame_indices = frame_indices[self.num_templates:]

        return TemporalTrackerTrainingSamples_SamplingResult(
            tuple(SamplingResult_Element(dataset_index, sequence_index, track.get_object_id(), index) for index in
                  template_frame_indices),
            tuple(SamplingResult_Element(dataset_index, sequence_index, track.get_object_id(), index) for index in
                  search_region_frame_indices),
            tuple(True for _ in range(self.num_search_regions))
        )


def get_frame_ids_order(visible: np.ndarray, num_template_frames: int, num_search_frames: int, max_gap: int,
                        rng_engine: np.random.Generator):
    # get template and search ids in an 'order' manner, the template and search regions are arranged in chronological order
    frame_ids = []
    gap_increase = 0
    while (None in frame_ids) or (len(frame_ids)==0):
        base_frame_id = _sample_visible_ids(visible, rng_engine, num_ids=1, min_id=0,
                                                 max_id=len(visible))
        frame_ids = _sample_visible_ids(visible, rng_engine, num_ids=num_template_frames + num_search_frames,
                                                  min_id=base_frame_id[0] - max_gap - gap_increase,
                                                  max_id=base_frame_id[0] + max_gap + gap_increase)
        if (frame_ids is None) or (None in frame_ids):
            gap_increase += 5
            if gap_increase > 1000:
                print("too large frame gap, check the sampler, current gap: " + str(gap_increase))
                return None
            continue
        frame_ids = np.sort(frame_ids)
        if rng_engine.random() < 0.5:
            frame_ids = frame_ids[::-1]
        template_frame_ids = frame_ids[0: num_template_frames]
        search_frame_ids = frame_ids[num_template_frames:]
        # Increase gap until a frame is found
        gap_increase += 5
        if gap_increase > 1000:
            print("too large frame gap, check the sampler, current gap: " + str(gap_increase))
            return None
    return template_frame_ids, search_frame_ids


def _sample_visible_ids(visible, rng_engine: np.random.Generator, num_ids=1, min_id=None, max_id=None,
                        allow_invisible=False, force_invisible=False):
    """ Samples num_ids frames between min_id and max_id for which target is visible

    args:
        visible - 1d Tensor indicating whether target is visible for each frame
        num_ids - number of frames to be samples
        min_id - Minimum allowed frame number
        max_id - Maximum allowed frame number

    returns:
        list - List of sampled frame numbers. None if not sufficient visible frames could be found.
    """
    if num_ids == 0:
        return []
    if min_id is None or min_id < 0:
        min_id = 0
    if max_id is None or max_id > len(visible):
        max_id = len(visible)
    # get valid ids
    if force_invisible:
        valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
    else:
        if allow_invisible:
            valid_ids = [i for i in range(min_id, max_id)]
        else:
            valid_ids = [i for i in range(min_id, max_id) if visible[i]]

    # No visible ids
    if len(valid_ids) == 0:
        return None

    return rng_engine.choice(valid_ids, size=num_ids)



def _do_training_sample_sampling(num_samples: int,
                             length: int,
                             frame_range: Optional[int], mask: np.ndarray = None,
                             rng_engine: np.random.Generator = np.random.default_rng(),
                             frame_range_auto_extend_step: int = 5,
                             frame_range_auto_extend_max_retry_count: int = 10,
                             disable_frame_range_if_not_found: bool = False):
    '''
    :param length: length of the sequence
    :param frame_range:
         when sampling_method == causal:
            z_index ---------------> x_index
                    max frame_range
         when sampling_method == interval:
            x_index <-------------- z_index --------------> x_index
                    max frame_range         max frame_range
    :param mask: validity mask
    :param sampling_method: causal or interval
    :param rng_engine: random number generator (numpy)
    :return:
        Union(Tuple(int, int), int)
            Tuple(int, int): (z_index, x_index), when a match search frame is found
            int: z_index, when no match search frame is found
    '''
    assert frame_range > 0 or frame_range is None
    z_index = _get_random_index(length, mask, rng_engine)

    if z_index is None:
        return None

    if length == 1:
        return np.full(num_samples, z_index)

    for _ in range(frame_range_auto_extend_max_retry_count + 1):
        x_candidate_indices, frame_range_extendable = _get_search_frame_candidates(z_index, length, frame_range, mask)

        if x_candidate_indices is not None:
            return rng_engine.choice(x_candidate_indices, num_samples)

        if not frame_range_extendable:
            break

        if frame_range_auto_extend_step == 0:
            break
        frame_range += frame_range_auto_extend_step

    if disable_frame_range_if_not_found and frame_range is not None:
        x_candidate_indices, _ = _get_search_frame_candidates(z_index, length, None, mask)
        if x_candidate_indices is not None:
            return rng_engine.choice(x_candidate_indices, num_samples)

    return None


def _get_search_frame_candidates(z_index: int, length: int, frame_range: Optional[int], mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], bool]:
    frame_range_extendable = False
    if frame_range is not None:
        x_frame_begin = z_index - frame_range
        if x_frame_begin > 0:
            frame_range_extendable = True
        else:
            x_frame_begin = 0
        x_frame_end = z_index + frame_range + 1
        if x_frame_end >= length:
            x_frame_end = length
        else:
            frame_range_extendable = True
    else:
        x_frame_begin = 0
        x_frame_end = length

    x_candidate_indices = np.arange(x_frame_begin, x_frame_end)
    if mask is not None:
        x_candidate_indices_mask = mask[x_frame_begin: x_frame_end]
        x_candidate_indices = x_candidate_indices[x_candidate_indices_mask]

    if len(x_candidate_indices) == 0:
        x_candidate_indices = None

    return x_candidate_indices, frame_range_extendable


def _get_random_index(length, mask, rng_engine: np.random.Generator):
    if mask is None:
        if length == 1:
            return 0
        return rng_engine.integers(0, length)
    else:
        assert length == len(mask)
        assert mask.dtype == np.bool_
        indices = np.arange(0, len(mask))
        indices = indices[mask]
        if len(indices) == 0:
            return None
        return rng_engine.choice(indices)
