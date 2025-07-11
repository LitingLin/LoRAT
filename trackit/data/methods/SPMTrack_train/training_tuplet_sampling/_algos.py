import numpy as np
from typing import Optional, Tuple
from trackit.data.source import TrackingDataset_Sequence, TrackingDataset_Track
from ._types import SiamesePairSamplingMethod
import random

def _get_random_track(sequence: TrackingDataset_Sequence, rng_engine: np.random.Generator):
    number_of_tracks = sequence.get_number_of_tracks()
    assert number_of_tracks > 0
    if number_of_tracks == 1:
        index_of_track = 0
    else:
        index_of_track = rng_engine.integers(0, number_of_tracks)
    track = sequence.get_track_by_index(index_of_track)
    return track


def _get_random_frame_from_track(track: TrackingDataset_Track, ensure_object_exists, rng_engine: np.random.Generator):
    track_length = len(track)
    assert track_length > 0
    if track_length == 1:
        frame_index = 0
        if ensure_object_exists and track.get_all_object_existence_flag() is not None:
            assert track.get_all_object_existence_flag()[0]
    else:
        if not ensure_object_exists or track.get_all_object_existence_flag() is None:
            frame_index = rng_engine.integers(0, track_length)
        else:
            valid_frames = np.arange(0, track_length)[track.get_all_object_existence_flag()]
            frame_index = rng_engine.choice(valid_frames)
    frame = track[frame_index]
    return frame


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
        assert len(indices) > 0
        return rng_engine.choice(indices)


def _get_search_frame_candidates(z_index: int, length: int, frame_range: Optional[int], mask: Optional[np.ndarray] = None,
                                 sampling_method: SiamesePairSamplingMethod = SiamesePairSamplingMethod.causal) -> Tuple[Optional[np.ndarray], bool]:
    frame_range_extendable = False
    if sampling_method == SiamesePairSamplingMethod.causal:
        x_frame_begin = z_index + 1
        if x_frame_begin >= length:
            return None, frame_range_extendable
        if frame_range is not None:
            x_frame_end = z_index + frame_range + 1
            if x_frame_end >= length:
                x_frame_end = length
            else:
                frame_range_extendable = True
        else:
            x_frame_end = length
        if x_frame_begin >= x_frame_end:
            return None, frame_range_extendable
    elif sampling_method == SiamesePairSamplingMethod.reverse_causal:
        x_frame_end = z_index
        if x_frame_end == 0:
            return None, frame_range_extendable
        if frame_range is not None:
            x_frame_begin = z_index - frame_range
            if x_frame_begin > 0:
                frame_range_extendable = True
            else:
                x_frame_begin = 0
        else:
            x_frame_begin = 0
        if x_frame_begin >= x_frame_end:
            return None, frame_range_extendable
    elif sampling_method == SiamesePairSamplingMethod.interval:
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
    else:
        raise NotImplementedError

    x_candidate_indices = np.arange(x_frame_begin, x_frame_end)

    if sampling_method in (SiamesePairSamplingMethod.causal, SiamesePairSamplingMethod.reverse_causal):
        if mask is not None:
            x_candidate_indices_mask = mask[x_frame_begin: x_frame_end]
            x_candidate_indices = x_candidate_indices[x_candidate_indices_mask]
    elif sampling_method == SiamesePairSamplingMethod.interval:
        if mask is not None:
            x_candidate_indices_mask = np.copy(mask[x_frame_begin: x_frame_end])
            x_candidate_indices_mask[z_index - x_frame_begin] = False
            x_candidate_indices = x_candidate_indices[x_candidate_indices_mask]
        else:
            x_candidate_indices = np.delete(x_candidate_indices, z_index - x_frame_begin)
    else:
        raise NotImplementedError

    if len(x_candidate_indices) == 0:
        x_candidate_indices = None

    return x_candidate_indices, frame_range_extendable


def _do_siamfc_pair_sampling(length: int, frame_range: Optional[int], mask: np.ndarray = None,
                             sampling_method: SiamesePairSamplingMethod = SiamesePairSamplingMethod.causal,
                             rng_engine: np.random.Generator = np.random.default_rng(),
                             frame_range_auto_extend_step: int = 5,
                             frame_range_auto_extend_max_retry_count: int = 20,
                             disable_frame_range_if_not_found: bool = False,
                             num_template_frames: int = 1,
                             num_search_frames: int = 1,
                             max_sample_interval: int = 1):
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
    cnt = 0
    while True:
        base_z_index = _get_random_index(length, mask, rng_engine)
        if length == 1:
            return [base_z_index]*num_template_frames, [base_z_index]*num_search_frames

        z_candidate_indices, _ = _get_search_frame_candidates(base_z_index, length, frame_range, mask, sampling_method) # 要选剩余的templates

        if z_candidate_indices is None:
            print("z_candidate_indices is None! Try again.")
            continue

        assert z_candidate_indices is not None
        if len(z_candidate_indices) < num_template_frames:
            cnt += 1
            continue
        z_indexes = rng_engine.choice(z_candidate_indices, size=num_template_frames - 1)

        z_indexes = list(z_indexes) + [base_z_index]
        z_indexes.sort(reverse=False)

        # template 获取完毕

        left_candidate_indices, _ = _get_search_frame_candidates(z_indexes[0], length, max_sample_interval, mask, sampling_method)
        left_candidate_indices = left_candidate_indices[left_candidate_indices < z_indexes[0]]
        right_candidate_indices, _ = _get_search_frame_candidates(z_indexes[-1], length, max_sample_interval, mask, sampling_method)
        right_candidate_indices = right_candidate_indices[right_candidate_indices > z_indexes[-1]]

        if len(left_candidate_indices) < num_search_frames and len(right_candidate_indices) < num_search_frames:
            cnt += 1
            frame_range += frame_range_auto_extend_step
            if cnt == frame_range_auto_extend_max_retry_count:
                raise
            continue
        elif len(left_candidate_indices) < num_search_frames:
            x_indexes = list(rng_engine.choice(right_candidate_indices, size=num_search_frames))
        elif len(right_candidate_indices) < num_search_frames:
            x_indexes = list(rng_engine.choice(left_candidate_indices, size=num_search_frames))
        else:
            # 随机
            if random.random() < 0.5:
                x_indexes = list(rng_engine.choice(left_candidate_indices, size=num_search_frames))
            else:
                x_indexes = list(rng_engine.choice(right_candidate_indices, size=num_search_frames))

        indexes = z_indexes + x_indexes
        if random.random() < 0.5:
            indexes.sort(reverse=False)
        else:
            indexes.sort(reverse=True)
        return indexes[:num_template_frames], indexes[num_template_frames:]


def _do_negative_siamfc_pair_sampling(length: int, frame_range: int=None, mask: np.ndarray=None, rng_engine: np.random.Generator = np.random.default_rng()):
    z_index = _get_random_index(length, mask, rng_engine)
    if mask is None or length == 1:
        return z_index,

    not_mask = ~mask
    not_mask[z_index] = False

    indices = np.arange(length)
    if frame_range is not None:
        begin = min(z_index - frame_range, 0)
        end = max(z_index + frame_range + 1, length)
        indices = indices[begin: end]
        not_mask = not_mask[begin: end]
    indices = indices[not_mask]
    if len(indices) == 0:
        return z_index,
    x_index = rng_engine.choice(indices)
    return z_index, x_index


def get_random_positive_siamese_training_pair_from_track(
        track: TrackingDataset_Track,
        frame_range: Optional[int], sampling_method: SiamesePairSamplingMethod,
        rng_engine: np.random.Generator,
        frame_range_auto_extend_step: int = 5,
        frame_range_auto_extend_max_retry_count: int = 20,
        disable_frame_range_if_not_found: bool = False,
        num_template_frames: int = 1,
        num_search_frames: int = 1,
        max_sample_interval: int = 1
):
    return _do_siamfc_pair_sampling(len(track), frame_range, track.get_all_object_existence_flag(), sampling_method,
                                    rng_engine, frame_range_auto_extend_step, frame_range_auto_extend_max_retry_count,
                                    disable_frame_range_if_not_found, num_template_frames, num_search_frames, max_sample_interval)


def get_random_negative_siamese_training_pair_from_track(
        track: TrackingDataset_Track,
        frame_range: Optional[int], rng_engine: np.random.Generator):
    track_length = len(track)
    assert track_length > 0
    if track_length == 1:
        frame_indices = (0,)
        assert track[0].get_existence_flag() is None or track[0].get_existence_flag()  # z must exist
    else:
        frame_indices = _do_negative_siamfc_pair_sampling(len(track), frame_range, track.get_all_object_existence_flag(), rng_engine)
    return frame_indices
