from typing import Callable
import functools
import numpy as np

from trackit.data.source import _TrackingDataset_BaseFrame
from trackit.miscellanies.image.io import decode_image_with_auto_retry


def get_frame_decoder(frame: _TrackingDataset_BaseFrame, prefetch=True) -> Callable[[], np.ndarray]:
    if prefetch and frame.is_file_backed():
        file_path = frame.get_frame_file_path()
        try:
            with open(file_path, 'rb') as f:
                raw_content = f.read()
        except OSError:
            return frame.get_frame
        return functools.partial(decode_image_with_auto_retry, raw_content, file_path)
    else:
        return frame.get_frame
