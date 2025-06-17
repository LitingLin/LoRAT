import time
import torch
import torchvision
import torchvision.transforms.functional
import numpy as np
import exifread
import io
import threading
from PIL import Image, ImageOps
from turbojpeg import TurboJPEG, TJPF_RGB


# https://github.com/lilohuang/PyTurboJPEG/blob/master/README.md
def _transpose_image(image: np.ndarray, orientation: int) -> np.ndarray:
    """
    Apply the corresponding rotation or flip to a NumPy image array based on the EXIF orientation value.
    See Orientation in https://www.exif.org/Exif2-2.PDF for details.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        orientation (int): The EXIF orientation value (1-8).

    Returns:
        np.ndarray: The correctly oriented image.
    """
    if orientation == 1:  # 1: Normal
        return image
    elif orientation == 2:
        return np.fliplr(image)
    elif orientation == 3:
        return np.rot90(image, 2)
    elif orientation == 4:
        return np.flipud(image)
    elif orientation == 5:
        return np.rot90(np.flipud(image), -1)
    elif orientation == 6:
        return np.rot90(image, -1)
    elif orientation == 7:
        return np.rot90(np.flipud(image))
    elif orientation == 8:
        return np.rot90(image)
    return image


def _apply_exif_orientation(image: np.ndarray, image_bytes: bytes) -> np.ndarray:
    """
    Read EXIF orientation from image bytes and apply it to the decoded image.
    This is a helper for libraries like TurboJPEG and Torchvision that don't handle EXIF automatically.

    Args:
        image (np.ndarray): The decoded image as a NumPy array.
        image_bytes (bytes): The raw bytes of the original image file.

    Returns:
        np.ndarray: The image with orientation corrected.
    """
    # Use details=False and stop_tag for performance, as we only need the orientation.
    tags = exifread.process_file(io.BytesIO(image_bytes), details=False, stop_tag='Image Orientation')
    orientation_tag = tags.get('Image Orientation')

    if orientation_tag and orientation_tag.values:
        orientation_val = orientation_tag.values[0]
        return _transpose_image(image, orientation_val)
    return image


def _decode_image_with_pil(image_bytes: bytes, handle_exif_orientation: bool = False) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    if handle_exif_orientation:
        image = ImageOps.exif_transpose(image)
    image = image.convert('RGB')
    image = np.array(image, dtype=np.uint8)
    return image


local_store = threading.local()
def _decode_image_with_turbojpeg(image_bytes: bytes, handle_exif_orientation: bool = False) -> np.ndarray:
    if not hasattr(local_store, 'jpeg_object'):
        local_store.jpeg_object = TurboJPEG()
    try:
        image = local_store.jpeg_object.decode(image_bytes, TJPF_RGB)
        if handle_exif_orientation:
            image = _apply_exif_orientation(image, image_bytes)
    except Exception as e:
        print(f"Failed to decode image with TurboJPEG: {e}")
        # Fallback to PIL if TurboJPEG fails
        image = _decode_image_with_pil(image_bytes, handle_exif_orientation)
    return image


def _decode_image_with_torchvision(image_bytes: bytes, handle_exif_orientation: bool = False) -> np.ndarray:
    return torchvision.io.image.decode_image(torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8), torchvision.io.image.ImageReadMode.RGB, apply_exif_orientation=handle_exif_orientation).permute(1, 2, 0).contiguous().numpy()


def decode_image(image_bytes: bytes, handle_exif_orientation: bool = False) -> np.ndarray:
    if image_bytes.startswith(b'\xff\xd8\xff'):
        return _decode_image_with_turbojpeg(image_bytes, handle_exif_orientation)
    else:
        return _decode_image_with_pil(image_bytes, handle_exif_orientation)


def decode_image_with_auto_retry(prefetched_image_bytes: bytes, image_path: str, handle_exif_orientation: bool = False, retry_times: int = 2) -> np.ndarray:
    for i in range(retry_times):
        try:
            if prefetched_image_bytes is None:
                with open(image_path, 'rb') as f:
                    prefetched_image_bytes = f.read()
            image = decode_image(prefetched_image_bytes, handle_exif_orientation)
            return image
        except Exception as e:
            print(f'Failed to decode image {image_path}')
            if i == retry_times - 1:
                raise e
            time.sleep(0.1)
            print(f'Retry {i + 1}/{retry_times}')
            prefetched_image_bytes = None


def read_image_with_auto_retry(image_path: str, retry_count=3) -> np.ndarray:
    for index in range(retry_count):
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            return decode_image(image_bytes)
        except (RuntimeError, FileNotFoundError, OSError) as e:
            if index + 1 == retry_count:
                raise RuntimeError(f"Failed to read image file {image_path}.\nException: {str(e)}")
            time.sleep(0.1)


def read_file_with_auto_retry(file_path: str, retry_count=3) -> bytes:
    for i in range(retry_count):
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            print(e)
            print('Retry {}/{}'.format(i+1, retry_count))
            time.sleep(0.1)
            if i == retry_count-1:
                raise e


# https://stackoverflow.com/questions/50134468/convert-boolean-numpy-array-to-pillow-image
def _img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)


def write_image(image: np.ndarray, image_path: str, jpeg_quality: int = 95):
    if image.dtype == np.bool_:
        image = _img_frombytes(image)
    else:
        image = Image.fromarray(image)
    if image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
        with open(image_path, 'wb') as f:
            image.save(f, format='jpeg', quality=jpeg_quality)
    elif image_path.endswith('.png'):
        with open(image_path, 'wb') as f:
            image.save(f, format='png')
    else:
        raise ValueError(f"Unsupported image format: {image_path}")
