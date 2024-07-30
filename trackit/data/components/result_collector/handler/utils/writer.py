import os.path
import zipfile
from typing import BinaryIO, TextIO, Sequence
import io


class FolderWriter:
    def open_binary_file_handle(self, path: Sequence[str]) -> BinaryIO:
        raise NotImplementedError()

    def open_text_file_handle(self, path: Sequence[str]) -> TextIO:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class ZipfileWriter(FolderWriter):
    def __init__(self, zip_file_path: str):
        self._zip_file = zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED)

    def open_binary_file_handle(self, path: Sequence[str]):
        return self._zip_file.open('/'.join(path), 'w')

    def open_text_file_handle(self, path: Sequence[str]):
        return io.TextIOWrapper(self._zip_file.open('/'.join(path), 'w'), encoding='utf-8', newline='')

    def close(self):
        self._zip_file.close()


class PlainFolderWriter(FolderWriter):
    def __init__(self, folder_file_path: str):
        self._folder_file_path = folder_file_path
        self._folder_cache = set()

    def open_binary_file_handle(self, path: Sequence[str]):
        return open(self._make_dirs_and_get_full_file_path(path), 'wb')

    def open_text_file_handle(self, path: Sequence[str]):
        return open(self._make_dirs_and_get_full_file_path(path), 'w', encoding='utf-8', newline='')

    def _make_dirs_and_get_full_file_path(self, path: Sequence[str]):
        if len(path) > 1:
            rel_folder_path = os.path.join(*path[:-1])
            if rel_folder_path not in self._folder_cache:
                full_folder_path = os.path.join(self._folder_file_path, rel_folder_path)
                os.makedirs(full_folder_path, exist_ok=True)
                self._folder_cache.add(rel_folder_path)
                file_path = os.path.join(full_folder_path, path[-1])
            else:
                file_path = os.path.join(self._folder_file_path, rel_folder_path, path[-1])
        else:
            file_path = os.path.join(self._folder_file_path, *path)
        return file_path

    def close(self):
        pass
