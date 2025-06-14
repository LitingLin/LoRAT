from enum import Flag, auto
from trackit.datasets.common.seed import BaseSeed


class NFSDatasetVersionFlag(Flag):
    fps_30 = auto()
    fps_240 = auto()


class NFS_Seed(BaseSeed):
    def __init__(self, root_path: str=None, version: NFSDatasetVersionFlag = NFSDatasetVersionFlag.fps_30, manual_anno_only=False):
        if root_path is None:
            root_path = self.get_path_from_config('NFS_PATH')
        if isinstance(version, str):
            version = NFSDatasetVersionFlag[version]
        flags = []
        if version == NFSDatasetVersionFlag.fps_30:
            flags.append('fps_30')
        elif version == NFSDatasetVersionFlag.fps_240:
            flags.append('fps_240')
        else:
            raise Exception

        if manual_anno_only:
            flags.append('no_generated')
        self.nfs_version = version
        self.manual_anno_only = manual_anno_only
        super().__init__('NFS', root_path, extra_flags=flags)

    def construct(self, constructor):
        from .Impl.NFS import construct_NFS
        construct_NFS(constructor, self)
