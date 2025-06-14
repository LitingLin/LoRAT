import hashlib
from typing import Optional, Sequence


def generate_dataset_unique_id(dataset_name: str,
                               dataset_splits: Optional[Sequence[str]],
                               dataset_extra_flags: Optional[Sequence[str]],
                               dataset_filters, with_uuid=True):
    components = [dataset_name]
    if dataset_splits is not None and len(dataset_splits) > 0:
        assert not isinstance(dataset_splits, str) and all(isinstance(ds, str) for ds in dataset_splits), dataset_splits
        components.append('+'.join(dataset_splits))
    if dataset_extra_flags is not None and len(dataset_extra_flags) > 0:
        assert not isinstance(dataset_extra_flags, str) and all(isinstance(flag, str) for flag in dataset_extra_flags), dataset_extra_flags
        components.append('+'.join(dataset_extra_flags))
    if with_uuid:
        m = hashlib.md5()
        m.update(bytes(dataset_name, encoding='utf-8'))
        if dataset_filters is not None:
            m.update(bytes(str(dataset_filters), encoding='utf-8'))
        if dataset_extra_flags is not None:
            m.update(bytes(str(dataset_extra_flags), encoding='utf-8'))
        components.append(m.hexdigest())
    return '-'.join(components)
