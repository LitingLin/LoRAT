import hashlib
from typing import Optional, Sequence


def generate_dataset_unique_id(dataset_name: str, dataset_splits: Optional[Sequence[str]], filters, with_uuid=True):
    components = [dataset_name]
    if dataset_splits is not None and len(dataset_splits) > 0:
        assert not isinstance(dataset_splits, str) and all(isinstance(ds, str) for ds in dataset_splits), dataset_splits
        components.append('+'.join(dataset_splits))
    if with_uuid:
        m = hashlib.md5()
        m.update(bytes(dataset_name, encoding='utf-8'))
        dataset_filters = filters
        if dataset_filters is not None:
            m.update(bytes(str(dataset_filters), encoding='utf-8'))
        components.append(m.hexdigest())
    return '-'.join(components)
