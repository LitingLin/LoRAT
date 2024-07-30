from typing import Optional, Sequence
from .._common import _BaseFilter


class DataCleaning_ObjectCategory(_BaseFilter):
    def __init__(self, category_ids_to_remove: Optional[Sequence[int]] = None, make_category_id_sequential: bool = True):
        self.category_ids_to_remove = category_ids_to_remove
        self.make_category_id_sequential = make_category_id_sequential
