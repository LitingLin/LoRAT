import torch

class set_default_dtype:
    def __init__(self, dtype: torch.dtype):
        self._default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_default_dtype(self._default_dtype)
