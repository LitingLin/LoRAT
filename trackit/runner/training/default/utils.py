import torch.nn as nn


def criterion_has_parameters(criterion: nn.Module) -> bool:
    for _ in criterion.parameters():
        return True
    return False
