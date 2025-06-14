import torch.nn as nn


def count_model_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to analyze.
        trainable_only (bool): If True, count only trainable parameters. Default is False.

    Returns:
        int: The total number of parameters in the model.
    """
    total_params = 0
    counted_params = set()

    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        if param in counted_params:
            continue

        param_size = param.numel()
        counted_params.add(param)
        total_params += param_size

    return total_params
