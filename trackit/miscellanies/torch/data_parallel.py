import torch


def should_use_data_parallel(device: torch.device) -> bool:
    """
    Determines if DataParallel should be used based on the current distributed
    setup and available CUDA devices.

    Args:
        device (torch.device): The device on which the model will run.

    Returns:
        bool: True if DataParallel should be used, False otherwise.
    """
    from trackit.miscellanies.torch.distributed import get_world_size
    return (
        get_world_size() == 1 and
        device.type == 'cuda' and
        torch.cuda.is_available() and
        torch.cuda.device_count() > 1
    )
