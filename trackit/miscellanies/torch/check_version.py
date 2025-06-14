def is_torch_version_greater_or_equal(required_version: tuple[int, ...]):
    """
    Check if the installed PyTorch version is greater than or equal to the required version.

    Args:
        required_version (tuple): A tuple of integers representing the required version (e.g., (2, 1) for version 2.1)

    Returns:
        bool: True if the installed version is greater than or equal to the required version, False otherwise
    """
    import torch

    installed_version_str = torch.__version__

    if '+' in installed_version_str:
        installed_version_str = installed_version_str.split('+')[0]

    installed_parts = tuple(int(x) for x in installed_version_str.split('.')[:len(required_version)])

    return installed_parts >= required_version
