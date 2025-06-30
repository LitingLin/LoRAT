def seed_all_rng(seed: int):
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(random.randint(0, 255))
    import torch
    torch.manual_seed(random.randint(0, 255))
