import numpy as np

from trackit.miscellanies.torch.distributed import get_rank
from trackit.core.runtime.application.builder import build_application

from ..utils.reproducibility import seed_all_rng


def prepare_application(runtime_vars, config, wandb_instance):
    global_synchronized_rng = np.random.Generator(np.random.PCG64(runtime_vars.seed))
    runtime_vars.seed = runtime_vars.seed + get_rank()
    local_rng = np.random.Generator(np.random.PCG64(runtime_vars.seed))
    instance_specific_rng = np.random.Generator(np.random.PCG64(runtime_vars.instance_id))

    seed_all_rng(runtime_vars.seed)

    return build_application(config, runtime_vars, global_synchronized_rng, local_rng, instance_specific_rng, wandb_instance)
