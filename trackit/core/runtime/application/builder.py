import numpy as np


def build_application(config: dict, runtime_vars,
                      global_synchronized_rng: np.random.Generator,
                      local_rng: np.random.Generator,
                      instance_specific_rng: np.random.Generator,
                      wandb_instance):
    if config['run']['type'] == 'default':
        from .default.builder import build_default_application
        return build_default_application(config, runtime_vars,
                                         global_synchronized_rng, local_rng, instance_specific_rng,
                                         wandb_instance)
    else:
        raise NotImplementedError(f"application type {config['run']['type']} is not supported")
