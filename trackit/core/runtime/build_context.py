from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np
import torch

from trackit.core.runtime.services import ServicesRegistry
from trackit.models import ModelManager


@dataclass()
class BuildContext:
    name: str
    model: ModelManager
    device: torch.device
    seed: int
    pin_memory: bool
    master_node_ip_addr: str
    run_id: str
    global_synchronized_rng: np.random.Generator
    local_rng: np.random.Generator
    instance_specific_rng: np.random.Generator
    wandb_instance: Optional[Any]
    variables: dict = field(default_factory=dict)
    services: ServicesRegistry = field(default_factory=ServicesRegistry)
