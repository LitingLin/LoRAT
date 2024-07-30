import socket
from trackit.miscellanies.torch.distributed import is_dist_initialized, get_world_size, get_local_world_size, get_num_nodes
from typing import Optional, Sequence
from .collective_communication import CollectiveCommunication
from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkContext:
    ip: str
    hostname: str


@dataclass(frozen=True)
class DistributedGroupNetworkContext:
    local: NetworkContext
    group: Optional[Sequence[NetworkContext]]


_context: Optional[DistributedGroupNetworkContext] = None


def init_distributed_group_network_context():
    global _context
    assert _context is None, "DistributedGroupNetworkContext has been initialized"
    local_machine_hostname = socket.gethostname()
    try:
        local_machine_ip = socket.gethostbyname(local_machine_hostname)
    except socket.gaierror:
        local_machine_ip = "N/A"

    group_network_context = None

    if is_dist_initialized():
        collective_communication = CollectiveCommunication(get_world_size() * 1024)
        distributed_group_network_contexts = collective_communication.all_gather((local_machine_ip, local_machine_hostname))

        group_network_context = []
        for i_node in range(get_num_nodes()):
            group_network_context.append(NetworkContext(*distributed_group_network_contexts[i_node * get_local_world_size()]))
            for i_local_rank in range(get_local_world_size() - 1):
                rank = i_node * get_local_world_size() + i_local_rank
                assert distributed_group_network_contexts[rank] == distributed_group_network_contexts[rank + 1]

        group_network_context = tuple(group_network_context)
        assert len(set(network_context.ip for network_context in group_network_context)) == len(group_network_context)

    _context = DistributedGroupNetworkContext(NetworkContext(local_machine_ip, local_machine_hostname), group_network_context)
    return _context


def get_distributed_group_network_context() -> DistributedGroupNetworkContext:
    assert _context is not None, "DistributedGroupNetworkContext has not been initialized"
    return _context
