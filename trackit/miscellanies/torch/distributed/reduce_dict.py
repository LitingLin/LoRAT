import numpy as np
import torch
import torch.distributed as dist
from . import get_world_size, get_aux_process_group, get_backend, get_aux_backend


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    if len(input_dict) < 1:
        return input_dict
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    value_type = type(next(iter(input_dict.values())))

    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        if value_type is torch.Tensor:
            values = torch.stack(values, dim=0)
        else:
            values = np.stack(values, axis=0)
            values = torch.from_numpy(values)

        orig_device = values.device
        process_group = None
        backend = get_backend()
        if orig_device.type == 'cpu':
            process_group = get_aux_process_group()
            if process_group is not None:
                backend = get_aux_backend()

        if backend == 'nccl':
            values = values.cuda()
            all_reduce_op = dist.ReduceOp.AVG if average else dist.ReduceOp.SUM
            dist.all_reduce(values, op=all_reduce_op, group=process_group)
        else:
            dist.all_reduce(values, op=dist.ReduceOp.SUM, group=process_group)
            if average:
                values.div_(world_size)

        values = values.to(orig_device)
        reduced_dict = {k: v for k, v in zip(names, values)}
        if value_type is np.ndarray:
            reduced_dict = {k: v.numpy() for k, v in reduced_dict.items()}
        elif value_type is float:
            reduced_dict = {k: v.item() for k, v in reduced_dict.items()}
    return reduced_dict


class reduce_dict_async:
    def __init__(self, input_dict, average=True):
        """
        Args:
            input_dict (dict): all the values will be reduced
            average (bool): whether to do average or sum
        Reduce the values in the dictionary from all processes so that all processes
        have the averaged results. Returns a dict with the same fields as
        input_dict, after reduction.
        """
        self.result = None
        self.async_handle = None
        if len(input_dict) < 1:
            self.result = input_dict
            return
        world_size = get_world_size()
        if world_size < 2:
            self.result = input_dict
            return

        value_type = type(next(iter(input_dict.values())))

        with torch.inference_mode():
            names = []
            values = []
            # sort the keys so that they are consistent across processes
            for k in sorted(input_dict.keys()):
                names.append(k)
                values.append(input_dict[k])
            if value_type is torch.Tensor:
                values = torch.stack(values, dim=0)
            else:
                values = np.stack(values, axis=0)
                values = torch.from_numpy(values)

            orig_device = values.device
            process_group = None
            backend = get_backend()
            if orig_device.type == 'cpu':
                process_group = get_aux_process_group()
                if process_group is not None:
                    backend = get_aux_backend()

            if backend == 'nccl':
                values = values.cuda()
                all_reduce_op = dist.ReduceOp.AVG if average else dist.ReduceOp.SUM
                self.async_handle = dist.all_reduce(values, op=all_reduce_op, group=process_group, async_op=True)
            else:
                self.async_handle = dist.all_reduce(values, op=dist.ReduceOp.SUM, group=process_group, async_op=True)
            self.names = names
            self.values = values
            self.orig_device = orig_device
            self.backend = backend
            self.value_type = value_type
            self.average = average

    def wait_and_get(self):
        if self.async_handle is not None:
            self.async_handle.wait()

            with torch.inference_mode():
                world_size = get_world_size()
                if self.backend != 'nccl':
                    if self.average:
                        self.values.div_(world_size)

                self.values = self.values.to(self.orig_device)
                reduced_dict = {k: v for k, v in zip(self.names, self.values)}
                if self.value_type is np.ndarray:
                    reduced_dict = {k: v.numpy() for k, v in reduced_dict.items()}
                elif self.value_type is float:
                    reduced_dict = {k: v.item() for k, v in reduced_dict.items()}
                del self.values
                del self.names
                del self.orig_device
                del self.backend
                del self.value_type

                self.async_handle = None
                self.result = reduced_dict
        return self.result
