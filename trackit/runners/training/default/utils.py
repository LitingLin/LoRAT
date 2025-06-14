import torch
import torch.nn as nn
from trackit.criteria import CriterionOutput


def criterion_has_parameters(criterion: nn.Module) -> bool:
    for _ in criterion.parameters():
        return True
    return False


def get_loss_metrics(criterion_output: CriterionOutput):
    metrics = {}
    if criterion_output.metrics is None:
        metrics['loss'] = criterion_output.loss.detach()
    else:
        metrics['loss'] = sum(criterion_output.metrics.values())
        metrics.update(criterion_output.metrics)
    if criterion_output.extra_metrics is not None:
        metrics.update(criterion_output.extra_metrics)

    for name in list(metrics.keys()):
        metric = metrics[name]
        if isinstance(metric, torch.Tensor):
            metric = metric.cpu().item()
            metrics[name] = metric

    return metrics


class PerParameterStatistics:
    def __init__(self, print_interval: int = 100):
        self._module_statistics = {}
        self._print_interval = print_interval
        self._step = 0

    def collect(self, model: nn.Module):
        # for now collect gradient norms only
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self._module_statistics:
                    self._module_statistics[name] = []
                self._module_statistics[name].append(torch.linalg.vector_norm(param.grad, 2.0))

        self._step += 1
        if self._step % self._print_interval == 0:
            self.print_statistics()
            self._module_statistics.clear()

    def print_statistics(self):
        print(f"Per-parameter statistics at step {self._step}:")
        for name, norms in self._module_statistics.items():
            avg_norm = torch.mean(torch.stack(norms)).item()
            print(f"  {name}: avg grad norm = {avg_norm:.4f}")
        print("-" * 40)