from typing import Dict


class EarlyStoppingManager:
    """
    A class to manage early stopping during model training.

    This class monitors a specified metric and stops training if the metric
    does not improve for a given number of epochs.
    """

    def __init__(self, metric_name: str, mode: str = 'min', patience: int = 10):
        """
        Initializes the EarlyStoppingManager.

        Args:
            metric_name (str): The name of the metric to monitor (e.g., 'val_loss').
            mode (str, optional): The mode for monitoring the metric.
                Can be 'max' or 'min'. Defaults to 'max'.
            patience (int, optional): The number of epochs to wait for improvement
                before stopping. Defaults to 10.
        """
        if mode not in ['max', 'min']:
            raise ValueError("Mode should be 'max' or 'min'")

        self.metric_name = metric_name
        self.mode = mode
        self.patience = patience
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0

    def should_stop(self, metrics: Dict[str, float], epoch: int) -> bool:
        """
        Checks if training should be stopped based on the provided metrics.

        Args:
            metrics (Dict[str, float]): A dictionary of metrics for the current epoch.
            epoch (int): The current epoch number.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if self.metric_name not in metrics:
            return False

        current_metric = metrics[self.metric_name]

        improved = (self.mode == 'max' and current_metric > self.best_metric) or \
                   (self.mode == 'min' and current_metric < self.best_metric)

        if improved:
            self.best_metric = current_metric
            self.best_epoch = epoch

        epochs_without_improvement = epoch - self.best_epoch

        if epochs_without_improvement >= self.patience:
            print(
                f"Early stopping at epoch {epoch}. Best metric ({self.metric_name}): {self.best_metric:.4f} at epoch {self.best_epoch}")
            return True

        return False
