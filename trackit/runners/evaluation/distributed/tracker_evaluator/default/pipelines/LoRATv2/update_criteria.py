import operator
from typing import Sequence

import numpy as np


class UpdateCriteria:
    def __call__(self, predicted_scores: np.ndarray) -> Sequence[int]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def start(self):
        pass

    def stop(self):
        pass


class ScoreBasedCriteria(UpdateCriteria):
    def __init__(self, name: str, score_threshold: float, mode: str = 'gt'):
        self.name = name
        self.score_threshold = score_threshold
        assert mode in ('lt', 'gt', 'le', 'ge')
        self.operator = getattr(operator, mode)

    def __call__(self, predicted_scores: np.ndarray):
        pass_indices = [index for index, predicted_score in enumerate(predicted_scores)
                         if self.operator(predicted_score, self.score_threshold)]
        self.total += len(predicted_scores)
        self.num_pass += len(pass_indices)
        self.total_predicted_score += sum(predicted_scores)
        return pass_indices

    def start(self):
        self.total = 0
        self.num_pass = 0
        self.total_predicted_score = 0

    def stop(self):
        print(f"{self.name}: "
              f"pass ratio = {self.num_pass / self.total:.4f}, "
              f"mean of predicted score = {self.total_predicted_score / self.total:.4f}")
