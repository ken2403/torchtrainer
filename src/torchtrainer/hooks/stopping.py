from typing import Any, Dict, List
import torch

from ..hooks import Hook
from ..trainer import Trainer

__all__ = ["EarlyStopping", "MaxStepStopping", "NaNStopping"]


class EarlyStopping(Hook):
    def __init__(
        self,
        patience: int,
        threshold_ratio: float = 0.0001,
    ):
        """
        Hook to stop training if validation loss fails to improve.

        Args:
            patience (int): number of epochs which can pass without improvement
                of validation loss before training ends.
            threshold_ratio (float, optional): counter increases if
                curr_val_loss > (1-threshold_ratio) * best_loss
                Defaults to 0.0001.
        """
        self.best_loss = float("Inf")
        self.counter = 0
        self.threshold_ratio = threshold_ratio
        self.patience = patience
        self._name = "EarlyStopping"

    @property
    def state_dict(self):
        return {"counter": self.counter, "best_loss": self.best_loss}

    @state_dict.setter
    def state_dict(self, state_dict: Dict[str, Any]):
        self.counter = state_dict["counter"]
        self.best_loss = state_dict["best_loss"]

    def on_validation_end(self, trainer: Trainer, mean_val_loss: torch.Tensor):
        if mean_val_loss > (1 - self.threshold_ratio) * self.best_loss:
            self.counter += 1
        else:
            self.best_loss = mean_val_loss
            self.counter = 0

        if self.counter > self.patience:
            trainer._stop = True
            trainer._stop_by = self._name


class MaxStepStopping(Hook):
    def __init__(
        self,
        max_steps: int,
    ):
        """
        Hook to stop training when a maximum number of steps is reached.

        Args:
            max_steps (int): maximum number of steps.
        """
        self.max_steps = max_steps
        self._name = "MaxStepStopping"

    def on_batch_begin(self, trainer: Trainer):
        # stop training if max_steps is reached
        if trainer.step > self.max_steps:
            trainer._stop = True
            trainer._stop_by = self._name


class NaNStopping(Hook):
    def __init__(self):
        """
        Hook to stop training when traing loss is NaN.
        """
        self.nan_loss_order = None
        self._name = "NaNStopping"

    @property
    def state_dict(self):
        return {"nan_loss_order": self.nan_loss_order}

    @state_dict.setter
    def state_dict(self, state_dict: Dict[str, Any]):
        self.nan_loss_order = state_dict["nan_loss_order"]

    def on_batch_end(
        self,
        trainer: Trainer,
        n_batch: int,
        train_batch: Any,
        result_list: List[torch.Tensor],
        loss_list: List[torch.Tensor],
    ):
        for i, loss in enumerate(loss_list):
            if loss.isnan().any():
                self.nan_loss_order = i
                trainer._stop = True
                trainer._stop_by = self._name
