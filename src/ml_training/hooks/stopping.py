from typing import Any, Dict, List
import torch

from ml_training.hooks import Hook
from ml_training import Trainer

__all__ = ["EarlyStopping", "NaNStopping"]


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

    def on_batch_begin(self, trainer: Trainer):
        # stop training if max_steps is reached
        if trainer.step > self.max_steps:
            trainer._stop = True


class NaNStopError(Exception):
    pass


class NaNStopping(Hook):
    def __init__(self) -> None:
        """
        Hook to stop training when traing loss is None.
        """
        self.i = None

    @property
    def state_dict(self):
        return {"loss_order": self.i}

    @state_dict.setter
    def state_dict(self, state_dict: Dict[str, Any]):
        self.i = state_dict["i"]

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
                self.i = i
                trainer._stop = True
                raise NaNStopError(
                    f"The {i}th value of training loss has become nan! Stop training."
                )
