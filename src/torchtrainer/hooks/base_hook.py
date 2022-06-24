from typing import Any, List, Dict

import torch

from ..trainer import Trainer

__all__ = ["Hook"]


class Hook:
    """
    Base class for hooks.

    Notes:
        referrence:
        https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/train/hooks/base_hook.py
    """

    @property
    def state_dict(self):
        return {}

    @state_dict.setter
    def state_dict(self, state_dict: Dict):
        pass

    def on_train_begin(self, trainer: Trainer):
        pass

    def on_train_ends(self, trainer: Trainer):
        pass

    def on_train_failed(self, trainer: Trainer):
        pass

    def on_epoch_begin(self, trainer: Trainer):
        """
        Log at the beginning of train epoch.

        Args:
            trainer (Trainer): instance of Trainer class.
        """
        pass

    def on_batch_begin(self, trainer: Trainer):
        """
        Log at the beginning of train batch.

        Args:
            trainer (Trainer): instance of Trainer class.
        """
        pass

    def on_batch_end(
        self,
        trainer: Trainer,
        n_batch: int,
        train_batch: Any,
        result_list: List[torch.Tensor],
        loss_list: List[torch.Tensor],
    ):
        pass

    def on_validation_begin(self, trainer: Trainer):
        pass

    def on_validation_batch_begin(self, trainer: Trainer):
        pass

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        n_batch: int,
        val_batch: Any,
        result_list: List[torch.Tensor],
        loss_list: List[torch.Tensor],
    ):
        pass

    def on_validation_end(self, trainer: Trainer, mean_val_loss: torch.Tensor):
        pass

    def on_epoch_end(self, trainer: Trainer):
        pass
