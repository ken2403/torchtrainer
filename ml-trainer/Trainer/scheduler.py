import math

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


__all__ = ["WarmupCosineDecayAnnealingLR"]


class WarmupCosineDecayAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_epoch: int,
        num_warmup: int,
        T_max: int,
        eta_min: float = 1e-7,
        lr_max: float = 1e-3,
        lr_min: float = 1e-10,
        decay_coef: float = 1.5,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        Warup at the begining of learning process, and after warming up, LR is
        decaing with cosine function while cosine annealing.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer object.
            num_epoch (int): number of epoch.
            num_warmup (int):number of warm up epoch
            T_max (int):  Maximum number of iterations of annealing period.
                (see ref:
                https://pytorch.org/docs/1.9.0/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html).
            eta_min (float, optional): Annealing minimum LR. Defaults to 1e-7.
            lr_max (float, optional): Max LR of warm up process. Defaults to 1e-3.
            lr_min (float, optional): Global minimum LR. Defaults to 1e-10.
            decay_coef (float, optional): Decay coefficient. Defaults to 1.5.
            last_epoch (int, optional): The index of last epoch. Defaults to -1.
            verbose (bool, optional): _description_If True, prints a message to stdout
                for each updatde. Defaults to False.
        """
        assert num_warmup < num_epoch, "Please set 'num_warmup' lower than 'num_epoch'"
        assert decay_coef > 0, "Please set 'decay_coef' higher than 0."
        self.num_epoch = num_epoch
        self.num_warmup = num_warmup
        self.T_max = T_max
        self.eta_min = eta_min
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.decay_coef = decay_coef
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        if self.last_epoch == 0:
            return self.base_lrs
        if self.last_epoch < self.num_warmup:
            grads = (self.lr_max - np.array(self.base_lrs)) / (self.num_warmup - 1)
            return [
                base_lr + self.last_epoch * grad
                for grad, base_lr in zip(grads, self.base_lrs)
            ]
        if self.last_epoch >= self.num_warmup:
            epoch_diff = self.last_epoch - self.num_warmup
            decay = math.cos(epoch_diff / self.decay_coef / self.num_epoch * math.pi / 2)
            if (epoch_diff - 1 - self.T_max) % (2 * self.T_max) == 0:
                return [
                    max(
                        min(
                            decay
                            * (
                                group["lr"]
                                + (self.lr_max - self.eta_min)
                                * (1 - math.cos(math.pi / self.T_max))
                                / 2
                            ),
                            self.lr_max,
                        ),
                        self.lr_min,
                    )
                    for group in self.optimizer.param_groups
                ]
            return [
                max(
                    min(
                        decay
                        * (
                            (1 + math.cos(math.pi * epoch_diff / self.T_max))
                            / (1 + math.cos(math.pi * (epoch_diff - 1) / self.T_max))
                            * (group["lr"] - self.eta_min)
                            + self.eta_min
                        ),
                        self.lr_max,
                    ),
                    self.lr_min,
                )
                for group in self.optimizer.param_groups
            ]
