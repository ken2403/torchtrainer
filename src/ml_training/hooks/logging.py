import os
import pathlib
import time
from typing import Callable, List, Any

import torch

from ml_training.hooks import Hook
from ml_training.trainer.trainer import Trainer

__all__ = ["LoggingHook", "CSVHook"]


class LoggingHook(Hook):
    def __init__(
        self,
        log_path: pathlib.Path,
        metrics: List[Callable[[Any], torch.Tensor]],
        log_train_loss: bool = True,
        log_validation_loss: bool = True,
        log_learning_rate: bool = True,
    ):
        """
        Base class for logging hooks.

        Args:
            log_path (pathlib.Path): path to directory in which log files will be stored.
            metrics (List): metrics to log; each metric has to be a function which gets
                one train_loader and returns torch.Tensor.
            log_train_loss (bool, optional): enable logging of training loss.
                Defaults to True.
            log_validation_loss (bool, optional): enable logging of validation loss.
                Defaults to True.
            log_learning_rate (bool, optional): enable logging of current learning rate.
                Defaults to True.
        """
        self.log_train_loss = log_train_loss
        self.log_validation_loss = log_validation_loss
        self.log_learning_rate = log_learning_rate
        self.log_path = log_path

        self._train_loss_list = []
        self._train_counter = 0
        self._val_loss_list = []
        self._val_counter = 0
        self.metrics = metrics
        self._metrics_results = []

    def on_epoch_begin(self, trainer: Trainer):
        # reset train_loss and counter
        if self.log_train_loss:
            self._train_loss_list = []
            self._train_counter = 0
        else:
            self._train_loss_list = None
        if self.log_validation_loss:
            self._val_loss_list = []
            self._val_counter = 0
        else:
            self._val_loss_list = None

    def on_batch_end(
        self,
        trainer: Trainer,
        n_batch: int,
        train_batch: Any,
        result_list: List[torch.Tensor],
        loss_list: List[torch.Tensor],
    ):
        if self.log_train_loss:
            for i, loss in enumerate(loss_list):
                if self._train_counter == 0:
                    self._train_loss_list.append(loss.detach().clone().cpu().data)
                else:
                    self._train_loss_list[i] += float(loss.detach().clone().cpu().data)
            self._train_counter += n_batch

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        n_batch: int,
        val_batch: Any,
        result_list: List[torch.Tensor],
        loss_list: List[torch.Tensor],
    ):
        if self.log_validation_loss:
            for i, loss in enumerate(loss_list):
                if self._val_counter == 0:
                    self._val_loss_list.append(loss.detach().clone().cpu().data)
                else:
                    self._val_loss_list[i] += float(
                        loss.data.detach().clone().cpu().data
                    )
            self._val_counter += n_batch
        if len(self.metrics) == 0:
            pass
        if len(self._metrics_result) == 0:
            for metric in self.metrics:
                m = metric(val_batch, result_list)
                self._metrics_results.append(m)
        else:
            for i, metric in enumerate(self.metrics):
                m = metric(val_batch, result_list)
                self._metrics_results[i] += m


class CSVHook(LoggingHook):
    def __init__(
        self,
        log_path: pathlib.Path,
        metrics: List[Callable[[Any], torch.Tensor]],
        metrics_names: List[str],
        log_train_loss: bool = True,
        log_validation_loss: bool = True,
        log_learning_rate: bool = True,
        every_n_epochs: int = 1,
        n_loss: int = 1,
    ):
        """
        Hook for logging training process to CSV files.

        Args:
            log_path (pathlib.Path): path to directory in which log files will be stored.
            metrics (List): metrics to log; each metric has to be a function which gets
                one train_loader and returns torch.Tensor.
            metrics_names (List): list of metrics name.
            log_train_loss (bool, optional): enable logging of training loss.
                Defaults to True.
            log_validation_loss (bool, optional): enable logging of validation loss.
                Defaults to True.
            log_learning_rate (bool, optional): enable logging of current learning rate.
                Defaults to True.
            every_n_epochs (int, optional): epochs after which logging takes place.
                Defaults to 1.
            n_loss (int, optional): number of log loss data. Defaults to 1.
        """
        log_path = log_path.joinpath("log.csv")
        super().__init__(
            log_path, metrics, log_train_loss, log_validation_loss, log_learning_rate
        )

        self._offset = 0
        self._restart = False
        self.metrics_names = metrics_names
        self.every_n_epochs = every_n_epochs
        self.n_loss = n_loss

    def on_train_begin(self, trainer: Trainer):
        if self.log_path.exists():
            remove_file = False
            with open(self.log_path, "r") as f:
                # Ensure there is one entry apart from header
                lines = f.readlines()
                if len(lines) > 1:
                    self._offset = float(lines[-1].split(",")[0]) - time.time()
                    self._restart = True
                else:
                    remove_file = True

            # Empty up to header, remove to avoid adding header twice
            if remove_file:
                self.log_path.unlink()
        else:
            self._offset = -time.time()
            # Create the log dir if it does not exists, since write cannot
            # create a full path
            log_dir = self.log_path.parent
            if not log_dir.exists():
                log_dir.mkdir()

        if not self._restart:
            log = ""
            log += "Time"

            if self.log_learning_rate:
                for i in range(self.n_loss):
                    log += f",Learning rate {i+1}"

            if self.log_train_loss:
                for i in range(self.n_loss):
                    log += f",Train loss {i+1}"

            if self.log_validation_loss:
                for i in range(self.n_loss):
                    log += f",Validation loss {i+1}"

            if len(self.metrics) > 0:
                log += ","

            for i, _ in enumerate(self.metrics):
                log += self.metrics_names[i]
                if i < len(self.metrics) - 1:
                    log += ","

            with open(self.log_path, "a+") as f:
                f.write(log + os.linesep)

    def on_validation_end(
        self,
        trainer: Trainer,
        mean_val_loss: torch.Tensor,
    ):
        if trainer.epoch % self.every_n_epochs == 0:
            ctime = time.time() + self._offset
            log = str(ctime)

            if self.log_learning_rate:
                for i in range(self.n_loss):
                    log += "," + str(trainer.optimizer_list[i].param_groups[0]["lr"])

            if self.log_train_loss:
                for i in range(self.n_loss):
                    log += "," + str(self._train_loss_list[i] / self._train_counter)

            if self.log_validation_loss:
                for i in range(self.n_loss):
                    log += "," + str(self._val_loss_list[i] / self._val_counter)

            if len(self.metrics) > 0:
                log += ","

            for i, result in enumerate(self._metrics_results):
                m = result.detac().clone().cpu() / self._val_counter
                log += str(m)
                if i < len(self.metrics) - 1:
                    log += ","

            with open(self.log_path, "a") as f:
                f.write(log + os.linesep)
