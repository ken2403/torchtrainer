import pathlib
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


__all__ = ["Trainer"]


class Trainer:
    """
    Class to train a pytorch model. This contains an internal training loop which takes
    care of validation and can be extended with custom functionality using hooks.
    """

    def __init__(
        self,
        model_path: pathlib.Path,
        model: nn.Module,
        n_epoch: int,
        device: torch.device,
        train_loader,
        val_loader,
        optimizer_list: List[torch.optim.Optimizer],
        scheduler: Optional[List[torch.optim.lr_scheduler._LRScheduler]] = None,
        keep_n_checkpoints: int = 1,
        checkpoint_interval: int = 10,
        validation_interval: int = 1,
        hooks: List = [],
        best_label: str = "",
        loss_is_normalized: bool = True,
    ):
        """
        Class to train a model.
        This contains an internal training loop which takes care of validation and can be
        extended with custom functionality using hooks.

        Args:
            model_path (pathlib.Path): path to the model directory.
            model (nn.Module): model to be trained.
            n_epoch (int): number of training epoch.
            device (torch.device): calculation device.
            train_loader (torch.utils.data.Dataloader): data loader for training set.
            val_loader (torch.utirls.data.Dataloader): data loader for validation set
            optimizer_list (List[torch.optim.Optimizer]): training optimizer.
            scheduer (List[torch.optim.lr_schduler._LRScheduler], optional): training LR
                scheduler. Defaults to None.
            keep_n_checkpoints (int, optional): number of saved checkpoints.
                Defaults to 1.
            checkpoint_interval (int, optional): intervals after which
                checkpoints is saved. Defaults to 10.
            validation_interval (int, optional):  intervals after which validation
                calculation is saved. Defaults to 1.
            hooks (List[Hook], optional): hooks to customize training process.
                Defaults to [].
            best_labl (str, optional): best model's name label. Defaults to "".
            loss_is_normalized (bool, optional): if True, the loss per data point
                will be reported. Otherwise, the accumulated loss is reported.
                Defaults to True.
        """
        # set path
        self.model_path = model_path
        self.checkpoint_path = self.model_path.joinpath("checkpoints")
        self.best_model = self.model_path.joinpath(f"best_model_{best_label}")
        # set dataloader
        self.train_loader = train_loader
        self.val_loader = val_loader
        # set training settings
        self.n_epoch = n_epoch
        self.device = device
        self.optimizer_list = optimizer_list
        self.scheduler = scheduler
        self.keep_n_checkpoints = keep_n_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.validation_interval = validation_interval
        self.hooks = hooks
        self.loss_is_normalized = loss_is_normalized
        # private attribute
        self._model = model
        self._stop = False
        self._stop_by = None
        # get newest checkpoint
        if self.checkpoint_path.exists():
            self.restore_checkpoint()
        else:
            self.checkpoint_path.mkdir()
            self.epoch = 0
            self.step = 0
            self.best_loss = float("inf")
            self.store_checkpoint()

    def _check_is_parallel(self):
        return True if isinstance(self._model, torch.nn.DataParallel) else False

    def _load_model_state_dict(self, state_dict):
        if self._check_is_parallel():
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)

    def _optimizer_to(self, device):
        for optimizer in self.optimizer_list:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

    @property
    def state_dict(self):
        state_dict = {
            "epoch": self.epoch,
            "step": self.step,
            "best_loss": self.best_loss,
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizer_list],
            "schedulers": None
            if self.scheduler is None
            else [scheduler.state_dict() for scheduler in self.scheduler],
            "hooks": [h.state_dict for h in self.hooks],
        }
        if self._check_is_parallel():
            state_dict["model"] = self._model.module.state_dict()
        else:
            state_dict["model"] = self._model.state_dict()
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        self.best_loss = state_dict["best_loss"]
        for op, s in zip(self.optimizer_list, state_dict["optimizers"]):
            op.load_state_dict(s)
        if self.scheduler is None:
            self.scheduler = None
        else:
            for sche, s in zip(self.scheduler, state_dict["schedulers"]):
                sche.load_state_dict(s)
        for h, s in zip(self.hooks, state_dict["hooks"]):
            h.state_dict = s
        self._load_model_state_dict(state_dict["model"])

    def store_checkpoint(self):
        # save Training object
        chkpt = self.checkpoint_path.joinpath(f"checkpoint-{self.epoch}.pth.tar")
        torch.save(self.state_dict, chkpt)
        # remove old chechpoint
        chpts = [
            str(f.name)
            for f in self.checkpoint_path.iterdir()
            if str(f).endswith(".pth.tar")
        ]
        if len(chpts) > self.keep_n_checkpoints:
            chpt_epochs = [int(f.split(".")[0].split("-")[-1]) for f in chpts]
            sidx = np.argsort(chpt_epochs)
            for i in sidx[: -self.keep_n_checkpoints]:
                self.checkpoint_path.joinpath(chpts[i]).unlink()

    def restore_checkpoint(self, epoch=None):
        # get newest checkpoint
        if epoch is None:
            epoch = max(
                [
                    int(str(f).split(".")[0].split("-")[-1])
                    for f in self.checkpoint_path.iterdir()
                    if str(f.name).startswith("checkpoint")
                ]
            )
        # load exist newest checkpoint
        chkpt = self.checkpoint_path.joinpath(f"checkpoint-{epoch}.pth.tar")
        self.state_dict = torch.load(chkpt)

    def train(
        self,
        batch: int,
        train_step: Callable[[Any], Tuple[List[torch.Tensor]]],
        val_step: Optional[Callable[[Any], Tuple[List[torch.Tensor]]]] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Training the model.

        Args:
            batch (int): number of batch.
            train_step (Callable): one training step which should include
                pre-processing of the batch (ex: send the tensor to the device),
                input to the model, and calculation of loss.
                This function returns loss list and predicted value list.
            val_step (Callbale or None): if None, using same step for training.
                Defaults to None.
            verbose (bool): verbosity of showing result of training step.
                Defaults to False.

        Note:
            Depending on the `hooks`, training can stop earlier than `n_epoch`.
        """
        if val_step is None:
            val_step = train_step

        self._model.to(self.device)
        self._optimizer_to(self.device)
        self._stop = False

        for h in self.hooks:
            h.on_train_begin(self)

        try:
            for epoch in range(self.n_epoch):
                # count the current epoch
                self.epoch += 1

                if verbose:
                    print(f"epoch {epoch+1} start")

                for h in self.hooks:
                    h.on_epoch_begin(self)

                if self._stop:
                    # decrease self.epoch if training is aborted on epoch begin
                    self.epoch -= 1
                    break

                # Training
                train_loss_sum_list = []
                n_train = 0
                self._model.train()
                for train_batch in self.train_loader:
                    # accumulate the number of data
                    n_train += batch

                    for optimizer in self.optimizer_list:
                        optimizer.zero_grad()

                    for h in self.hooks:
                        h.on_batch_begin(self)

                    # call training step
                    loss_list, result_list = train_step(train_batch, self._model)

                    # train loss accumulation
                    if n_train == batch:
                        for tr_loss in loss_list:
                            train_loss_sum_list.append(tr_loss.detach().clone().cpu())
                    else:
                        for i, tr_loss in enumerate(loss_list):
                            train_loss_sum_list[i] += tr_loss.detach().clone().cpu()

                    for loss, optimizer in zip(loss_list, self.optimizer_list):
                        loss.backward()
                        optimizer.step()

                    self.step += 1

                    for h in self.hooks:
                        h.on_batch_end(self, batch, train_batch, result_list, loss_list)

                    if self._stop:
                        break

                if self.scheduler is not None:
                    for scheduler in self.scheduler:
                        scheduler.step()

                # weighted average over batches
                if self.loss_is_normalized:
                    for i, _ in enumerate(train_loss_sum_list):
                        train_loss_sum_list[i] /= n_train

                if self.epoch % self.checkpoint_interval == 0:
                    self.store_checkpoint()

                if verbose:
                    print("Training loss:")
                    for i, loss in enumerate(train_loss_sum_list):
                        print(f"\t {i}: {loss:.3f}")

                # Validation
                self._model.eval()
                if self.epoch % self.validation_interval == 0 or self._stop:

                    for h in self.hooks:
                        h.on_validation_begin(self)

                    val_loss_sum_list = []
                    n_val = 0
                    for val_batch in self.val_loader:
                        # append batch_size
                        n_val += batch

                        for h in self.hooks:
                            h.on_validation_batch_begin(self)

                        # call val_step
                        val_loss_list, val_result_list = val_step(val_batch, self._model)

                        # val loss accumulation
                        if n_val == batch:
                            for val_loss in val_loss_list:
                                val_loss_sum_list.append(val_loss.detach().clone().cpu())
                        else:
                            for i, val_loss in enumerate(val_loss_list):
                                val_loss_sum_list[i] += val_loss.detach().clone().cpu()

                        for h in self.hooks:
                            h.on_validation_batch_end(
                                self, batch, val_batch, val_result_list, val_loss_list
                            )

                    # weighted average over batches
                    if self.loss_is_normalized:
                        for i, _ in enumerate(val_loss_sum_list):
                            val_loss_sum_list[i] /= n_val

                    if verbose:
                        print("validation loss:")
                        for i, loss in enumerate(val_loss_sum_list):
                            print(f"\t {i}: {loss:.3f}")

                    mean_val_loss = 0
                    for val_loss_sum in val_loss_sum_list:
                        mean_val_loss += val_loss_sum
                    mean_val_loss /= len(val_loss_sum_list)

                    if self.best_loss > mean_val_loss:
                        self.best_loss = mean_val_loss
                        torch.save(self._model, self.best_model)
                        if verbose:
                            print(f"model is saved in epoch {epoch+1}")

                    for h in self.hooks:
                        h.on_validation_end(self, mean_val_loss)

                for h in self.hooks:
                    h.on_epoch_end(self)

                if verbose:
                    print("-" * 50)
                    print("")

                if self._stop:
                    break

            for h in self.hooks:
                h.on_train_ends(self)

            if verbose:
                print("-" * 50)
                if self._stop:
                    print(f"Training is stopped by {self._stop_by}.")
                    print(f"Training stopped with epoch {epoch+1}")
                else:
                    print(f"Training completed with epoch {epoch+1}")
                print("-" * 50)

            # store checkpoints
            self.store_checkpoint()

        except Exception as e:
            for h in self.hooks:
                h.on_train_failed(self)

            raise e

        return self.best_loss
