import pathlib
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn

__all__ = ["Trainer"]


class Trainer:
    def __init__(
        self,
        model_path: pathlib.Path,
        model: nn.Module,
        n_epoch: int,
        device: torch.device,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Union[torch.optim.lr_scheduler._LRScheduler, None],
        train_loader: torch.utils.data.dataloader.Dataloader,
        val_loader: torch.utils.data.dataloader.Dataloader,
        regularization: bool = False,
        reg_lambda: Union[float, None] = None,
        keep_n_checkpoints: int = 1,
        checkpoint_interval: int = 10,
        validation_interval: int = 1,
        hooks: list = [],
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
            loss_fn (Callable): loss function which return loss value.
            optimizer (torch.optim.Optimizer): training optimizer.
            scheduer (torch.optim.lr_schduler._LRScheduler or None): training LR
                scheduler.
            train_loader (torch.utils.data.Dataloader): data loader for training set.
            val_loader (torch.utirls.data.Dataloader): data loader for validation set.
            regularization (bool, optinal): if True, add a regularization term
                to the loss. Defaults to False.
            reg_lambda (float or None): L1 regularization lambda.
                Defaults to None
            keep_n_checkpoints (int, optional): number of saved checkpoints.
                Defaults to 1.
            checkpoint_interval (int, optional): intervals after which
                checkpoints is saved. Defaults to 10.
            validation_interval (int, optional):  intervals after which validation
                calculation is saved. Defaults to 1.
            hooks (list, optional): hooks to customize training process.
                Defaults to [].
            best_labl (str, optional): best model's name label. Defaults to "".
            loss_is_normalized (bool, optional): if True, the loss per data point
                will be reported. Otherwise, the accumulated loss is reported.
                Defaults to True.
        """
        # set path
        self.model_path = model_path
        self.checkpoint_path = self.model_path.join("checkpoints")
        self.best_model = self.model_path.join(f"best_model_{best_label}")
        # set dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        # set training settings
        self.n_epoch = n_epoch
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.regularization = regularization
        if self.regularization:
            assert self.reg_lambda is not None, "Please set 'reg_lambda' parameter."
        self.reg_lambda = reg_lambda
        self.keep_n_checkpoints = keep_n_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.validation_interval = validation_interval
        self.hooks = hooks
        self.loss_is_normalized = loss_is_normalized
        # private attribute
        self._model = model
        self._stop = False
        # get newest checkpoint
        if self.checkpoint_path.exits():
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
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    @property
    def state_dict(self):
        state_dict = {
            "epoch": self.epoch,
            "step": self.step,
            "best_loss": self.best_loss,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
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
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler = (
            None
            if self.scheduler is None
            else self.scheduler.load_state_dict(state_dict["scheduler"])
        )
        self._load_model_state_dict(state_dict["model"])
        for h, s in zip(self.hooks, state_dict["hooks"]):
            h.state_dict = s

    def store_checkpoint(self):
        # save Training object
        chkpt = self.checkpoint_path.join(f"checkpoint-{self.epoch}.pth.tar")
        torch.save(self.state_dict, chkpt)
        # remove old chechpoint
        chpts = [
            f for f in self.checkpoint_path.iterdir() if str(f).endswith(".pth.tar")
        ]
        if len(chpts) > self.keep_n_checkpoints:
            chpt_epochs = [int(f.split(".")[0].split("-")[-1]) for f in chpts]
            sidx = np.argsort(chpt_epochs)
            for i in sidx[: -self.keep_n_checkpoints]:
                self.checkpoint_path.join(chpts[i]).unlink()

    def restore_checkpoint(self, epoch=None):
        # get newest checkpoint
        if epoch is None:
            epoch = max(
                [
                    int(f.split(".")[0].split("-")[-1])
                    for f in self.checkpoint_path.iterdir()
                    if str(f).startswith("checkpoint")
                ]
            )
        # load exist newest checkpoint
        chkpt = self.checkpoint_path.join(f"checkpoint-{str(epoch)}.pth.tar")
        self.state_dict = torch.load(chkpt)

    def train(self):
        """
        Training the model.

        Note:
            Depending on the `hooks`, training can stop earlier than `n_epoch`.
        """
        self._model.to(self.device)
        self._optimizer_to(self.device)
        self._stop = False

        for h in self.hooks:
            h.on_train_begin(self)

        try:
            for _ in range(self.n_epoch):
                # count the current epoch
                self.epoch += 1

                for h in self.hooks:
                    h.on_epoch_begin(self)

                if self._stop:
                    # decrease self.epoch if training is aborted on epoch begin
                    self.epoch -= 1
                    break

                # Training
                self._model.train()
                if self.device.type == "cuda":
                    scaler = torch.cuda.amp.GradScaler()
                for x_train, y_train in self.train_loader:
                    self.optimizer.zero_grad()

                    for h in self.hooks:
                        h.on_batch_begin(self, x_train, y_train)

                    # move input to gpu, if needed
                    x_train = x_train.to(self.device)
                    y_train = y_train.to(self.device)
                    with torch.cuda.amp.autocast():
                        result = self._model(x_train)
                        loss = self.loss_fn(result, y_train)

                    # L1 regularization
                    if self.regularization:
                        l1_reg = torch.tensor(0.0, requires_grad=True)
                        for param in self._model.parameters():
                            if param.requires_grad:
                                l1_reg = l1_reg + torch.norm(param, 1)
                        loss = loss + self.reg_lambda * l1_reg

                    if self.device.type == "cuda":
                        # Calls backward() on scaled loss to create scaled gradients.
                        scaler.scale(loss).backward()
                        # scaler.step() first unscales the gradients of
                        # the optimizer's assigned params.
                        scaler.step(self.optimizer)
                        # Updates the scale for next iteration.
                        scaler.update()
                        # after_model = copy.deepcopy(self._model)
                    else:
                        loss.backward()
                        self.optimizer.step()

                    for h in self.hooks:
                        h.on_batch_end(self, x_train, y_train, result, loss)

                    if self._stop:
                        break

                if self.epoch % self.checkpoint_interval == 0:
                    self.store_checkpoint()

                # Validation
                self._model.eval()
                if self.epoch % self.validation_interval == 0 or self._stop:

                    for h in self.hooks:
                        h.on_validation_begin(self)

                    val_loss = 0.0
                    n_val = 0
                    for x_val, y_val in self.val_loader:
                        # append batch_size
                        vsize = x_val.size(0)
                        n_val += vsize

                        for h in self.hooks:
                            h.on_validation_batch_begin(self)

                        # move input to gpu, if needed
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)

                        val_result = self._model(x_val)
                        val_batch_loss = (
                            self.loss_fn(val_result, y_val).data.cpu().numpy()
                        )
                        if self.loss_is_normalized:
                            val_loss += val_batch_loss * vsize
                        else:
                            val_loss += val_batch_loss

                        for h in self.hooks:
                            h.on_validation_batch_end(
                                self, x_val, y_val, val_result, val_loss
                            )

                    # weighted average over batches
                    if self.loss_is_normalized:
                        val_loss /= n_val

                    if self.best_loss > val_loss:
                        self.best_loss = val_loss
                        torch.save(self._model, self.best_model)

                    for h in self.hooks:
                        h.on_validation_end(self, val_loss)

                for h in self.hooks:
                    h.on_epoch_end(self)

                if self._stop:
                    break

            for h in self.hooks:
                h.on_train_ends(self)

            # store checkpoints
            self.store_checkpoint()

        except Exception as e:
            for h in self.hooks:
                h.on_train_failed(self)

            raise e
