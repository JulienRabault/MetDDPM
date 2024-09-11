import os
from os import cpu_count

import torch
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from torch import distributed as dist, nn

from configs.config import GlobalConfig
from datasets.base_dataset import BaseDataset
from diffusion_models.custom_model import BaseModel
from configs.config import Configurable
from core.optim import OptimizerBase

from core.scheduler import SchedulerBase

from utils.distributed import get_rank_num, is_main_gpu, synchronize, get_rank
from core.trackers import Trackers


class Trainer(Configurable):
    required_keys = ["model", "optimizer", "scheduler", "dataset"]

    def __init__(self, *args, **kwargs) -> None:
        self.gpu_id = get_rank()
        self.dataset = BaseDataset.from_typed_config(self.dataset)
        self.model = BaseModel.from_typed_config(self.model, image_size=self.dataset.image_size)
        assert self.model.is_guided() == self.dataset.is_guided(), f"Model and dataset must have the same guided status\n" \
                                                                   f"For guided dataset define guiding_col in dataset config"
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     pin_memory=True,
                                     persistent_workers=True,
                                     # non_blocking=True,
                                     shuffle=not torch.cuda.device_count() >= 2,
                                     num_workers=os.cpu_count() if not hasattr(self, 'num_workers') is None else self.num_workers,
                                     sampler=(
                                         DistributedSampler(
                                             self.dataset, rank=get_rank_num(), shuffle=False, drop_last=False
                                         )
                                         if torch.cuda.device_count() >= 2
                                         else None
                                     ), )
        self.optimizer = OptimizerBase.from_typed_config(self.optimizer.copy(), self.model.parameters())
        self.scheduler = SchedulerBase.from_typed_config(self.scheduler.copy(), self.optimizer,
                                                         steps_per_epoch=len(self.dataloader))
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._move_to_device(self.model)
        if torch.cuda.device_count() >= 2:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.gpu_id], output_device=self.gpu_id
            )
            self.model = self.model.module
        self.optimizer = self._move_to_device(self.optimizer)
        self.scheduler = self._move_to_device(self.scheduler)
        self.epochs_run = 0
        self.best_loss = float("inf")
        self.guided_diffusion = hasattr(self.dataset, "guiding_col")
        self.tracker = Trackers(self.tracking, self.global_config["output_dir"])

    def train(self):
        """
        Start the training process.
        Returns:
            None
        """
        if is_main_gpu():
            self.tracker.init()
            loop = tqdm(
                range(self.epochs_run, self.epochs),
                desc=f"Training...",
                unit="epoch",
                postfix="",
            )
        else:
            loop = range(self.epochs_run, self.epochs)

        for epoch in loop:
            avg_loss = self._run_epoch(epoch)
            if is_main_gpu():
                loop.set_postfix_str(
                    f"Epoch loss : {avg_loss:.5f} | Lr : {(self.optimizer.param_groups[0]['lr']):.6f}"
                )
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self._save_snapshot(
                        epoch,
                        os.path.join(self.global_config.output_dir, f"{self.global_config.run_name}", "best.pt", ),
                        avg_loss,
                    )
                if epoch % self.any_time == 0.0:
                    self._save_snapshot(
                        epoch,
                        os.path.join(self.global_config.output_dir, f"{self.global_config.run_name}",
                                     f"save_{epoch}.pt", ),
                        avg_loss,
                    )
                log = {
                    "avg_loss": avg_loss.item(),
                    "lr": (
                        self.optimizer.param_groups[0]["lr"]
                    ),
                }
                self.tracker.log(epoch, log)

                self._save_snapshot(
                    epoch,
                    os.path.join(self.global_config.output_dir, f"{self.global_config.run_name}", "last.pt", ),
                    avg_loss,
                )

        if is_main_gpu():
            self.tracker.finish()

            self.logger.info(
                f"Training finished , best loss : {self.best_loss:.6f}, lr : f{(self.optimizer.param_groups[0]['lr']):.6f}, "
                f"saved at {os.path.join(self.global_config.output_dir, f'{self.global_config.config.run_name}', 'best.pt')}"
            )

    def _run_epoch(self, epoch):
        """
        Run a training epoch.
        Args:
            epoch (int): Current epoch number.
        Returns:
            float: Average loss for the epoch.
        """

        iters = len(self.dataloader)
        if dist.is_initialized():
            self.dataloader.sampler.set_epoch(epoch)
        total_loss = 0
        # tqdm provides a progress bar during training
        loop = tqdm(
            enumerate(self.dataloader),
            total=iters,
            desc=f"Epoch {epoch}/{self.epochs + self.epochs_run}",
            unit="batch",
            leave=False,
            postfix="",
            disable=not is_main_gpu(),
        )
        for i, batch in loop:
            batch_tensor = self._move_to_device(batch["Tensor"])
            loss = self._run_batch(batch_tensor)
            total_loss += loss

            if is_main_gpu():
                loop.set_postfix_str(f"Loss : {total_loss / (i + 1):.6f}")

            if self.log_by_iteration:
                log = {
                    "avg_loss_it": loss.item(),
                    "lr_it": (
                        self.optimizer.param_groups[0]["lr"]
                    ),
                }
                self.tracker.log(i, log)

            if self.scheduler.step_each_batch:
                self.scheduler.step()

        self.logger.debug(
            f"Epoch {epoch} | Batchsize: {self.batch_size} | Steps: {len(self.dataloader) * epoch} | "
            f"Last loss: {total_loss / len(self.dataloader)} | "
            f"Lr : {self.optimizer.param_groups[0]['lr']}"
        )
        self.scheduler.step(total_loss / len(self.dataloader))

        if epoch % self.any_time == 0.0 and is_main_gpu():
            s = min(4, len(batch_tensor["condition"]))
            args = {"batch_size": s}
            if self.model.is_guided():
                args |= {"condition": batch_tensor["condition"][:s]}
            samples = self.model.sample(**args)
            self.plot_grid(f"samples_grid_{epoch}.jpg", samples.cpu().numpy())

        if epoch % self.any_time == 0.0:
            synchronize()

        return total_loss / len(self.dataloader)

    def _run_batch(self, batch):
        """
        Run a single training batch.
        Args:
            batch: Input batch for training.
        Returns:
            float: Loss value for the batch.
        """
        self.optimizer.zero_grad()
        loss = self.model(**batch)
        loss.backward()
        self.optimizer.step()
        loss = loss.detach().cpu()
        return loss

    def _save_snapshot(self, epoch, path, loss):
        """
        Save a snapshot of the training progress.
        Args:
            epoch (int): Current epoch number.
            path: Path to save the snapshot.
            loss: Loss value at the epoch.
        Returns:
            None
        """
        config_metadata = self.dataset.to_config().copy()
        model_config = self.model.to_config(parents=False).copy()
        model_config.update({"type": self.model.__class__.__name__, 'image_size': config_metadata["image_size"]})
        config_metadata.update({"type": self.dataloader.dataset.__class__.__name__})
        del config_metadata["data_dir"]
        del config_metadata["csv_file"]
        snapshot = {
            "MODEL": {
                "MODEL_STATE": self.model.state_dict(),
                "MODEL_CONFIG": model_config,
            },
            "TRAIN_INFO": {
                "EPOCHS_RUN": epoch,
                "BEST_LOSS": loss,
                "OPTIMIZER_TYPE": self.optimizer.__class__.__name__,
                "OPTIMIZER_STATE": self.optimizer.state_dict(),
                "SCHEDULER_TYPE": self.scheduler.__class__.__name__,
                "SCHEDULER_STATE": self.scheduler.state_dict(),
            },
            "DATAPROCESS": config_metadata,
            "GLOBAL_CONFIG": self.global_config.to_dict(),
        }
        torch.save(snapshot, path)
        self.logger.info(
            f"Epoch {epoch} | Training snapshot saved at {path} | Loss: {loss}"
        )

    def _move_to_device(self, obj_torch):
        """
        Move the model to a device.
        Args:
            obj_torch: obj_torch to move the device.
        Returns:
            None
        """
        if isinstance(obj_torch, dict):
            return {k: self._move_to_device(v) for k, v in obj_torch.items()}
        elif isinstance(obj_torch, list):
            return [self._move_to_device(v) for v in obj_torch]
        elif isinstance(obj_torch, torch.Tensor):
            return obj_torch.to(self.gpu_id)
        else:
            try:
                return obj_torch.to(self.gpu_id)
            except:
                return obj_torch

    @classmethod
    def from_snapshot(cls, snapshot_path):
        """
        Load a snapshot of the training progress.
        Args:
            snapshot_path: Path to the snapshot.
        Returns:
            Trainer: Trainer object with loaded snapshot.
        """
        snapshot = torch.load(snapshot_path)
        global_config = GlobalConfig(config=snapshot["GLOBAL_CONFIG"])
        trainer = cls.from_config(global_config.to_dict()["config"])
        trainer.model.load_state_dict(snapshot["MODEL"]["MODEL_STATE"])
        trainer.optimizer.load_state_dict(snapshot["TRAIN_INFO"]["OPTIMIZER_STATE"])
        trainer.scheduler.load_state_dict(snapshot["TRAIN_INFO"]["SCHEDULER_STATE"])
        trainer.epochs_run = snapshot["TRAIN_INFO"]["EPOCHS_RUN"]
        trainer.best_loss = snapshot["TRAIN_INFO"]["BEST_LOSS"]
        trainer.logger.info(
            f"Snapshot loaded from {snapshot_path} | Epochs run: {trainer.epochs_run} | Best loss: {trainer.best_loss}"
        )
        return trainer

    def plot_grid(self, file_name, np_img):
        """
        Plot a grid of images.
        Args:
            file_name (str): Name of the file to save the plot.
            np_img (numpy.ndarray): Array of images to plot.
        """
        nb_image = len(np_img)
        if nb_image <= 1:
            return
        fig, axes = plt.subplots(
            nrows=min(6, nb_image),
            ncols=len(self.dataset.var_indexes),
            figsize=(10, 10),
        )
        for i in range(min(6, nb_image)):
            for j in range(len(self.dataset.var_indexes)):
                cmap = (
                    "viridis" if self.dataset.var_indexes[j] != "t2m" else "bwr"
                )
                image = np_img[i, j]
                if len(self.dataset.var_indexes) > 1 and min(6, nb_image) > 1:
                    im = axes[i, j].imshow(image, cmap=cmap, origin="lower")
                    axes[i, j].axis("off")
                    fig.colorbar(im, ax=axes[i, j])
                else:
                    im = axes[i].imshow(image, cmap=cmap, origin="lower")
                    axes[i].axis("off")
                    fig.colorbar(im, ax=axes[i])
        # Save the plot to the specified file path
        plt.savefig(
            os.path.join(f"{self.global_config.output_dir}", f"{self.global_config.run_name}", "samples", file_name),
            bbox_inches="tight",
        )
        plt.close()
