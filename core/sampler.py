import abc
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from datasets.base_dataset import BaseDataset
from diffusion_models.custom_model import BaseModel
from configs.config import Configurable, GlobalConfig
from utils.distributed import is_main_gpu, get_rank, get_rank_num


class Sampler(Configurable, abc.ABC):
    required_keys = ["snapshot_path"]

    def __init__(self,  *args, **kwargs):
        self.gpu_id = get_rank()
        self.load_snapshot(self.snapshot_path)
        self.model.to(self.gpu_id)
        if torch.cuda.device_count() >= 2:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.gpu_id], output_device=self.gpu_id
            )
            self.model = self.model.module
        self.output_dir = self.global_config.output_dir

    @abc.abstractmethod
    def sample(self, filename_format="_sample_{i}.npy"):
        pass

    def load_snapshot(self, snapshot_path):
        """
        Load the model from the given path.
        Args:
            snapshot_path (str): Path to the model file.
        Returns:
            None
        """
        self.snapshot = torch.load(snapshot_path, map_location=self.gpu_id)
        GlobalConfig(config=self.snapshot["GLOBAL_CONFIG"])
        self.model = BaseModel.from_typed_config(self.snapshot['MODEL']['MODEL_CONFIG'])
        self.model.load_state_dict(self.snapshot['MODEL']["MODEL_STATE"])
        self.model.eval()


class SimpleSampler(Sampler):
    required_keys = ["n_samples"]

    aliases = ["simple_sampler", "simple"]

    def __init__(self,  *args, **kwargs):
        super().__init__( *args, **kwargs)
        assert not self.model.is_guided(), "Model is guided, use GuidedSampler instead"

    @torch.no_grad()
    def sample(self, filename_format="_sample_{i}.npy"):

        if is_main_gpu():
            self.logger.info(f"Sampling {self.n_samples} images...")

        with tqdm(total=self.n_samples // self.config.batch_size, desc="Sampling ", unit="batch",
                  disable=not is_main_gpu()) as pbar:
            b = 0
            i = self.gpu_id if type(self.gpu_id) is int else 0
            while b < self.n_samples:
                batch_size = min(self.n_samples - b, self.batch_size)
                samples = self.model.sample(batch_size=batch_size)
                for s in samples:
                    filename = filename_format.format(i=str(i))
                    save_path = os.path.join(self.output_dir, self.config.run_name, "samples", filename)
                    np.save(save_path, s)
                    i += max(torch.cuda.device_count(), 1)
                b += batch_size
                pbar.update(1)

        self.logger.info(
            f"Sampling done. Images saved in {self.global_config.output_dir}/{self.global_config.run_name}/samples/")


class GuidedSampler(Sampler):
    required_keys = ["dataset"]

    aliases = ["guided_sampler", "guided"]

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

    def load_snapshot(self, snapshot_path):
        super().load_snapshot(snapshot_path)
        dataprocess = self.snapshot['DATAPROCESS']
        data_config = self.dataset
        dataprocess.update(data_config)
        self.dataset = BaseDataset.from_typed_config(dataprocess)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     pin_memory=True,
                                     persistent_workers=True,
                                     # non_blocking=True,
                                     shuffle=not torch.cuda.device_count() >= 2,
                                     num_workers=os.cpu_count() if not hasattr(self,'num_workers' ) is None else self.num_workers,
                                     sampler=(
                                         DistributedSampler(
                                             self.dataset, rank=get_rank_num(), shuffle=False, drop_last=False
                                         )
                                         if torch.cuda.device_count() >= 2
                                         else None
                                     ), )

    @torch.no_grad()
    def sample(self, filename_format="_sample_{i}.npy"):
        if is_main_gpu():
            self.logger.info(
                f"Sampling {len(self.dataloader) * self.batch_size * (torch.cuda.device_count() if torch.cuda.is_available() else 1)} images...")

        for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Sampling ",
                                     unit="batch"):
            batch_tensor = batch['Tensor']
            cond = batch_tensor['img'].to(self.gpu_id)
            ids = batch['sample_id']
            samples = self.model.sample(batch_size=len(cond), condition=cond)
            for s, img_id in zip(samples, ids):
                filename = filename_format.format(i=img_id)
                save_path = os.path.join(self.global_config.output_dir, self.global_config.run_name, "samples", filename)
                np.save(save_path, s)

        self.logger.info(f"Sampling done. Images saved in {self.global_config.output_dir}/{self.global_config.run_name}/samples/")
