import os
from abc import abstractmethod, ABC

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from configs.config import Configurable

var_dict = {
    "rr": 0,
    "u": 1,
    "v": 2,
    "t2m": 3,
    "orog": 4,
    "z500": 5,
    "t850": 6,
    "tpw850": 7,
}


class BaseDataset(Dataset, Configurable, ABC):
    required_keys = ['data_dir', 'csv_file', 'mean_file', 'max_file']

    guided = False

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.labels = pd.read_csv(self.csv_file, index_col=False)
        if "Unnamed: 0" in self.labels:
            self.labels = self.labels.drop("Unnamed: 0", axis=1)
        self.VI = [var_dict[var] for var in self.var_indexes]
        self.value_sup, self.value_inf = self.init_normalization()
        self.transforms = self.prepare_transformations()

    def is_guided(self):
        return self.guided

    def init_normalization(self):
        try:
            means = np.load(
                os.path.join(self.data_dir, self.mean_file)
            )[self.VI]
            maxs = np.load(os.path.join(self.data_dir, self.max_file))[
                self.VI
            ]
        except (FileNotFoundError, KeyError):
            try:
                means = np.load(self.mean_file)[self.VI]
                maxs = np.load(self.max_file)[self.VI]
            except (FileNotFoundError, KeyError):
                raise ValueError(
                    "The mean_file and max_file must be specropfied in the parser using --mean_file and --max_file options"
                )

        means = list(tuple(means))
        stds = list(tuple((1.0 / 0.95) * (maxs)))

        return stds, means

    @abstractmethod
    def prepare_transformations(self):
        raise NotImplementedError

    @abstractmethod
    def inversion_transforms(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    def file_to_torch(self, file_name):
        """
        Convert a file to a torch tensor.
        Args:
            file_name (str or list): Name of the file or list of file names.
        Returns:
            torch.Tensor: Torch tensor representing the sample.
        """
        if type(file_name) == list:
            file_name = file_name[0]
        sample_path = os.path.join(self.data_dir, file_name)
        sample = np.float32(np.load(sample_path + ".npy"))[
                 self.VI, self.crop[0]: self.crop[1], self.crop[2]: self.crop[3]
                 ]
        sample = sample.transpose((1, 2, 0))
        sample = self.transforms(sample)
        return sample
