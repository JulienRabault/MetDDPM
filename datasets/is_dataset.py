import os
import re

import numpy as np

import torchvision.transforms as transforms

from datasets.base_dataset import BaseDataset, var_dict
from datasets.multi_option_normalize import MultiOptionNormalize


class ISDataset(BaseDataset):
    required_keys = ['crop', 'var_indexes']

    def __init__(self, *args, **kwargs):
        super().__init__()
        if hasattr(self, 'guiding_col'):
            self.guided = True
            self.guiding_col = self.guiding_col
        else:
            self.guided = False

    def prepare_transformations(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.value_sup, self.value_inf),
            ])

    def __getitem__(self, index):
        file_name = self.labels.iloc[index, 0]
        sample = self.file_to_torch(file_name)
        sample_id = re.search(r"\d+", file_name).group()
        if self.guided:
            ensemble_id = self.labels.loc[index, self.guiding_col]
            # TODO : a opti
            group = self.labels[self.labels[self.guiding_col] == ensemble_id]
            group_ensemble = group[group['Name'] != self.labels.iloc[index, 0]]
            row = group_ensemble.sample(n=1)
            ens = row['Name'].values[0]
            condition = self.file_to_torch(ens)
            return {"sample_id": sample_id, "Tensor": {"img": sample, "condition": condition}}
        else:
            return {"sample_id": sample_id, "Tensor": {"img": sample}}

    def __len__(self):
        return len(self.labels)

    def inversion_transforms(self):
        detransform_func = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0] * len(self.config.var_indexes),
                    std=[1 / el for el in self.value_sup],
                ),
                transforms.Normalize(
                    mean=[-el for el in self.value_inf],
                    std=[1.0] * len(self.config.var_indexes),
                ),
            ]
        )
        return detransform_func


class rrDataset(ISDataset):
    required_keys = ['rr_transform', 'normalization']

    def __init__(self, config, path, csv_file):
        """
        Initialize the rrISDataset.
            This subclasses ISDataset and overwrites prepare_transformations / init_normalization methods
            because there are many ways we can desire to calibrate the rain
        Args:
            config: Configuration settings.
            path (str): Directory path containing data.
            csv_file (str): CSV file containing labels and information.

        """
        super().__init__(config, path, csv_file, add_coords=False)

    def prepare_tranformations(self):
        transformations = []
        normalization = self.dataset_config.normalization["func"]
        if normalization != "None":
            if self.dataset_config.rr_transform["symetrization"]:
                if normalization == "means":
                    # mean of rr is 0
                    self.value_inf[var_dict["rr"]] = np.zeros_like(
                        self.value_inf[var_dict["rr"]]
                    )
                elif normalization == "minmax":
                    # min of 'negative rain' is -max
                    self.value_inf[var_dict["rr"]] = -self.value_sup[
                        var_dict["rr"]
                    ]
        transformations.append(transforms.ToTensor())
        transformations.append(
            MultiOptionNormalize(
                self.value_sup,
                self.value_inf,
                self.dataset_config,
                self.config,
            )
        )
        return transformations

    def inversion_transforms(self):
        detransform_func = MultiOptionNormalize(
            self.value_sup, self.value_inf, self.dataset_config, self.config
        ).denorm
        return detransform_func

    def init_normalization(self):
        normalization_func = self.dataset_config.normalization["func"]
        if normalization_func == "mean":
            stds, means = self.load_stat_files(
                normalization_func, "std", "mean"
            )
            return stds[self.VI] * 1.0 / 0.95, means[self.VI]

        if normalization_func == "minmax":
            maxs, mins = self.load_stat_files(normalization_func, "max", "min")
            return maxs[self.VI], mins[self.VI]

        if normalization_func == "quant":
            q99, q01 = self.load_stat_files(normalization_func, "q99", "q01")
            return q99[self.VI], q01[self.VI]

        return None, None

    def load_stat_files(self, normalization_func, str_sup, str_inf):
        # Your normalization files should be name "[var]_[stat_version]_log_log_..._[ppx].npy" with:
        #   var: 'min', 'max' or 'mean', 'std' or 'Q01', 'Q99' or 'Q10', Q90'
        #   stat_version: an identifier for the stat file
        #   log_log...: 'log_' will be repeated log_transform_iteration times
        #   ppx: if the stats are per pixel, _ppx must be added at the end of the file
        norm_vars = []
        for name in (str_sup, str_inf):
            filename = f"{name}_{self.dataset_config.stat_version}"
            filename += (
                    "_log"
                    * self.dataset_config.rr_transform["log_transform_iteration"]
            )

            if self.dataset_config.normalization["per_pixel"]:
                filename += "_ppx"

            filename += ".npy"

            try:
                path = os.path.join(
                    self.data_dir, self.dataset_config.stat_folder, filename
                )
                norm_var = np.load(path).astype("float32")
            except FileNotFoundError as err:
                raise FileNotFoundError(
                    f"{name} file was not found at this location: {path}"
                )
            norm_vars.append(norm_var)
        return norm_vars
