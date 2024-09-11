import torch
import numpy as np
from configs.config import Configurable


class MultiOptionNormalize(Configurable):
    required_keys = ['value_sup', 'value_inf', 'rr_transform', 'normalization']

    def __call__(self, sample):
        if not isinstance(sample, torch.Tensor):
            raise TypeError(f"Input sample should be a torch tensor. Got {type(sample)}.")
        if sample.ndim < 3:
            raise ValueError(
                f"Expected sample to be a tensor image of size (..., C, H, W). Got tensor.size() = {sample.size()}."
            )

        # Transforming rain rates to logits (iterative transforms)
        for _ in range(self.rr_transform['log_transform_iteration']):
            sample[self.value_inf['rr']] = torch.log(1 + sample[self.value_inf['rr']])

        # Randomly symmetrizing rain rates around 0 (50% of rain rates are negative)
        if self.rr_transform['symetrization'] and np.random.random() <= 0.5:
            sample[self.value_inf['rr']] = -sample[self.value_inf['rr']]

        # Adding random noise (AT RUNTIME) to rain rates below a certain threshold
        if self.rr_transform['gaussian_std'] != 0:
            gaussian_std_map = np.random.choice([-1, 1], size=sample.shape[1:], p=[0.5, 0.5]) * self.rr_transform[
                'gaussian_std']
            gaussian_noise = np.random.normal(0, self.rr_transform['gaussian_std'], size=sample.shape[1:]) % \
                             self.rr_transform['gaussian_std']
            mask_no_rr = sample[self.value_inf['rr']] <= self.rr_transform['gaussian_std']
            sample[self.value_inf['rr']] += torch.from_numpy(gaussian_noise * mask_no_rr)

        # Performing different types of normalization (centering around mean or capping min-max/quantiles)
        if self.normalization['func'] == 'mean':
            sample = (sample - self.value_inf) / self.value_sup
        elif self.normalization['func'] in ['minmax', 'quant']:
            sample = -1 + 2 * ((sample - self.value_inf) / (self.value_sup - self.value_inf))

        return sample

    def denorm(self, sample):
        # Revert the __call__ function to produce "physical space" samples
        # Sample can be batched, and should be either of shape N x C X H x W or C x H x W

        # Non-batched ops
        if sample.ndim == 3:
            # Reverting log transforms
            for _ in range(self.rr_transform['log_transform_iteration']):
                sample[self.value_inf['rr']] = torch.exp(sample[self.value_inf['rr']]) - 1

            # Reverting symmetrization
            if self.rr_transform['symetrization']:
                sample[self.value_inf['rr']] = -sample[self.value_inf['rr']]

            # Reverting Gaussian noise by setting below threshold to 0
            if self.rr_transform['gaussian_std'] != 0:
                mask_no_rr = sample[self.value_inf['rr']] <= self.rr_transform['gaussian_std']
                sample[self.value_inf['rr']] = torch.abs(sample[self.value_inf['rr']] * (1 - mask_no_rr))

        # Batched ops
        else:
            # Reverting log transforms
            for _ in range(self.rr_transform['log_transform_iteration']):
                sample[:, self.value_inf['rr']] = torch.exp(sample[:, self.value_inf['rr']]) - 1

            # Reverting symmetrization
            if self.rr_transform['symetrization']:
                sample[:, self.value_inf['rr']] = torch.abs(sample[:, self.value_inf['rr']])

            # Reverting Gaussian noise by setting below threshold to 0
            if self.rr_transform['gaussian_std'] != 0:
                mask_no_rr = sample[:, self.value_inf['rr']] <= self.rr_transform['gaussian_std']
                sample[:, self.value_inf['rr']] = torch.abs(sample[:, self.value_inf['rr']] * (1 - mask_no_rr))

        # Reverting normalizations
        if self.normalization['func'] == 'mean':
            sample = sample * self.value_sup + self.value_inf
        elif self.normalization['func'] in ['minmax', 'quant']:
            sample = self.value_inf + 0.5 * (self.value_sup - self.value_inf) * ((sample + 1.0))

        return sample
