import abc
import inspect

from configs.config import Configurable
from denoising_diffusion_pytorch import GaussianDiffusion, Unet


class BaseModel(abc.ABC, Configurable):
    guided = False

    @abc.abstractmethod
    def sample(self, batch_size, return_all_timesteps=False, condition=None):
        pass

    @abc.abstractmethod
    def forward(self, img, *args, **kwargs):
        pass

    @abc.abstractmethod
    def is_guided(self):
        pass


class DiffusionModel(BaseModel, GaussianDiffusion):

    required_keys = ['umodel']

    def __init__(self, *args, **kwargs):
        if 'channels' not in self.umodel:
            self.umodel['channels'] = len(self.global_config.config['dataset']['var_indexes'])
        umodel = Unet(**self.umodel,
                      self_condition=self.guided,
                      )
        meta_data = self.to_config(exclude=['umodel', 'global_config']).copy()
        # TODO: probleme car la signature des modeles n'a pas args et kwargs
        init_signature = inspect.signature(GaussianDiffusion.__init__)
        valid_kwargs = {k: v for k, v in (meta_data | kwargs).items() if k in init_signature.parameters}
        super().__init__(umodel, **valid_kwargs)
        # GaussianDiffusion not need tuple for image_size
        self.image_size = self.image_size[0] if isinstance(self.image_size, tuple) else self.image_size

    def is_guided(self):
        return self.guided
