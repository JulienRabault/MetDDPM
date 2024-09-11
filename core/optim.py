import inspect
import logging

import torch.optim as optim
from configs.config import Configurable, GlobalConfig


def get_optimizer_and_keys(type_name):
    """
    Get the optimizer class and its required keys for a given optimizer type name.
    Args:
        type_name (str): The name of the optimizer type.
    Returns:
        tuple: A tuple containing the optimizer class and its required keys, or (None, None) if the type name is not found.
    """
    optimizer_dict = get_all_optimizers()
    if type_name in optimizer_dict:
        return optimizer_dict[type_name]
    else:
        return None, None

def get_required_keys(scheduler_class):
    """
    Get the required keys for a given scheduler class by inspecting its `__init__` method.
    Args:
        scheduler_class (type): The scheduler class to inspect.
    Returns:
        list: A list of required keys for the scheduler class.
    """
    init_signature = inspect.signature(scheduler_class.__init__)
    required_keys = [param.name for param in init_signature.parameters.values() if
                     param.kind == param.POSITIONAL_OR_KEYWORD and param.name != "params" and param.name != "optimizer" and param.name != "self" and param.default is inspect._empty]
    return required_keys

def get_all_optimizers():
    """
    Generate a dictionary of all optimizer classes and their required keys available in torch.optim and custom optimizers.
    ...
    """
    optimizer_dict = {}

    for name, obj in optim.__dict__.items():
        if (
                isinstance(obj, type)
                and issubclass(obj, optim.Optimizer)
                and obj is not optim.Optimizer
        ):
            required_keys = get_required_keys(obj)
            optimizer_dict[name] = (obj, required_keys)

    for subclass in OptimizerBase.__subclasses__():
        optimizer_dict[subclass.__name__] = (subclass, subclass.required_keys)

    return optimizer_dict

class OptimizerBase(Configurable, optim.Optimizer):
    required_keys = []

    @classmethod
    def from_typed_config(cls, config_data, params, **kwargs):
        config_data = cls._safe_open(config_data)
        try:
            type_name = config_data['type']
        except KeyError:
            raise ValueError(f"Missing required key: type for class {cls.__name__} in config file for {cls.__name__}")

        optimizer_class, required_keys = get_optimizer_and_keys(type_name)
        if optimizer_class is None:
            raise Exception(f"Type {type_name} not found, please check the configuration file. "
                            f"List of available types: {[el.__name__ for el in cls.__subclasses__()]}")

        cls._check_config(config_data, typed=True, dynamic_keys=required_keys)
        instance = optimizer_class.__new__(optimizer_class)
        # remove type in config_data for
        config_data.pop('type', None)
        for key, value in config_data.items():
            if not hasattr(instance, key):
                setattr(instance, key, value)
                logging.debug(f"key: {key}, value: {value}")
        setattr(instance, 'global_config', GlobalConfig())
        instance.__init__(params, **config_data, **kwargs)
        return instance
