import logging

import torch.optim.lr_scheduler as lr_scheduler
from configs.config import Configurable, GlobalConfig
import inspect

EXCLUDE_SCHEDULERS = ["_LRScheduler", "LRScheduler", "StepLR", "MultiStepLR", "SequentialLR"]

class SchedulerBase(Configurable, lr_scheduler._LRScheduler):
    required_keys = []

    use_step_each_batch_list = ["CyclicLR", "OneCycleLR"]

    @classmethod
    def from_typed_config(cls, config_data, optim, **kwargs):
        config_data = cls._safe_open(config_data)
        try:
            type_name = config_data['type']
        except KeyError:
            raise ValueError(f"Missing required key: type for class {cls.__name__} in config file for {cls.__name__}")

        scheduler_class, required_keys = get_scheduler_and_keys(type_name)
        if scheduler_class is None:
            raise Exception(f"Type {type_name} not found, please check the configuration file. "
                            f"List of available types: {[el.__name__ for el in cls.__subclasses__()]}")

        cls._check_config(config_data, typed=True, dynamic_keys=required_keys)
        instance = scheduler_class.__new__(scheduler_class)
        # remove type in config_data for
        config_data.pop('type', None)
        for key, value in config_data.items():
            if not hasattr(instance, key):
                setattr(instance, key, value)
                logging.debug(f"key: {key}, value: {value}")
        setattr(instance, 'global_config', GlobalConfig())
        instance.__init__(optim, **config_data, **kwargs)
        if type_name in cls.use_step_each_batch_list:
            setattr(instance, "step_each_batch", True)
        else:
            setattr(instance, "step_each_batch", False)
        return instance

def get_all_schedulers(excludes=[]):
    """
    Generate a dictionary of all scheduler classes and their required keys available in torch.optim.lr_scheduler.
    Args:
        exclude_classes (list): A list of scheduler class names to exclude from the dictionary.
    Returns:
        dict: A dictionary where keys are scheduler class names and values are tuples containing the scheduler class and its required keys.
    """
    scheduler_dict = {}

    # Récupération de tous les objets du module lr_scheduler
    schedulers = inspect.getmembers(lr_scheduler, inspect.isclass)
    for name, cls in schedulers:
        if issubclass(cls, lr_scheduler.LRScheduler) and name not in excludes and name != "SchedulerBase":
            required_keys = get_required_keys(cls)
            scheduler_dict[name] = (cls, required_keys)
    # Ajout des sous-classes de SchedulerBase
    for subclass in SchedulerBase.__subclasses__():
        required_keys = subclass.required_keys
        scheduler_dict[subclass.__name__] = (subclass, required_keys)
    return scheduler_dict

def get_scheduler_and_keys(type_name):
    """
    Get the scheduler class and its required keys for a given type name.
    Args:
        type_name (str): The name of the scheduler type.
    Returns:
        tuple: A tuple containing the scheduler class and its required keys, or (None, None) if the type name is not found.
    """
    scheduler_dict = get_all_schedulers()
    if type_name in scheduler_dict:
        return scheduler_dict[type_name]
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
