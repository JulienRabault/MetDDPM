import logging
import json
import os
import yaml
from utils.distributed import synchronize
import datetime
from utils.distributed import get_rank_num


class GlobalConfig:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GlobalConfig, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, config=None):
        if config is not None:
            self.__dict__.update(config.copy())

    def __setitem__(self, name, value):
        if not isinstance(name, str):
            raise TypeError("GlobalConfig keys must be strings")
        self.__dict__.setdefault(name, None)
        self.__dict__[name] = value

    def __getitem__(self, name):
        if not isinstance(name, str):
            raise TypeError("GlobalConfig keys must be strings")
        if name not in self.__dict__:
            raise KeyError(f"GlobalConfig does not have key: {name},\n see :{self.__dict__}")
        return self.__dict__.get(name, None)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def to_dict(self):
        return self.__dict__


global_config = GlobalConfig()


class Configurable:
    """
    Base class for configurable objects.

    This class provides methods to load and validate configuration data from a YAML file or a dictionary. Subclasses
    should define `required_keys` and `aliases` class attributes. When a subclass is initialized with configuration
    data, the attributes defined in the configuration will be automatically set on the instance.
    """
    required_keys = []

    # aliases for the class name
    aliases = []

    logger = logging.getLogger(f"logddp_{get_rank_num()}")

    @classmethod
    def from_config(cls, config_data, *args, **kwargs):
        """of
        Create an instance  the class from configuration data.

        Args:
            config_data (str or dict): Configuration data in the form of a dictionary or a path to a YAML file.
            **kwargs: Additional keyword arguments to pass to the class constructor.

        Returns:
            instance: An instance of the class with attributes set according to the configuration data.
        """
        config_data = cls._safe_open(config_data)
        cls._check_config(config_data)
        instance = cls.__new__(cls)
        for key, value in config_data.items():
            if not hasattr(instance, key):
                setattr(instance, key, value)
                logging.debug(f"key: {key}, value: {value}")
        setattr(instance, 'global_config', GlobalConfig())
        instance.__init__(*args, **kwargs)
        return instance

    @classmethod
    def from_typed_config(cls, config_data, *args, **kwargs):
        """
        Create an instance of a subclass from typed configuration data.

        This method finds the correct subclass based on the 'type' key in the configuration data.

        Args:
            config_data (str or dict): Configuration data in the form of a dictionary or a path to a YAML file.
            **kwargs: Additional keyword arguments to pass to the class constructor.

        Returns:
            instance: An instance of the correct subclass with attributes set according to the configuration data.
        """
        config_data = cls._safe_open(config_data)
        try:
            type_name = config_data['type']
        except KeyError:
            raise ValueError(f"Missing required key: type for class {cls.__name__} in config file for {cls.__name__}")

        def find_subclass_recursive(parent_cls):
            """
            Recursively search for the correct subclass based on the 'type' key.
            """
            for subclass in parent_cls.__subclasses__() + [parent_cls]:
                if type_name in subclass.aliases + [subclass.__name__]:
                    subclass._check_config(config_data, typed=True)
                    instance = subclass.__new__(subclass)
                    for key, value in config_data.items():
                        if not hasattr(instance, key):
                            setattr(instance, key, value)
                            logging.debug(f"key: {key}, value: {value}")
                    setattr(instance, 'global_config', GlobalConfig())
                    instance.__init__( *args, **kwargs)
                    return instance
                if subclass != parent_cls:
                    recursive_result = find_subclass_recursive(subclass)
                    if recursive_result:
                        return recursive_result
            return None

        result = find_subclass_recursive(cls)

        if result is not None:
            return result
        else:
            raise Exception(f"Type {type_name} non trouvé, veuillez vérifier le fichier de configuration. "
                            f"Liste des types disponibles : {[el.__name__ for el in cls.__subclasses__()]}")

    def __repr__(self):
        """
        Return a string representation of the instance.
        """
        return f"{self.__class__.__name__}({self.__dict__})"

    @classmethod
    def _preconditions(cls):
        """
        Check if all preconditions are met before running the algorithm.
        """
        pass

    @staticmethod
    def _safe_open(config_data):
        """
        Open and load configuration data from a YAML file or return the provided dictionary.
        """
        if not isinstance(config_data, (str, dict)):
            raise TypeError("Invalid type for config_data. Expected str (file path) or dict.")

        if isinstance(config_data, str):
            try:
                with open(config_data, 'r') as file:
                    config_data = yaml.safe_load(file)
            except Exception as e:
                raise IOError(f"Error loading config file: {e}")

        if not isinstance(config_data, dict):
            raise TypeError("Invalid type for config_data. Expected dict after loading from YAML.")

        return config_data

    @classmethod
    def _check_config(cls, config_data, typed=False, dynamic_keys=None):
        """
        Check if the configuration data contains all required keys and no invalid keys.
        Args:
            config_data (dict): Configuration data to check.
            typed (bool): Whether the configuration data is typed.
            dynamic_keys (list): List of dynamic keys to add to the required keys.
        """
        required_keys = []
        if typed:
            required_keys = cls.required_keys + ['type']
        current_class = cls

        # Add required keys from all parent classes
        while hasattr(current_class, 'required_keys'):
            required_keys += current_class.required_keys
            current_class = current_class.__base__

        # Add dynamic keys if provided, dynamic keys are keys that are not known in advance
        # useful for configuration files that can have different keys depending on the context or not custom class
        if dynamic_keys is not None:
            required_keys += dynamic_keys

        invalid_keys = set(config_data.keys()) - set(required_keys) - set(cls.__dict__)
        if invalid_keys:
            Warning(f"Supplementary keys in configuration for class {cls.__name__}: {', '.join(invalid_keys)}")

        missing_keys = [key for key in required_keys if key not in config_data]
        if missing_keys:
            raise ValueError(f"Missing required keys for class {cls.__name__}: {', '.join(missing_keys)}")

        cls._preconditions()

    def to_config(self, exclude=[], add={}, parents=True):
        """
        Return a dictionary representation of the instance.
        """
        config = {}
        for key, value in self.__dict__.items():
            if key not in exclude:
                config[key] = value
        config.update(add)
        return config


def load_yaml(yaml_path):
    with open(yaml_path, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data


class StartConfig(Configurable):

    def __init__(self, *args, **kwargs):
        self.resume = kwargs['resume']
        self.debug = kwargs['debug']
        paths = [os.path.join(self.output_dir, self.run_name),
                 os.path.join(self.output_dir, self.run_name, "samples"),
                 ]
        if self.MODE.lower() == "train":
            paths.append(os.path.join(self.output_dir, self.run_name, "WANDB"))
            paths.append(os.path.join(self.output_dir, self.run_name, "WANDB/cache"))
        self._validate_config()
        self._next_run_dir(paths)
        GlobalConfig(self.to_dict())

    def __str__(self):
        def recursive_str(d, indent=0):
            string = ""
            for key, value in d.items():
                if not isinstance(value, GlobalConfig):
                    if isinstance(value, dict):
                        string += f"{' ' * indent}{key}:\n{recursive_str(value, indent + 2)}"
                    else:
                        string += f"{' ' * indent}{key}: {value}\n"
            return string

        config_string = ""
        config_string += recursive_str(self.__dict__)
        return config_string

    def _validate_config(self):
        if self.resume:
            assert hasattr(self, 'snapshot'), f"if you use -r, snapshot must be specified"
            if self.snapshot is None or not os.path.isfile(self.snapshot):
                raise FileNotFoundError(
                    f"self.resume={self.resume} but snapshot_path={self.snapshot} is None or doesn't exist"
                )
            if not (self.run_name in self.snapshot and self.output_dir in self.snapshot):
                raise ValueError(
                    f"self.resume={self.resume} but snapshot_path={self.snapshot} doesn't contain run_name={self.run_name} "
                    f"and output_dir={self.output_dir}"
                )
            if self.MODE.lower() != "train":
                raise ValueError("--resume flag can only be used in Train mode.")
        else:
            assert hasattr(self, 'config'), f"config must be specified"
        return

    def to_dict(self):
        # Convert configuration to a dictionary
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }

    def to_json(self):
        # Convert configuration to a JSON string
        return json.dumps(self.to_dict())

    def to_yaml(self):
        # Convert configuration to a YAML string
        return yaml.dump(self.to_dict())

    def save(self, path):
        # Save configuration to a YAML file
        with open(path, "w+") as f:
            yaml.dump(global_config.to_dict(), f)

    def _next_run_dir(self, paths, suffix=None):
        # Create directories for the next run
        if self.resume:
            if hasattr(self, "snapshot"):
                if not os.path.exists(self.snapshot):
                    raise FileNotFoundError(
                        f"The following directories do not exist: {self.snapshot}"
                    )
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"The following directories do not exist: {path}"
                    )
        else:
            current_datetime = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S%f"
            )[:-3]
            train_num = 1
            train_name = self.run_name
            while os.path.exists(os.path.join(self.output_dir, train_name)):
                if suffix is not None:
                    train_name = (
                            self.basename + "__" + suffix + "_" + current_datetime
                    )
                else:
                    while os.path.exists(os.path.join(self.output_dir, train_name)):
                        if f"_{train_num}" in train_name:
                            train_name = (
                                    "_".join(train_name.split("_")[:-1])
                                    + f"_{train_num + 1}"
                            )
                            train_num += 1
                        else:
                            train_name = f"{train_name}_{train_num}"

            self.run_name = train_name
            paths = [
                os.path.join(self.output_dir, self.run_name),
                os.path.join(self.output_dir, self.run_name, "samples"),
            ]
            if self.MODE.lower() == "train":
                paths.append(os.path.join(self.output_dir, self.run_name, "WANDB"))
                paths.append(os.path.join(self.output_dir, self.run_name,"WANDB","cache"))
            synchronize()
            for path in paths:
                os.makedirs(path, exist_ok=True)