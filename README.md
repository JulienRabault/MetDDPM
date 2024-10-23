author : Julien Rabault (julien.rabault@irit.fr)

 # MetDPPM

## Description

MetDPPM is a deep learning project for training and sampling models. It utilizes PyTorch and supports Distributed Data Parallel (DDP) for distributed training across multiple GPUs. The project is designed to be modular, allowing easy addition of new models, datasets, optimizers, and schedulers.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Creating New Classes](#creating-new-classes)
  - [Configurable Classes](#configurable-classes)
  - [Adding New Models](#adding-new-models)
  - [Adding New Datasets](#adding-new-datasets)
  - [Adding New Optimizers](#adding-new-optimizers)
  - [Adding New Schedulers](#adding-new-schedulers)
- [License](#license)

## Installation

To install the dependencies and set up the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/MetDPPM.git
    cd MetDPPM
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure PyTorch is installed with CUDA support if you are using GPUs:

    ```bash
    pip install torch torchvision torchaudio
    ```

## Usage

To use the project, follow these instructions:

1. Prepare a YAML configuration file with the necessary parameters.

2. Run the main script with the appropriate arguments:

    ```bash
    python main.py -y path/to/config.yaml --debug
    ```

    - `-y` or `--yaml_path`: Path to the YAML configuration file.
    - `--debug`: Enable debug logging.
    - `-r` or `--resume`: Resume training from a snapshot.

## File Structure
```
MetDPPM/
│
├── main.py                          # Main script for training and sampling.
│
├── configs/
│   ├── config.py                    # Configuration module for the project.
│   ├── config_sample_a.yml          # Sample configuration file for sampling.
│   └── config_train_a.yml           # Sample configuration file for training.
│
├── core/
│   ├── optim.py                     # Optimizer module.
│   ├── sampler.py                   # Sampler module.
│   ├── scheduler.py                  # Scheduler module.
│   ├── trackers.py                  # Trackers module.
│   └── trainer.py                   # Trainer module.
│
├── datasets/
│   ├── base_dataset.py              # Base dataset module.
│   ├── is_dataset.py                # IS dataset module.
│   ├── multi_option_normalize.py   # Multi-option normalization module.
│   └── normalizer.py               # Normalizer module.
│
├── diffusion_models/
│   ├── custom_model.py              # Custom model module.
│   └── guided_gaussian_diffusion.py # Guided Gaussian diffusion model module.
│
└── utils/
└── distributed.py               # Utilities for distributed training.
```
## Creating New Classes

### Configurable Classes

The `Configurable` class is a base class for configurable objects. It provides methods to load and validate configuration data from a YAML file or a dictionary. Subclasses should define `required_keys` and `aliases` class attributes. When a subclass is initialized with configuration data, the attributes defined in the configuration will be automatically set on the instance.

#### Key Features of `Configurable` Class

- **`required_keys`**: A list of keys that must be present in the configuration data. This ensures that all necessary parameters are provided.
- **`aliases`**: A list of aliases for the class name. This allows for more flexible configuration files.
- **`from_config`**: A class method that creates an instance of the class from configuration data. It validates the configuration data and sets the attributes on the instance.
- **`from_typed_config`**: A class method that creates an instance of a subclass from typed configuration data. It finds the correct subclass based on the `type` key in the configuration data.
- **`_check_config`**: A class method that checks if the configuration data contains all required keys and no invalid keys.
- **`to_config`**: A method that returns a dictionary representation of the instance.

#### Example Usage

To create a new configurable class, you need to inherit from `Configurable` and define the `required_keys` and `aliases` class attributes. Here's a basic example:

```python
from configs.config import Configurable

class MyConfigurableClass(Configurable):
    required_keys = ['param1', 'param2']
    aliases = ['MyClass']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize your class parameters here
```
Adding New Models
To add a new model, create a new class that inherits from BaseModel and implements the required abstract methods.

Example
```python
from core.trainer import BaseModel

class NewModel(BaseModel):
    required_keys = ['new_model_param']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize your model parameters here

    def sample(self, batch_size, return_all_timesteps=False, condition=None):
        # Implement the sampling logic here
        pass

    def forward(self, img, *args, **kwargs):
        # Implement the forward pass logic here
        pass

    def is_guided(self):
        # Implement the logic to check if the model is guided
        pass
```
Example Configuration for NewModel

```json
model:
  type: NewModel
  new_model_param:
    channels: 3
    image_size: [64, 64]
    # other parameters...
```
#### Adding New Datasets
To add a new dataset, create a new class that inherits from BaseDataset and implements the required methods.

#### Adding New Optimizers
To add a new optimizer, create a new class that inherits from OptimizerBase and implements the required methods.

#### Adding New Schedulers
To add a new scheduler, create a new class that inherits from SchedulerBase and implements the required methods.

## License
 See the LICENSE file for details.

This README provides a comprehensive guide on how to set up the project, use it, and extend it by adding new models, datasets, optimizers, and schedulers. The `Configurable` class is explained in detail to help users understand how to create new configurable classes. An example configuration for a new model (`NewModel`) is also provided. If you have any additional files or specific details you'd like to include, please let me know!
