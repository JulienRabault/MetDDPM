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

- `main.py`: Main script for training and sampling.
- `configs/config.py`: Configuration module for the project.
- `configs/config_sample_a.yml`: Sample configuration file for sampling.
- `configs/config_train_a.yml`: Sample configuration file for training.
- `core/optim.py`: Optimizer module.
- `core/sampler.py`: Sampler module.
- `core/scheduler.py`: Scheduler module.
- `core/trackers.py`: Trackers module.
- `core/trainer.py`: Trainer module.
- `datasets/base_dataset.py`: Base dataset module.
- `datasets/is_dataset.py`: IS dataset module.
- `datasets/multi_option_normalize.py`: Multi-option normalization module.
- `datasets/normalizer.py`: Normalizer module.
- `diffusion_models/custom_model.py`: Custom model module.
- `diffusion_models/guided_gaussian_diffusion.py`: Guided Gaussian diffusion model module.
- `utils/distributed.py`: Utilities for distributed training.
