import argparse
import gc
import logging
import os
import sys
import warnings
import yaml

from configs.config import StartConfig
from core.sampler import Sampler
os.environ["WANDB_MODE"] = "offline"

import torch
from torch import distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from core.trainer import Trainer
from utils.distributed import get_rank_num, get_rank, is_main_gpu, synchronize

warnings.filterwarnings(
    "ignore",
    message="This DataLoader will create .* worker processes in total.*",
)
gc.collect()
# Free GPU cache
torch.cuda.empty_cache()


def setup_logger(config, log_file="ddpm.log", use_wandb=False):
    """
    Configure a logger with specified console and file handlers.
    Args:
        config: The configuration object.
        log_file (str): The name of the log file.
    Returns:
        logging.Logger: The configured logger.
    """
    # Use a logger specific to the GPU rank
    console_format = (
        f"[GPU {get_rank_num()}] %(asctime)s - %(levelname)s - %(message)s"
        if torch.cuda.device_count() > 1
        else "%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(f"logddp_{get_rank_num()}")
    logger.setLevel(logging.DEBUG if config.debug else logging.INFO)
    logger.propagate = False  # Prevent double printing

    # Console handler for printing log messages to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if config.debug else logging.INFO)
    console_formatter = logging.Formatter(console_format)
    console_handler.setFormatter(console_formatter)

    # File handler for saving log messages to a file
    file_handler = logging.FileHandler(
        os.path.join(config.output_dir, config.run_name, log_file), mode="w+"
    )
    file_handler.setLevel(logging.DEBUG if config.debug else logging.INFO)
    file_formatter = logging.Formatter(console_format)
    file_handler.setFormatter(file_formatter)

    # Add both handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    if use_wandb:
        logging.getLogger("wandb").setLevel(logging.WARNING)

    return logger


def ddp_setup():
    """
    Configuration for Distributed Data Parallel (DDP).
    """
    if torch.cuda.device_count() < 2:
        return
    # Initialize the process group for DDP
    init_process_group(
        "nccl" if dist.is_nccl_available() else "gloo",
        world_size=torch.cuda.device_count(),
    )
    torch.cuda.set_device(get_rank())


def load_yaml(yaml_path):
    """
    Load YAML data from a file.

    Args:
        yaml_path (str): Path to the YAML file.

    Returns:
        dict: The loaded YAML data.
    """
    with open(yaml_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data


if __name__ == "__main__":

    # Parse command line arguments and load configuration
    parser = argparse.ArgumentParser(
        description="Deep Learning Training and Testing Script"
    )
    parser.add_argument(
        "-y",
        "--yaml_path",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    parser.add_argument("-r", "--resume", action="store_true", help="Resume training")

    ddp_setup()

    # Config.create_arguments(parser)
    args = parser.parse_args()

    start_config = StartConfig.from_config(args.yaml_path , **vars(args))

    assert 'run_name' in start_config.to_dict(), f"run_name must be specified in {args.yaml_path}"
    assert 'output_dir' in start_config.to_dict(), f"output_dir must be specified in {args.yaml_path}"

    local_rank = get_rank()

    synchronize()
    setup_logger(start_config)
    logger = logging.getLogger(f"logddp_{get_rank_num()}")

    if is_main_gpu():
        start_config.save(
            os.path.join(start_config.output_dir, start_config.run_name, "config.yml")
        )
        logger.info(start_config)
        logger.info(f"Mode {start_config.MODE} selected")

    synchronize()
    logger.debug(f"Local_rank: {local_rank}")

    # Execute the main training or sampling function based on the mode
    if start_config.MODE == "train":
        if start_config.resume:
            trainer = Trainer.from_snapshot(start_config.snapshot)
        else:
            trainer = Trainer.from_config(start_config.config)
        trainer.train()
    elif start_config.MODE == "sample":
        sampler = Sampler.from_typed_config(start_config.config)
        sampler.sample()
    else:
        raise ValueError(f"Invalid mode: {start_config.MODE}")

    # Clean up distributed processes if initialized
    if dist.is_initialized():
        destroy_process_group()
