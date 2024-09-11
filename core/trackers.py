import csv
from abc import ABC, abstractmethod
import os
import time
import wandb
import mlflow
from pathlib import Path
from configs.config import Configurable


class BaseTracker(ABC, Configurable):
    def __init__(self, *args, output_run='.', **kwargs):
        self.output_run = output_run

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def log(self, epoch, log_dict):
        pass

    def close(self):
        pass

class Wandb(BaseTracker):

    def init(self, config=None):
        t = time.strftime("%d-%m-%y_%H-%M", time.localtime(time.time()))
        self.logger.debug("WANDB initialized")
        os.environ["WANDB_CACHE_DIR"] = os.path.join(
            self.output_run, "WANDB, cache"
        )
        os.environ["WANDB_DIR"] = os.path.join(
            self.output_run, "WANDB"
        )
        self.run = wandb.init(
            project=self.config.wandb_config.project,
            resume="auto" if self.config.wandb_config.resume else None,
            mode=os.environ.get("WANDB_MODE", "online"),
            entity=self.config.wandb_config.entity,
            name=f"{self.config.run_name}_{t}/",
            config=self.config.to_dict(),
        )

    def log(self, epoch, log_dict):
        wandb.log(log_dict, step=epoch)

    def close(self):
        wandb.finish()


class Mlflow(BaseTracker):

    def init(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        experiment_name = self.experiment_name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        self.run = mlflow.start_run(nested=True, run_name=self.global_config.run_name)
        mlflow.log_params(self.global_config.to_dict())

    def log(self, epoch, log_dict):
        mlflow.log_metrics(log_dict, step=epoch)

    def close(self):
        mlflow.end_run()


class CsvLogger():
    def __init__(self, output_run):
        self.csv_filename = os.path.join(
            output_run, "logs_train.csv"
        )

    def log(self, epoch, log_dict):
        file_exists = Path(self.csv_filename).is_file()
        with open(self.csv_filename, "a" if file_exists else "w", newline="") as csvfile:
            fieldnames = ["epoch"] + list(log_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({**{"epoch": epoch}, **log_dict})


class Trackers:
    def __init__(self, loggers_configs, output_run):
        self.loggers = []
        self.loggers_configs = loggers_configs
        self.output_run = output_run
        self.is_init = False

    def init(self):
        for logger in self.loggers_configs:
            logger_cls = BaseTracker.from_typed_config(logger, output_run=self.output_run)
            logger_cls.init()
            self.loggers.append(logger_cls)
        self.loggers.append(CsvLogger(self.output_run))
        self.is_init = True

    def log(self, epoch, log_dict):
        if not self.is_init:
            raise ValueError("MultiLogger is not initialized")
        for logger in self.loggers:
            logger.log(epoch, log_dict)

    def finish(self):
        if not self.is_init:
            raise ValueError("MultiLogger is not initialized")
        for logger in self.loggers:
            logger.finish()
        self.is_init = False
