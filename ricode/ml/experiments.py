#!/usr/bin/env python3
import dataclasses
import enum
import itertools
import logging
import os
import subprocess
import sys
import time
import traceback
import typing
import warnings
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from functools import total_ordering
from json import dumps, load
from logging import StreamHandler
from pathlib import Path
from time import sleep
from typing import Generator, Literal, Optional, Sequence, TextIO

import torch

import typing_extensions
from torch import Tensor

from with_argparse import with_dataclass


@dataclass
class ExperimentArgs:
    models: list[str]
    datasets: list[str]
    n_splits: int
    experiment_dir: Path
    architectures: list[Literal["iter", "asp", "diffusionner"]]
    dev: bool = False
    use_bfloat16: bool = False
    use_fsdp: bool = False
    load_ckpt: Optional[Path] = None
    split_as_dataset: bool = False
    summary_only: bool = False
    optimize_for: str = "f1"
    work_dir: Optional[Path] = None
    cleanup_status_dir: bool = False


@with_dataclass(dataclass=ExperimentArgs, allow_glob={"datasets"})
def run_experiment(
    args: ExperimentArgs, watcher_class: "Optional[type[ExperimentWatcher]]" = None
):
    args.work_dir = args.work_dir or Path.cwd()
    args.architectures = args.architectures or ["iter"]
    watcher_class = watcher_class or ExperimentWatcher
    log_handlers: Sequence[StreamHandler[TextIO]] = [
        main_log := logging.StreamHandler(sys.stdout),
    ]
    if not args.summary_only:
        log_file = args.experiment_dir / (
            ("run_experiment" if not args.dev else "dev_run_experiment") + ".log"
        )
        if "CUDA_VISIBLE_DEVICES" in os.environ or "RANK" in os.environ:
            log_file = log_file.with_name(
                log_file.name
                + ".cuda"
                + os.environ.get("RANK", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
            )
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handlers = [
            log_handlers[0],
            typing.cast(StreamHandler[TextIO], logging.FileHandler(log_file, mode="a")),
        ]
    main_log = log_handlers[-1]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=log_handlers,
    )
    logger = logging.getLogger("run_experiment")

    with watcher_class(
        args.experiment_dir,
        args.models,
        args.datasets,
        list(range(args.n_splits)),
        args.architectures,
        1,
        0,
        logger,
        main_log_file=main_log,
        use_bfloat16=args.use_bfloat16,
        use_fsdp=args.use_fsdp,
        use_seed_as_dataset=args.split_as_dataset,
        load_ckpt=args.load_ckpt,
        optimize_for=args.optimize_for,
        summary_only=args.summary_only,
        work_dir=args.work_dir,
        cleanup_status_dir=args.cleanup_status_dir,
    ) as watcher:
        logger.info(f"Waiting 5 seconds to start (dev mode = {args.dev})")
        time.sleep(5)

        # env variables
        if args.dev:
            os.environ["WANDB_MODE"] = "disabled"
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        watcher.run()
        watcher.summary()


@total_ordering
class ExperimentStatus(enum.Enum):
    STATUS_NOT_STARTED = 0
    STATUS_ALREADY_STARTED = 1
    STATUS_NEEDS_MORE_GPU = 2
    STATUS_DONE = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


@dataclasses.dataclass(eq=True, frozen=True)
class ExperimentInfo:
    model: str
    dataset: str
    seed: int
    architecture: str = "iter"

    def __repr__(self):
        return (
            f"ExperimentInfo(\n"
            f"\tmodel={self.model}\n"
            f"\tdataset={self.dataset}\n"
            f"\tarchitecture={self.architecture}\n"
            f"\tseed={self.seed}\n"
            f")"
        )


def cleanup_datasets(datasets: list[str]):
    return list(
        sorted(
            {
                (dataset if "_split" not in dataset else dataset.split("_split")[0])
                for dataset in datasets
            }
        )
    )


class ExperimentWatcher:
    experiment_dir: Path
    failed_experiments: set[ExperimentInfo]

    def __init__(
        self,
        experiment_dir: Path,
        models: list[str],
        datasets: list[str],
        seeds: list[int],
        architectures: Sequence[str],
        gpus_per_job: int,
        seed_offset: int,
        logger: logging.Logger,
        main_log_file: StreamHandler[TextIO],
        use_bfloat16: bool,
        use_fsdp: bool,
        use_seed_as_dataset: bool,
        load_ckpt: Optional[Path],
        summary_only: bool,
        optimize_for: str,  # f1
        work_dir: Path,
        cleanup_status_dir: bool,
    ):
        self.experiment_dir = experiment_dir
        self.gpus_per_job = gpus_per_job
        self.original_seeds = seeds
        self.seed_offset = seed_offset

        self.models = models
        self.datasets = cleanup_datasets(datasets)
        self.architectures = architectures

        self.failed_experiments = set()
        self.stop_iteration = False
        self.summary_only = summary_only

        self.logger = logger
        self.main_log_file = main_log_file

        self.use_bfloat16 = use_bfloat16
        self.use_fsdp = use_fsdp
        self.use_seed_as_dataset = use_seed_as_dataset
        self.load_ckpt = load_ckpt
        self.cleanup_status_dir = cleanup_status_dir

        self.optimize_for = optimize_for

        self.work_dir = work_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.logging_dir.mkdir(parents=True, exist_ok=True)

    @property
    def status_dir(self) -> Path:
        return self.experiment_dir / ".status"

    @property
    def logging_dir(self) -> Path:
        return self.experiment_dir / ".logs"

    @property
    def seeds(self) -> list[int]:
        return list(map(lambda x: x + self.seed_offset, self.original_seeds))

    def compute_run_info(self) -> Generator[ExperimentInfo, None, None]:
        for model, dataset, seed, architecture in itertools.product(
            self.models, self.datasets, self.seeds, self.architectures
        ):
            yield ExperimentInfo(model, dataset, seed, architecture)

    @staticmethod
    def get_experiment_dir(info: ExperimentInfo, base_dir: Path):
        if (base_dir / info.model).exists():
            return base_dir / info.model  # legacy support

        def cleanup_name(inp: str) -> str:
            if "_split" in inp:
                inp = inp.split("_split")[0]
            return inp.replace("/", "_")

        return (
            base_dir
            / info.architecture
            / cleanup_name(info.dataset)
            / cleanup_name(info.model)
        )

    def get_runtime_dir(self, info: ExperimentInfo):
        return self.get_experiment_dir(info, self.experiment_dir)

    def get_experiment_state(self, info: ExperimentInfo):
        experiment_run_file = self.get_experiment_dir(
            info, self.status_dir
        ) / self.get_run_name(info, False)
        experiment_done_file = self.get_experiment_dir(
            info, self.status_dir
        ) / self.get_run_name(info, True)

        if info in self.failed_experiments:
            # if we failed before, don't try again
            return ExperimentStatus.STATUS_DONE

        if experiment_done_file.exists():
            return ExperimentStatus.STATUS_DONE
        elif experiment_run_file.exists():
            return ExperimentStatus.STATUS_ALREADY_STARTED
        else:
            return ExperimentStatus.STATUS_NOT_STARTED

    def missing_runs(
        self: "ExperimentWatcher",
    ) -> Generator[ExperimentInfo, None, None]:
        for experiment in self.compute_run_info():
            experiment_state = self.get_experiment_state(experiment)
            if experiment_state == ExperimentStatus.STATUS_NOT_STARTED:
                yield experiment

    def summary(self):
        metrics = defaultdict(lambda: defaultdict(list))
        bound_metrics = defaultdict(lambda: defaultdict(list))
        num_runs = len(self.seeds)

        skip_model_dataset_arch = set()
        for experiment in self.compute_run_info():
            experiment_dir = self.get_experiment_dir(
                experiment, self.experiment_dir
            ) / self.get_run_name(experiment)
            experiment_metrics = experiment_dir / "metrics.json"
            if not experiment_metrics.exists():
                skip_model_dataset_arch.add(
                    (experiment.architecture, experiment.model, experiment.dataset)
                )
                continue

            experiment_key = (
                experiment.dataset,
                experiment.architecture,
                experiment.model,
            )
            with experiment_metrics.open() as f:
                json_blob = load(f)

                for k, v in json_blob["test_metrics"].items():
                    metrics[experiment_key][k].append(v)
                if "test_metrics_bounds" in json_blob:
                    for k, v in json_blob["test_metrics_bounds"].items():
                        bound_metrics[experiment_key][k].append(v)

        def compute_std_mean_best(
            vals: Tensor, best_run: int
        ) -> tuple[float, float, float]:
            vals = vals.squeeze()
            if not vals.is_floating_point():
                vals = vals.to(torch.float64)
            if vals.numel() == 0:
                return 0.0, 0.0, 0.0
            if vals.dim() == 0:  # single element received as input
                return 0.0, vals.item(), vals.item()
            std_mean = torch.std_mean(vals)
            return std_mean[0].item(), std_mean[1].item(), vals[best_run].item()

        for (dataset, architecture, model), experiment_metrics in metrics.items():
            if any(len(v) < num_runs for k, v in experiment_metrics.items()):
                self.logger.info(
                    f"Skipping model '{model}' because metrics files are not available"
                )
                continue
            experiment_metrics = OrderedDict(
                {k: torch.tensor(v) for k, v in experiment_metrics.items()}
            )
            experiment_bound_metrics = bound_metrics[(dataset, architecture, model)]
            experiment_bound_metrics = OrderedDict(
                {k: torch.tensor(v) for k, v in experiment_bound_metrics.items()}
            )

            best_performing_run = experiment_metrics[self.optimize_for].argmax().item()

            result_metrics = OrderedDict()
            for k, v in experiment_metrics.items():
                v = v.squeeze()
                if not v.dtype.is_floating_point:
                    v = v.to(torch.float)
                bound = (
                    0.0,
                    0.0,
                    0.0,
                )
                if k in experiment_bound_metrics:
                    bound = compute_std_mean_best(
                        experiment_bound_metrics[k], best_performing_run
                    )
                strict = compute_std_mean_best(v, best_performing_run)

                result_metrics[k] = strict + bound

            for k, v in result_metrics.copy().items():
                std_dev, mean, maximum, bound_std, bound_mean, bound_maximum = v
                del result_metrics[k]
                result_metrics[k.lower()] = (
                    f"{mean:.6f} ± {std_dev:.4f} / {bound_mean:.6f} ± {bound_std:.4f} "
                    f"(best is {maximum:.6f} / {bound_maximum:.6f})"
                )

            self.logger.info(
                f"Model '{model}' results with {len(self.seeds)} runs (best w.r.t {self.optimize_for} "
                f"was {self.get_run_name_by_id(best_performing_run)} out of "
                f"{', '.join(f'{v:.6f}' for v in experiment_metrics[self.optimize_for].tolist())})"
            )
            self.logger.info(dumps(result_metrics, indent=2, ensure_ascii=False))

    def run(self):
        if self.summary_only:
            return

        experiment_lock = self.experiment_dir / ".lock"
        if not self.is_rank_zero():
            while not experiment_lock.exists():
                self.logger.info("Waiting for main process startup")
        else:
            experiment_lock.touch(exist_ok=True)

        while (
            info := next(self.missing_runs(), None)
        ) is not None and not self.stop_iteration:
            successful = False
            try:
                self.logger.info(f"Starting experiment {info}")
                successful = self.run_experiment(info)
            except KeyboardInterrupt:
                self.delete_run(info)
                self.stop_iteration = True
                print("Received KeyboardInterrupt, aborting ...")
                sleep(1)
            except Exception:
                self.delete_run(info)
                self.logger.info(f"Caught exception when running model {info}")
                self.logger.warning(traceback.format_exc())
                sleep(1)

            for _ in range(3):
                self.logger.info(".")
                sleep(1)

            if not successful:
                self.delete_run(info)
        if info is None:
            self.logger.info(
                f"Completed {len(list(self.compute_run_info()))} experiments"
            )

    def __enter__(self) -> typing_extensions.Self:
        removed = 0
        for info in self.compute_run_info():
            state = self.get_experiment_state(info)
            if state == ExperimentStatus.STATUS_ALREADY_STARTED:
                if self.cleanup_status_dir:
                    if (++removed) == 0:
                        self.logger.info("Performing cleanup ...")
                    run_file = self.get_experiment_dir(
                        info, self.status_dir
                    ) / self.get_run_name(info, False)
                    self.logger.info("\t" + run_file.as_posix())
                    os.remove(run_file)
                else:
                    if (++removed) == 0:
                        self.logger.info("Currently running jobs:")
                    self.logger.info(info)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        experiment_lock = self.experiment_dir / ".lock"
        if self.is_rank_zero() and experiment_lock.exists():
            os.remove(experiment_lock)

    @staticmethod
    def rank():
        return int(os.environ.get("RANK", "0"))

    @staticmethod
    def world_size():
        return int(os.environ.get("WORLD_SIZE", "1"))

    @classmethod
    def is_rank_zero(cls):
        return cls.rank() == 0

    @classmethod
    def is_local_rank_zero(cls):
        return (cls.rank() % cls.world_size()) == 0

    def run_experiment(self, info: ExperimentInfo) -> bool:
        experiment_name = f"{info.seed}"
        experiment_status_dir = self.get_experiment_dir(info, self.status_dir)
        experiment_status_dir.mkdir(parents=True, exist_ok=True)
        experiment_run_file = experiment_status_dir / self.get_run_name(info, False)
        experiment_done_file = experiment_status_dir / self.get_run_name(info, True)

        if experiment_done_file.exists():
            warnings.warn("Attempted to run completed experiment ...")
            return True

        experiment_log_dir = self.get_experiment_dir(info, self.logging_dir)
        experiment_log_dir.mkdir(parents=True, exist_ok=True)
        experiment_log_file = experiment_log_dir / (experiment_name + ".log")
        if not self.is_rank_zero():
            while not experiment_run_file.exists():
                self.logger.info(
                    "Waiting for rank zero to start the training process ..."
                )
                sleep(10)

            experiment_log_file = experiment_run_file.with_name(
                experiment_log_file.name
                + "_rank"
                + os.environ.get("RANK", "0")
                + ".log"
            )
        else:
            experiment_run_file.touch(exist_ok=True)
            self.logger.info(
                f"Created run status file {experiment_run_file.as_posix()}"
            )

        experiment_dir = self.get_experiment_dir(
            info, self.experiment_dir
        ) / self.get_run_name(info)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        args = self.get_run_args(info)
        self.logger.info(f"Run args = {' '.join(args)}")
        try:
            with open(experiment_log_file, "w") as log_file:
                output = run_subprocess_into_file(
                    args,
                    logfiles=[log_file, self.main_log_file.stream],
                    stderr=subprocess.STDOUT,
                    cwd=experiment_dir,
                )
                if output.returncode == 0 and self.is_rank_zero():
                    experiment_done_file.touch(exist_ok=True)
                elif output.returncode == 0:
                    while not experiment_done_file.exists():
                        self.logger.info("Waiting for main process finish")
                        sleep(10)
                else:
                    self.delete_run_file(experiment_run_file)
                    self.failed_experiments.add(info)
                self.logger.info(
                    f"Process {info.dataset}/{info.architecture}/{info.model}/{self.get_run_name(info)} produced return code {output.returncode}"
                )
                return output.returncode == 0
        except KeyboardInterrupt:
            self.logger.info("Received KeyboardInterrupt, aborting training process")
            self.delete_run_file(experiment_run_file)
            self.delete_run_file(experiment_done_file)
            sleep(3)
            raise

    def delete_run(self, info: ExperimentInfo):
        experiment_run_file = self.get_experiment_dir(
            info, self.status_dir
        ) / self.get_run_name(info, False)
        experiment_done_file = self.get_experiment_dir(
            info, self.status_dir
        ) / self.get_run_name(info, True)

        self.delete_run_file(experiment_run_file)
        self.delete_run_file(experiment_done_file)

    def delete_run_file(self, filepath: Path):
        if filepath.exists():
            self.logger.info(f"Deleting {filepath.as_posix()}")
            try:
                os.remove(filepath)
            except FileNotFoundError:
                self.logger.warning(f"Could not remove file {filepath}")

    def get_iter_run_args(self, info: ExperimentInfo):
        args = [
            "python3",
            (self.work_dir / "train.py").as_posix(),
            "--transformer",
            info.model,
            "--config",
            (
                info.dataset
                if not self.use_seed_as_dataset
                else info.dataset + "_split" + str(info.seed)
            ),
            "--no_tqdm",
            "--verbose",
        ]
        if not self.use_bfloat16:
            args.append("--use_float32")
        if self.use_fsdp:
            args.append("--use_fsdp")
        args.append("--seed")
        args.append(str(info.seed))
        if self.load_ckpt:
            args.extend(["--load_checkpoint", self.load_ckpt.as_posix()])
        args.append("--model_path")
        args.append(
            (self.get_runtime_dir(info) / self.get_run_name(info, False)).as_posix()
        )
        return args

    @staticmethod
    def get_run_name(info: ExperimentInfo, completed: bool = False):
        return f"{info.seed:04d}" + ("_done" if completed else "")

    @staticmethod
    def get_run_name_by_id(run_id: int):
        return f"{run_id:04d}"

    def get_run_args(self, info: ExperimentInfo):
        if info.architecture == "iter":
            return self.get_iter_run_args(info)
        elif hasattr(self, method_name := "get_" + info.architecture + "_run_args"):
            return getattr(self, method_name)(info)
        else:
            raise NotImplementedError(info.architecture)


def run_subprocess_into_file(
    *popenargs, logfiles: list[TextIO], capture_output=False, check=False, **kwargs
):
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError(
                "stdout and stderr arguments may not be used " "with capture_output."
            )
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if "stdout" not in kwargs:
        kwargs["stdout"] = subprocess.PIPE
    if "stderr" not in kwargs:
        kwargs["stderr"] = subprocess.PIPE

    process: subprocess.Popen[bytes]
    with subprocess.Popen(*popenargs, **kwargs) as process:
        try:
            proc_stdout = process.stdout
            if proc_stdout is None:
                raise ValueError("Cannot run subprocess without stdout")
            moin = True
            leftovers: Optional[bytearray] = bytearray()
            while moin:
                while proc_stdout.readable() > 0:
                    stdout_size = min(256, proc_stdout.readable())
                    if stdout_size > 0:
                        line: bytes = proc_stdout.read(stdout_size)
                        if not line:
                            moin = False
                            break
                        if leftovers is not None and len(leftovers) > 0:
                            leftovers.extend(line)
                            line = bytes(line)
                            leftovers = None
                        try:
                            decoded_line = line.decode("utf-8")
                            sys.stdout.write(decoded_line)
                            for logfile in logfiles:
                                logfile.write(decoded_line)
                                logfile.flush()
                        except UnicodeDecodeError:
                            if leftovers is None:
                                leftovers = bytearray()
                            leftovers.extend(line)
                # else:
            #                time.sleep(0.25)
            process.wait()
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise
        except Exception:  # Including KeyboardInterrupt, communicate handled that.
            process.kill()
            # We don't call process.wait() as .__exit__ does that for us.
            raise
        retcode = process.poll()
        if check and retcode:
            raise subprocess.CalledProcessError(
                retcode, process.args, output=None, stderr=None
            )
        elif retcode is None:
            retcode = 0
    return subprocess.CompletedProcess(process.args, retcode, None, None)
