#!/usr/bin/env python3
import dataclasses
import enum
import itertools
import json
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
from json import load
from logging import StreamHandler
from pathlib import Path
from time import sleep
from typing import (
    Any,
    Generator,
    Generic,
    Literal,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    TextIO,
    TypeVar,
)

import attrs
import typing_extensions
from with_argparse import with_dataclass

from ricode.ml.training_types import AttrsClass


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


@runtime_checkable
class ExperimentProtocol(Protocol):
    pass


TExperimentConfig = TypeVar("TExperimentConfig", bound=OrderedDict[str, Any])
TExperiment = TypeVar("TExperiment", bound=AttrsClass)


@attrs.define
class ExperimentWatcher(Generic[TExperimentConfig]):
    experiment_dir: Path
    experiment: TExperiment = attrs.field()

    @experiment.validator
    def _experiment_validator(self, attrib, value):
        if not attrs.has(type(value)):
            raise ValueError(
                f"{attrib.name} must ba a attrs-defined type, got {value!r}"
            )

    logger: logging.Logger
    config_type: type[TExperimentConfig] = attrs.field(default=OrderedDict)
    gpus_per_job: int = attrs.field(default=1)
    optimize_for: Optional[bool] = attrs.field(default=None)
    cleanup_status_dir: bool = attrs.field(default=False)

    # non-init fields
    failed_experiments: list[TExperimentConfig] = attrs.field(init=False, factory=list)
    stop_iteration: bool = attrs.field(default=False, init=False)

    @property
    def status_dir(self) -> Path:
        return self.experiment_dir / ".status"

    @property
    def logging_dir(self) -> Path:
        return self.experiment_dir / ".logs"

    def compute_run_info(self) -> Generator[TExperimentConfig, None, None]:
        field_keys = list(attrs.fields_dict(type(self.experiment)).keys())
        field_values = []
        for field_name in field_keys:
            value = getattr(self.experiment, field_name)
            if not isinstance(value, Sequence):
                raise ValueError(
                    f"Field {field_name} must be a Sequence, got {value!r}"
                )
            field_values.append(value)

        for args in itertools.product(*field_values):
            config_instance = self.config_type()
            for pos, field_name in enumerate(field_keys):
                config_instance[field_name] = args[pos]
            yield config_instance

    def get_experiment_dir(self, info: TExperimentConfig, base_dir: Path):
        def cleanup_name(inp: str) -> str:
            return inp.replace("/", "_")

        base_path = base_dir
        for key, value in info.items():
            base_path = base_path / f"{cleanup_name(key)}__{cleanup_name(value)}"

        return base_path

    def get_runtime_dir(self, info: TExperimentConfig):
        return self.get_experiment_dir(info, self.experiment_dir)

    def get_experiment_state(self, info: TExperimentConfig):
        status_dir = self.get_experiment_dir(info, self.status_dir)
        experiment_run_file = status_dir / self.get_status_filename(info, False)
        experiment_done_file = status_dir / self.get_status_filename(info, True)

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
    ) -> Generator[OrderedDict[str, Any], None, None]:
        for experiment in self.compute_run_info():
            experiment_state = self.get_experiment_state(experiment)
            if experiment_state == ExperimentStatus.STATUS_NOT_STARTED:
                yield experiment

    # @deprecated
    def summary(self):
        metrics = defaultdict(lambda: defaultdict(list))

        skip_model_dataset_arch = set()
        for experiment in self.compute_run_info():
            experiment_dir = self.get_experiment_dir(experiment, self.experiment_dir)
            experiment_metrics = experiment_dir / "metrics.json"
            if not experiment_metrics.exists():
                self.logger.warning(
                    f"Experiment {self._prettify_info(experiment, slim=True)} did not produce a metrics.json"
                )
                continue

            with experiment_metrics.open() as f:
                json_blob = load(f)

                value = (
                    json_blob["test_metrics"][self.optimize_for]
                    if self.optimize_for
                    else json_blob["test_metrics"]
                )

                if self.optimize_for:
                    self.logger.info(
                        f"{self._prettify_info(experiment, slim=True)}: {value:.8f}"
                    )
                else:
                    self.logger.info(
                        f"{self._prettify_info(experiment, slim=True)}: {value!r}"
                    )

    @classmethod
    def _prettify_info(cls, info: Mapping[str, Any], slim=False):
        return json.dumps(info, indent=None if slim else 2)

    def run(self):
        experiment_lock = self.experiment_dir / ".lock"
        if not self.is_rank_zero() and self.gpus_per_job > 1:
            while not experiment_lock.exists():
                self.logger.info("Waiting for main process startup")
        else:
            experiment_lock.touch(exist_ok=True)

        while (
            info := next(self.missing_runs(), None)
        ) is not None and not self.stop_iteration:
            successful = False
            try:
                self.logger.info(f"Starting experiment {self._prettify_info(info)}")
                successful = self.run_experiment(info)
            except KeyboardInterrupt:
                self.delete_run(info)
                self.stop_iteration = True
                print("Received KeyboardInterrupt, aborting ...")
                sleep(1)
            except Exception:
                self.delete_run(info)
                self.logger.info(
                    f"Caught exception when running model {self._prettify_info(info)}"
                )
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
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.logging_dir.mkdir(parents=True, exist_ok=True)

        removed = 0
        for info in self.compute_run_info():
            state = self.get_experiment_state(info)
            if state == ExperimentStatus.STATUS_ALREADY_STARTED:
                if self.cleanup_status_dir:
                    if (++removed) == 0:
                        self.logger.info("Performing cleanup ...")
                    run_file = self.get_experiment_dir(
                        info, self.status_dir
                    ) / self.get_status_filename(info, False)
                    self.logger.info("\t" + run_file.as_posix())
                    os.remove(run_file)
                else:
                    if (++removed) == 0:
                        self.logger.info("Currently running jobs:")
                    self.logger.info(info)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        experiment_lock = self.experiment_dir / ".lock"
        if (self.is_rank_zero() and self.gpus_per_job > 1) and experiment_lock.exists():
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

    def run_experiment(self, info: TExperimentConfig) -> bool:
        status_dir = self.get_experiment_dir(info, self.status_dir)
        status_dir.mkdir(parents=True, exist_ok=True)
        experiment_run_file = status_dir / self.get_status_filename(info, False)
        experiment_done_file = status_dir / self.get_status_filename(info, True)

        if experiment_done_file.exists():
            warnings.warn("Attempted to run completed experiment ...")
            return True

        experiment_log_dir = self.get_experiment_dir(info, self.logging_dir)
        experiment_log_dir.mkdir(parents=True, exist_ok=True)
        experiment_log_file = experiment_log_dir / "experiment.log"
        if not self.is_rank_zero() and self.gpus_per_job > 1:
            while not experiment_run_file.exists():
                self.logger.info(
                    "Waiting for rank zero to start the training process ..."
                )
                sleep(10)

            experiment_log_file = experiment_run_file.with_name(
                experiment_log_file.name + "_rank" + str(self.rank()) + ".log"
            )
        else:
            experiment_run_file.touch(exist_ok=True)
            self.logger.info(
                f"Created run status file {experiment_run_file.as_posix()}"
            )

        experiment_dir = self.get_experiment_dir(info, self.experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        args = self.get_run_args(info)
        self.logger.info(f"Run args = {' '.join(args)}")
        try:
            with open(experiment_log_file, "w") as log_file:
                output = run_subprocess_into_file(
                    args,
                    logfiles=[
                        log_file,
                        *[
                            getattr(handler, "stream")
                            for handler in self.logger.handlers
                            if hasattr(handler, "stream")
                        ],
                    ],
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                )
                if output.returncode == 0 and (
                    self.is_rank_zero() or self.gpus_per_job == 1
                ):
                    experiment_done_file.touch(exist_ok=True)
                elif output.returncode == 0:
                    while not experiment_done_file.exists():
                        self.logger.info("Waiting for main process finish")
                        sleep(10)
                else:
                    self.delete_run_file(experiment_run_file)
                    self.failed_experiments.append(info)
                self.logger.info(
                    f"Process {info!r} produced return code {output.returncode}"
                )
                return output.returncode == 0
        except KeyboardInterrupt:
            self.logger.info("Received KeyboardInterrupt, aborting training process")
            self.delete_run_file(experiment_run_file)
            self.delete_run_file(experiment_done_file)
            sleep(3)
            raise

    def delete_run(self, info: TExperimentConfig):
        experiment_run_file = self.get_experiment_dir(
            info, self.status_dir
        ) / self.get_status_filename(info, False)
        experiment_done_file = self.get_experiment_dir(
            info, self.status_dir
        ) / self.get_status_filename(info, True)

        self.delete_run_file(experiment_run_file)
        self.delete_run_file(experiment_done_file)

    def delete_run_file(self, filepath: Path):
        if filepath.exists():
            self.logger.info(f"Deleting {filepath.as_posix()}")
            try:
                os.remove(filepath)
            except FileNotFoundError:
                self.logger.warning(f"Could not remove file {filepath}")

    @staticmethod
    def get_status_filename(info: TExperimentConfig, completed: bool = False):
        if completed:
            return "completed"
        return "running"

    def get_run_args(self, info: TExperimentConfig):
        raise NotImplementedError(info)


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
