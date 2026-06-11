import copy
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
from datetime import datetime
from functools import total_ordering
from json import load
from logging import StreamHandler
from pathlib import Path
from time import sleep
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Literal,
    Mapping,
    NoReturn,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    TextIO,
    TypeVar,
)

import attr
import attrs
import typing_extensions
from with_argparse import with_dataclass

from ricode.ml.distributed import distributed_rank
from ricode.ml.training_types import AttrsClass
from ricode.utils import format_datetime
from ricode.utils.hashing import mapping_to_hash
from ricode.utils.imports import is_pandas_available
from ricode.utils.json_files import load_json_file_type
from ricode.utils.path import make_path

if is_pandas_available():
    import pandas as pd
else:
    pd = None


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
class ExperimentWatcher(Generic[TExperimentConfig, TExperiment]):
    experiment_dir: Path
    experiment: TExperiment = attrs.field()
    args_fn: Callable[[TExperiment, TExperimentConfig, Path], Sequence[str]]

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
        valid_field_keys = []
        for field_name in field_keys:
            value = getattr(self.experiment, field_name)
            if not isinstance(value, Sequence):
                raise ValueError(
                    f"Field {field_name} must be a Sequence, got {value!r}"
                )
            if len(value) == 0:
                warnings.warn(f"Field {field_name!r} has zero elements, ignoring")
                continue
            valid_field_keys.append(field_name)
            field_values.append(value)

        for args in itertools.product(*field_values):
            config_instance = self.config_type()
            for pos, field_name in enumerate(valid_field_keys):
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

    if is_pandas_available():

        def to_dataframe(self, keys: Sequence[str]) -> pd.DataFrame:
            """
            Return the results of the experiment as a DataFrame object.
            Args:
                keys: A sequence of keys to be included from the results

            Returns: the DataFrame
            """
            include_all_keys = False

            keys = list(keys)
            if len(keys) == 0:
                include_all_keys = True

            for key in attr.fields_dict(type(self.experiment)).keys():
                if getattr(self.experiment, key):
                    keys = ["experiment___" + key] + keys

            data = {}
            for experiment in self.compute_run_info():
                experiment_dir = self.get_experiment_dir(
                    experiment, self.experiment_dir
                )
                experiment_metrics = experiment_dir / "metrics.json"
                if not experiment_metrics.exists():
                    raise ValueError(experiment_metrics)

                with open(experiment_metrics) as f:
                    experiment_json = json.load(f)

                metrics = copy.deepcopy(experiment_json["test_metrics"])
                if include_all_keys:
                    for key in metrics.keys():
                        if key not in keys:
                            keys.append(key)
                for key, value in experiment.items():
                    metrics["experiment___" + key] = value
                for key in keys:
                    if key not in data:
                        data[key] = []
                    if key not in metrics:
                        raise ValueError(key, metrics.keys())
                    data[key].append(metrics[key])
            return pd.DataFrame.from_dict(data)

    else:

        def to_dataframe(self, keys: Sequence[str]) -> NoReturn:
            raise ImportError("pandas is not installed")

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
                continue
            except Exception:
                self.delete_run(info)
                self.logger.info(
                    f"Caught exception when running model {self._prettify_info(info)}"
                )
                self.logger.warning(traceback.format_exc())
                sleep(1)

            try:
                for _ in range(3):
                    self.logger.info(".")
                    sleep(1)
            except KeyboardInterrupt:
                pass

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
            self.delete_run_file(experiment_run_file)
            self.delete_run_file(experiment_done_file)
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
        return [sys.executable] + list(
            self.args_fn(
                self.experiment,
                info,
                self.get_experiment_dir(info, self.experiment_dir),
            )
        )


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


@attrs.define
class HashingExperimentWatcher(ExperimentWatcher[TExperimentConfig, TExperiment]):

    def get_experiment_dir(self, info: TExperimentConfig, base_dir: Path) -> Path:
        def cleanup_value(inp: Any) -> str:
            if not isinstance(inp, str):
                return str(inp)
            return inp

        cleanup_info = {key: cleanup_value(value) for key, value in info.items()}
        experiment_hash = mapping_to_hash(cleanup_info)
        return base_dir / ("ex_" + experiment_hash)


@attrs.define
class HashingExperimentWatcher2(ExperimentWatcher[TExperimentConfig, TExperiment]):
    config_path: str | Path | None = None

    def get_experiment_dir(self, info: OrderedDict[str, Any], base_dir: Path) -> Path:
        experiment_hash = mapping_to_hash(info)
        return base_dir / ("ex_" + experiment_hash)

    def compute_run_info(self) -> Generator[OrderedDict[str, Any], None, None]:
        yield from experiment_config_to_override_configs(self.config_path)


def do_experiments(
    experiment: TExperiment,
    directory: Path | None,
    args_fn: Callable[[TExperiment, TExperimentConfig, Path], Sequence[str]],
    config_type: type[TExperimentConfig] = OrderedDict,
    date: datetime | None = None,
    gpus_per_job: int = 1,
) -> HashingExperimentWatcher[TExperimentConfig, TExperiment]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if directory is None:
        if date is None:
            date = datetime.now()
        directory = make_path("experiments") / format_datetime(date)
    directory.mkdir(parents=True, exist_ok=True)

    log_handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            directory / f"run_experiment_rank{distributed_rank()}.log", "a"
        ),
    ]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=log_handlers,
    )
    logger = logging.getLogger("run_experiment")

    with HashingExperimentWatcher(
        directory,
        experiment,
        args_fn,
        logger,
        config_type,
        gpus_per_job=gpus_per_job,
    ) as watcher:
        logger.info(f"Starting experiment cycle in {directory}")
        watcher.run()
        watcher.summary()

    return watcher


def unstack_mapping(mapping: Any, stack: Sequence[str]):
    if not stack:
        return mapping

    if not isinstance(mapping, Mapping):
        raise ValueError(mapping, stack)

    return unstack_mapping(mapping[stack[0]], stack[1:])


def modifiers_from_mapping(
    content: Mapping[str, Any], stack: Sequence[str] | None = None
):
    if stack is None:
        stack = []

    modifiers = []
    for key, value in content.items():
        if not isinstance(value, (list, dict)):
            continue

        modifier_key = key
        if stack:
            modifier_key = ".".join(stack) + "." + modifier_key
        if isinstance(value, list):
            # easy case, possible values are given as a list
            modifiers.append((modifier_key, value))
        else:  # isinstance(value, dict)
            if "$overrides" in value:
                if not isinstance(value["$overrides"], list):
                    raise ValueError(modifier_key, type(value["$overrides"]))

                if isinstance(value["$overrides"], list):
                    # nested lists of overrides
                    for outer in value["$overrides"]:
                        overrides = []
                        for override in outer:
                            if isinstance(override, str):
                                override_content = load_json_file_type(override)
                                override_content = unstack_mapping(
                                    override_content, stack + [key]
                                )
                                overrides.append(override_content)
                            else:
                                overrides.append(override)
                        modifiers.append((modifier_key, overrides))
                else:
                    overrides = []
                    for override in value["$overrides"]:
                        if isinstance(override, str):
                            override_content = load_json_file_type(override)
                            if stack:
                                override_content = unstack_mapping(
                                    override_content, stack
                                )
                            overrides.append(override_content)
                        else:
                            overrides.append(override)
                    modifiers.append((modifier_key, overrides))

            value_copy = dict(value)
            value_copy.pop("$overrides", None)
            modifiers.extend(modifiers_from_mapping(value_copy, stack + [key]))

    return modifiers


def modifiers_to_override_configs(
    modifiers: Sequence[tuple[str, list[Mapping[str, Any]]]]
) -> Generator[OrderedDict[str, Any], None, None]:
    skeleton = OrderedDict()
    for modifier, _ in modifiers:
        if "." not in modifier:
            skeleton[modifier] = None
        else:
            intermediates = modifier.split(".")
            temp = skeleton
            for pos, intermediate in enumerate(intermediates):
                value = OrderedDict() if pos + 1 != len(intermediates) else None
                if intermediate in temp:
                    if temp[intermediate] is None:
                        temp[intermediate] = temp = value
                    else:
                        temp = temp[intermediate]
                else:
                    temp[intermediate] = temp = value

    for values in itertools.product(*list(map(lambda x: x[1], modifiers))):
        override_config = copy.deepcopy(skeleton)
        keys = list(map(lambda x: x[0], modifiers))
        for modifier, value in zip(keys, values):
            intermediates = modifier.split(".")
            temp = override_config
            for intermediate in intermediates[:-1]:
                temp = temp[intermediate]
            if isinstance(temp[intermediates[-1]], Mapping) and isinstance(
                value, Mapping
            ):
                temp[intermediates[-1]].update(value)
            else:
                temp[intermediates[-1]] = value
        yield override_config


def experiment_config_to_override_configs(config_path: Path):
    content = load_json_file_type(config_path)

    if not isinstance(content, Mapping):
        raise ValueError(config_path, type(content))

    modifiers = modifiers_from_mapping(content, None)
    modifiers.sort(key=lambda x: (x[0].count("."), x[0]))

    yield from modifiers_to_override_configs(modifiers)


def do_experiments_from_config(
    experiment_config: str | Path,
    experiment: TExperiment,
    directory: Path | None,
    args_fn: Callable[[TExperiment, TExperimentConfig, Path], Sequence[str]],
    config_type: type[TExperimentConfig] = OrderedDict,
    date: datetime | None = None,
    gpus_per_job: int = 1,
) -> HashingExperimentWatcher2[TExperimentConfig, TExperiment]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if directory is None:
        if date is None:
            date = datetime.now()
        directory = make_path("experiments") / format_datetime(date)
    directory.mkdir(parents=True, exist_ok=True)

    log_handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            directory / f"run_experiment_rank{distributed_rank()}.log", "a"
        ),
    ]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=log_handlers,
    )
    logger = logging.getLogger("run_experiment")

    with HashingExperimentWatcher2(
        directory,
        experiment,
        args_fn,
        logger,
        OrderedDict(),
        gpus_per_job=gpus_per_job,
        config_path=experiment_config,
    ) as watcher:
        logger.info(f"Starting experiment cycle in {directory}")
        watcher.run()
        watcher.summary()

    return watcher
