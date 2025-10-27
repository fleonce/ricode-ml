import collections
import contextlib
import functools
import gc
import json
import logging
import math
import operator
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
import warnings
from collections import OrderedDict
from contextlib import AbstractContextManager
from datetime import datetime
from functools import lru_cache
from logging import Logger
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Mapping,
    MutableMapping,
    Optional,
    TypeVar,
)

import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict
import torch.version
from more_itertools import first
from torch import Generator
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import PreTrainedModel

from ricode.ml import training_operators
from ricode.ml.dataloaders import Batch, ProfilingDataLoaderIterator
from ricode.ml.distributed import distributed_barrier
from ricode.ml.distributed.utils import finalise_distributed_environment_after_exit
from ricode.ml.model_setup.fully_sharded_dp import _is_fully_sharded_dp_model
from ricode.ml.model_setup.utils import setup_model
from ricode.ml.training_basics import (
    BasicHparams,
    BasicMetrics,
    MetricsDict,
    TensorboardLogger,
)
from ricode.ml.training_operators import safe_score_comparison
from ricode.ml.training_types import (
    ConfigInitProtocol,
    DataLoaderProtocol,
    EpochBasedTraining,
    EvaluateProtocol,
    ForwardBackwardProtocol,
    HasDatasetProperties,
    Hooks,
    ModelInitProtocol,
    OptimizerInitProtocol,
    StepBasedTraining,
    TrainingArgs,
)
from ricode.ml.training_utils import (
    get_commit_hash,
    get_working_tree_diff,
    is_clean_working_tree,
    move_to_device,
)

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


TConfig = TypeVar("TConfig")
TModel = TypeVar("TModel", bound=PreTrainedModel)
TDataset = TypeVar("TDataset", bound=HasDatasetProperties)
THparams = TypeVar("THparams", bound=BasicHparams)
TMetrics = TypeVar("TMetrics", bound=BasicMetrics)


def setup_seed(seed: int, strict: bool = True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(min(seed, 2**32 - 1))
    if strict:
        setup_reproducible()
    return torch.Generator().manual_seed(seed)


def setup_reproducible():
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)


_checkpoint_state_dict = OrderedDict()


def save_checkpoint(
    args: TrainingArgs[THparams, TDataset],
    model: TModel,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    model_path: str,
    disable: bool = False,
    memory_checkpoints: bool = False,
):
    if not disable:
        model_dir = Path(model_path)
        save_pretrained_kwargs = {}
        if args.use_fsdp:
            FSDP.set_state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
            )
            state_dict = model.state_dict()
            optimizer_state_dict = FSDP.optim_state_dict(model, optimizer)
            if args.rank == 0:
                save_pretrained_kwargs["state_dict"] = state_dict
                save_pretrained_kwargs["is_main_process"] = True
        else:
            optimizer_state_dict = optimizer.state_dict()
        if memory_checkpoints:
            if "state_dict" not in save_pretrained_kwargs:
                save_pretrained_kwargs["state_dict"] = model.state_dict()
            # global _checkpoint_state_dict

            _checkpoint_state_dict["state_dict"] = move_to_device(
                save_pretrained_kwargs["state_dict"], "cpu"
            )
            if args.use_fsdp:
                dist.barrier()
            return

        if not args.use_fsdp or args.rank == 0:
            model.save_pretrained(model_path, **save_pretrained_kwargs)

        optimizer_state = {
            "step": args.train_steps,
            "epoch": args.epoch,
            "best_score": args.best_score,
            "optimizer": optimizer_state_dict,
            "lr_scheduler": lr_scheduler.state_dict(),
            "generator": args.generator.get_state(),
        }

        if args.rank == 0:
            torch.save(optimizer_state, model_dir / "optimizer.bin")
        if args.use_fsdp:
            dist.barrier()


def _get_model_path(basename: str = "models", date: Optional[datetime] = None):
    if date is None:
        date = datetime.now()
    date_fmt = date.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{basename}/{date_fmt}"


def setup_model_path(
    model_name: Optional[str],
    dataset: HasDatasetProperties,
    use_fsdp: bool,
) -> str:
    if use_fsdp:
        raise ValueError(
            "For FSDP, all ranks must checkpoint from/to the same directory, "
            "use train.py with --model_path or run_experiment with --log_ckpts"
        )
    if model_name is None:
        basename = f"models/{dataset.name}"
    else:
        model_as_path = Path(model_name)
        if model_as_path.is_dir() and model_as_path.exists():
            basename = f"models/{dataset.name}/local_{model_as_path.name}"
        else:
            basename = f"models/{dataset.name}/{model_name}"
    return _get_model_path(basename)


def setup_logging(
    name: str,
    model_path: str,
    log_file: Optional[Path] = None,
    log_append: bool = False,
) -> logging.Logger:
    log_handlers: list[logging.StreamHandler]
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if not log_file:
        log_file = Path(f"{model_path}/train.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_append = log_file.exists()
    if log_file and ("RANK" not in os.environ or int(os.environ["RANK"]) == 0):
        log_handlers.append(
            logging.FileHandler(log_file, mode="w" if not log_append else "a")
        )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=log_handlers,
        force=True,
    )
    return logging.getLogger(name)


def reproducibility_logging(
    seed: int,
    logger: Logger,
    model_path: str,
    dataset: HasDatasetProperties,
    hparams: THparams,
    use_bfloat16: bool,
    use_fsdp: bool,
    use_tensorboard: bool,
) -> TensorboardLogger:
    logger.info(
        f"Python Version = {sys.version} "
        f"on {platform.system()} {platform.version()} ({platform.platform()})"
    )
    logger.info(
        f"Torch Version = {torch.__version__} "
        f"(CUDA {torch.version.cuda}) with bfloat16 = {use_bfloat16} "
        f"(Git commit = {torch.version.git_version})"
    )
    if is_clean_working_tree():
        logger.info(
            f"Git commit = {get_commit_hash()} on {socket.gethostname()} (tree is clean)"
        )
    else:
        logger.info(
            f"Git commit = {get_commit_hash()} on {socket.gethostname()} (tree is dirty)"
        )
        if os.environ.get("ITER_LOG_DIFF", "1") == "1":
            logger.warning(get_working_tree_diff())
    logger.info(f"Seed = {seed}")
    if use_fsdp:
        if "RANK" not in os.environ:
            raise ValueError(
                "RANK must be specified as an environment variable for FSDP"
            )
        if "WORLD_SIZE" not in os.environ:
            raise ValueError(
                "WORLD_SIZE must be specified as an environment variable for FSDP"
            )
        logger.info(f"Rank = {os.environ['RANK']} of {os.environ['WORLD_SIZE']}")
    torch.set_num_threads(2)

    return TensorboardLogger(
        model_path,
        {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "pytorch_cuda_version": torch.version.cuda,
            "pytorch_precision": "bf16" if use_bfloat16 else "fp32",
            "pytorch_git": torch.version.git_version,
            "git": get_commit_hash(),
            "seed": seed,
            "dataset": dataset.name,
            **hparams.__dict__,
        },
        disable=not use_tensorboard,
    )


def train_logging(
    model: TModel,
    hparams: THparams,
    logger: Logger,
    dataset: HasDatasetProperties,
    load_ckpt: Optional[Path],
):
    logger.info(model)

    # print dataset info
    logger.info(repr(dataset))

    logger.info(hparams.to_json())
    if load_ckpt is not None:
        logger.info(f"Loaded model from {load_ckpt.as_posix()}")
    numel = sum(param.numel() for param in model.parameters())
    num_bytes = sum(
        param.numel() * param.dtype.itemsize for param in model.parameters()
    )
    no_grad_numel = sum(
        param.numel() for param in model.parameters() if not param.requires_grad
    )
    logger.info(
        f"{numel / 1e6:.4f} M params ({numel} total) -- {num_bytes / 1e9} GiB in VRAM"
    )
    if no_grad_numel > 0:
        logger.info(str((numel - no_grad_numel) / 1e6) + " M activated params")


def setup_precision_context(
    use_bfloat16: bool,
    use_fsdp: bool,
) -> ContextManager:
    if use_bfloat16 and not use_fsdp:
        return torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def setup_profiler_context(
    do_profile: bool,
    rank: int,
) -> ContextManager:
    if do_profile and rank == 0:
        torch.cuda.memory._record_memory_history(max_entries=100_000)
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_modules=True,
            with_stack=True,
            record_shapes=True,
        )
    return contextlib.nullcontext()


def write_reproducibility_checkpoint(
    model_path: str,
    args: TrainingArgs[THparams, TDataset],
    model: Module,
    optimizer: Optimizer,
    num_epochs: int,
    reproducibility_variables: Mapping[str, Any] | None,
):
    reproducibility_variables = reproducibility_variables or {}

    if args.use_fsdp:
        state_dict, optimizer_state_dict = (
            torch.distributed.checkpoint.state_dict.get_state_dict(
                model,
                optimizer,
            )
        )
    else:
        state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

    repro_dict = {
        "model": state_dict,
        "optimizer": optimizer_state_dict,
        "hparams": args.hparams,
        "num_epochs": num_epochs,
    }

    with open(f"{model_path}/requirements.txt", "w") as f:
        subprocess.run("python -m pip freeze".split(), stdout=f)

    torch.save(repro_dict, f"{model_path}/repro.pt")
    with open(f"{model_path}/hparams.json", "w") as f:
        f.write(args.hparams.to_json() + "\n")

    for key, value in reproducibility_variables.items():
        with open(f"{model_path}/{key}.json", "w") as f:
            json.dump(value, f)

    distributed_barrier()


def load_checkpoint(
    device: str | torch.device,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    generator: Generator,
    model_ckpt: Path,
    load_optimizer_ckpt: bool,
    steps_per_epoch: int,
) -> tuple[int, int, Optional[tuple[int, int | float]]]:
    """
    Load the optimizer state and return the training epoch to start at
    """
    optimizer_state = model_ckpt / "optimizer.bin"
    if not optimizer_state.exists():
        warnings.warn(f"Cannot load optimizer state from {optimizer_state}")
        return 0, 0, None

    state = torch.load(
        model_ckpt / "optimizer.bin", weights_only=False, map_location=device
    )
    if load_optimizer_ckpt:
        warnings.warn(
            f"Loading optimizer and lr scheduler state from {optimizer_state}"
        )
        optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    generator.set_state(state["generator"].cpu())

    if load_optimizer_ckpt:
        if "epoch" in state:
            if "step" in state:
                return (
                    int(state["step"]),
                    int(state["epoch"]),
                    state.get("best_score", None),
                )
            return int(state["epoch"]) * steps_per_epoch, state["epoch"], None

        return (
            int(state["step"]),
            int(state["step"] // steps_per_epoch),
            state["best_score"],
        )
    return 0, 0, None


def load_model_checkpoint(
    model: TModel,
    config: TConfig,
    model_init: ModelInitProtocol[TConfig, TModel],
    model_path: Path,
    device: str,
    memory_checkpoints: bool,
) -> TModel:
    if memory_checkpoints:
        if _is_fully_sharded_dp_model(model):
            raise NotImplementedError
        else:
            # global _checkpoint_state_dict
            state_dict = _checkpoint_state_dict["state_dict"]
            model.load_state_dict(state_dict)
            return model
    else:
        del model
        torch.cuda.empty_cache()

        model = model_init(config, model_path)
        model = model.to(device)
        return model  # type: ignore


def train_step(
    model: TModel,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
):
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1, error_if_nonfinite=False
    )
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return norm


def oom_catcher(func: ForwardBackwardProtocol[TModel, THparams, TDataset]):
    @functools.wraps(func)
    def wrapper(
        model: TModel,
        args: TrainingArgs[THparams, TDataset],
        batch: Batch,
        device: torch.device | str,
        compute_context: AbstractContextManager,
        logger: logging.Logger,
    ):
        try:
            return func(model, args, batch, device, compute_context, logger)
        except torch.cuda.OutOfMemoryError:
            inputs = list()
            for name, tensor in batch.items():
                inputs.append(f"\n\t - {name}: {tuple(tensor.shape)}")
            logger.warning(
                f"Caught OutOfMemoryError during training, inputs were: {''.join(inputs)}"
            )
            logger.warning(torch.cuda.memory_summary(device=device, abbreviated=True))
            raise

    return wrapper


@oom_catcher
def forward_backward(
    model: TModel,
    args: TrainingArgs[THparams, TDataset],
    batch: Batch,
    device: torch.device | str,
    compute_context: AbstractContextManager,
    logger: logging.Logger,
) -> Mapping[str, torch.Tensor]:
    # possibly use a lower precision instead of 32 bit floating point ops
    with compute_context:
        # (1) forward pass
        output = model(**batch)

        if not hasattr(output, "loss"):
            raise ValueError(
                f'Model {type(model).__name__} returned {output} without a "loss" attribute'
            )

        loss = output.loss

        # (2) in case we do gradient accumulation, normalize across the batches
        if args.hparams.gradient_accumulation > 1:
            loss = loss / args.hparams.gradient_accumulation

    # (3) finally, the backward pass
    loss.backward()

    # (4) statistics: calculate the number of tokens in this batch
    num_tokens = batch.attention_mask.sum().item()

    return OrderedDict(
        loss=loss.detach(),
        batch_tokens=num_tokens,
    )


def check_parameters_for_nan(model: TModel, strict: bool):
    nan_parameters = set()
    for name, parameter in model.named_parameters():
        if parameter.isnan().any():
            nan_parameters.add(name)

    if len(nan_parameters) > 0:
        msg = (
            f"Found {len(nan_parameters)} parameters within the model that are nan and have not been "
            f"initialized: {list(sorted(nan_parameters))}"
        )
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg)


def get_training_steps(
    setup_dataloader: DataLoaderProtocol[TDataset, THparams],
    args: TrainingArgs[THparams, TDataset],
):
    if not isinstance(args.hparams, (EpochBasedTraining, StepBasedTraining)):
        raise ValueError(
            f"{type(args.hparams).__name__} must either include "
            f"num_steps and eval_every_n_steps "
            f"or num_epochs, max_epochs and eval_every_n_epochs "
        )

    epoch_dataloader = setup_dataloader(args, "train", False)

    try:
        steps_per_epoch = len(epoch_dataloader)
    except TypeError:
        if not isinstance(args.hparams, StepBasedTraining):
            raise
        steps_per_epoch = args.hparams.num_steps

    if isinstance(args.hparams, StepBasedTraining):
        args.hparams.patience *= args.hparams.eval_every_n_steps

        if args.hparams.num_steps > 0:
            return (
                args.hparams.num_steps,
                args.hparams.num_steps,
                args.hparams.eval_every_n_steps,
                steps_per_epoch,
            )
        raise ValueError(
            f"Training steps must be greater than zero, got {args.hparams.num_steps}"
        )

    if args.hparams.num_epochs > args.hparams.max_epochs:
        raise ValueError("num_epochs must be <= max_epochs")

    total_steps = steps_per_epoch * args.hparams.max_epochs
    steps_to_train = steps_per_epoch * (
        args.hparams.num_epochs or args.hparams.max_epochs
    )

    args.hparams.patience *= steps_per_epoch

    return (
        total_steps,
        steps_to_train,
        steps_per_epoch * args.hparams.eval_every_n_epochs,
        steps_per_epoch,
    )


def record_training_statistics(
    args: TrainingArgs[THparams, TDataset],
):
    # if args.
    # args.score_history["_cuda_utilization"].append()
    pass


def _restore_train_state(func):
    def wrapper(model: TModel, *args, **kwargs):
        train_state = model.training
        try:
            result = func(model, *args, **kwargs)
        finally:
            if model.training != train_state:
                warnings.warn(
                    f"Model was in {train_state=} before eval, is in {model.training=} after"
                )
                model.train(train_state)
        return result

    return wrapper


def _call_evaluate_function(
    evaluate_fn: EvaluateProtocol[TModel, TDataset, THparams, TMetrics],
    model: TModel,
    args: TrainingArgs[THparams, TDataset],
    split: str,
    dataloader_fn: DataLoaderProtocol[TDataset, THparams],
):
    metrics = evaluate_fn(
        model,
        args,
        split,
        dataloader_fn,
    )

    return metrics


@_restore_train_state
def do_evaluate(
    model: TModel,
    args: TrainingArgs[THparams, TDataset],
    evaluate_fn: EvaluateProtocol[TModel, TDataset, THparams, TMetrics],
    dataloader_fn: DataLoaderProtocol[TDataset, THparams],
    score_comparison: Callable[[float, float], bool] = operator.gt,
):
    logger = logging.getLogger("do_evaluate")

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.1)
    torch.cuda.empty_cache()

    # set model into eval mode
    old_state = model.training
    model.eval()

    with torch.no_grad():
        metrics = _call_evaluate_function(
            evaluate_fn, model, args, "eval", dataloader_fn
        )

    # restore training mode
    model.train(old_state)

    if args.use_fsdp:
        distributed_barrier()  # wait for all GPU procs to finish evaluation

    torch.cuda.empty_cache()
    time.sleep(0.1)
    torch.cuda.empty_cache()

    logger.info(metrics)

    metrics_dict = metrics.to_dict()
    args.score_history["_steps"].append(args.train_steps)
    for metric_name in args.scores_to_track:
        if metric_name not in metrics_dict:
            if metric_name == "_steps":
                continue
            raise ValueError(
                f"Cannot track metric {metric_name}, available keys are {metrics_dict.keys()}"
            )
    for metric_name, metric_value in metrics_dict.items():
        args.score_history[metric_name].append(metric_value)

    if args.hparams.optimize_for not in metrics_dict:
        raise ValueError(
            f"Cannot optimize for {args.hparams.optimize_for}, available keys are {metrics_dict.keys()}"
        )

    args.train_logger.log_metrics(metrics_dict, global_step=args.train_steps)

    outcome_score = metrics_dict[args.hparams.optimize_for]
    if args.best_score is not None:
        best_step, best_score = args.best_score
    else:
        best_step = args.train_steps
        best_score = (
            -sys.maxsize
            if safe_score_comparison(score_comparison, 1, 0)
            else sys.maxsize
        )

    stop_after_epoch = False
    try:
        new_best_score = score_comparison(outcome_score, best_score)
    except StopIteration:
        stop_after_epoch = True
        new_best_score = True
        logger.info(
            "Score comparison raised StopIteration, ending training after this epoch"
        )
    if new_best_score:
        args.best_score = args.train_steps, outcome_score

    if (
        args.hparams.patience
        and args.best_score
        and best_step + args.hparams.patience < args.train_steps
    ):
        logger.info(
            f"No improvement for {args.hparams.patience} steps, aborting training after step {args.train_steps}"
        )
        return False, True, outcome_score, True
    return new_best_score, stop_after_epoch, outcome_score, False


@finalise_distributed_environment_after_exit
def do_train(
    dataset: TDataset,
    hparams: THparams,
    model_class: type[TModel],
    config_class: type[TConfig],
    evaluate_fn: EvaluateProtocol[TModel, TDataset, THparams, TMetrics],
    optimizer_fn: OptimizerInitProtocol[TModel, TDataset, THparams],
    dataloader_fn: DataLoaderProtocol[TDataset, THparams],
    config_init_fn: ConfigInitProtocol[TDataset, THparams, TConfig] | None = None,
    step_fn: ForwardBackwardProtocol[TModel, THparams, TDataset] = forward_backward,
    transformer: Optional[str] = None,
    model_path: Optional[str] = None,
    log_file: Optional[Path] = None,
    log_append: bool = False,
    logger: Optional[logging.Logger] = None,
    seed: int = 42,
    use_tqdm: bool = True,
    use_bfloat16: bool = False,
    use_fsdp: bool = False,
    use_tensorboard: bool = True,
    num_epochs: int = 0,
    dont_ckpt: bool = False,
    do_profile: bool = False,
    load_ckpt: Optional[Path] = None,
    load_optimizer_ckpt: Optional[bool] = True,
    track_metrics: Optional[set[str]] = None,
    track_title: Optional[str] = None,
    model_init: Optional[ModelInitProtocol[TConfig, TModel]] = None,
    plot_kwargs: Optional[Mapping[str, Any]] = None,
    score_comparison: Optional[Callable[[float, float], bool]] = None,
    loss_is_batch_accumulated: bool = False,
    allow_nan_parameters: bool = False,
    memory_checkpoints: bool = False,
    device: Optional[str] = None,
    new_checkpoint_logic: Optional[bool] = True,
    check_nan_grad_norm: bool = False,
    reproducibility_variables: Optional[Mapping[str, Any]] = None,
    hooks: Optional[Hooks] = None,
) -> Optional[
    TMetrics
    | tuple[TMetrics, ...]
    | MetricsDict[TMetrics]
    | tuple[MetricsDict[TMetrics], ...]
]:
    """
    Training implementation

    Sets up datasets, seeding, reproducibility, model.
    Contains the main training loop, handles evaluation, too.
    """
    if device is None:
        device = "cpu" if not torch.cuda.is_available() else "cuda:0"

    if hooks is None:
        hooks = Hooks()

    if plot_kwargs is None or not plot_kwargs:
        plot_kwargs = {}

    if load_optimizer_ckpt is None:
        warnings.warn(
            "load_optimizer_ckpt is None, defaulting to load_optimizer_checkpoint=True"
        )
        load_optimizer_ckpt = True

    if score_comparison is None:
        if "loss" in hparams.optimize_for:
            score_comparison = training_operators.lt
        else:
            score_comparison = training_operators.gt

    generator = setup_seed(seed)
    rank, world_size = 0, 1

    track_metrics = (track_metrics or set()) | {hparams.optimize_for, "_steps"}
    if not model_path:
        model_path = setup_model_path(
            transformer,
            dataset,
            use_fsdp,
        )

    if logger is None:
        logger = setup_logging("train", model_path, log_file, log_append)

    # log seed, torch and cuda version, whether we use mixed precision, etc...
    train_logger = reproducibility_logging(
        seed,
        logger,
        model_path,
        dataset,
        hparams,
        use_bfloat16,
        use_fsdp,
        use_tensorboard,
    )

    args = TrainingArgs(
        local_device=device,
        seed=seed,
        generator=generator,
        hparams=hparams,
        dataset=dataset,
        use_fsdp=use_fsdp,
        use_tqdm=use_tqdm,
        train_logger=train_logger,
        scores_to_track=track_metrics,
        console_logger=logger,
    )

    if config_init_fn is None:
        config_init_fn = config_class

    config = config_init_fn(dataset, hparams)

    if model_init is None:
        model_init = setup_model(model_class, for_fully_sharded_dp=use_fsdp)

    if load_ckpt:
        model = model_init(config, str(load_ckpt))
    else:
        model = model_init(config, None)

    # check if the model contains any uninitialized parameters
    check_parameters_for_nan(model, strict=not allow_nan_parameters)

    # log model arch, hparams, # params, model features
    train_logging(model, hparams, logger, dataset, load_ckpt)

    # call the model init hook
    hooks.on_model_init(model, args)

    compute_loss_context_mngr = setup_precision_context(
        use_bfloat16,
        use_fsdp,
    )

    if not use_fsdp:
        model = model.to(device)

    (total_steps, steps_to_train, interval_to_eval_at, steps_per_epoch) = (
        get_training_steps(
            dataloader_fn,
            args,
        )
    )

    optimizer, lr_scheduler = optimizer_fn(
        model=model,
        args=args,
        num_training_steps=total_steps,
    )

    args.optimizer = optimizer
    args.lr_scheduler = lr_scheduler

    if load_ckpt:
        args.train_steps, args.epoch, _best_score = load_checkpoint(
            device,
            optimizer,
            lr_scheduler,
            generator,
            load_ckpt,
            load_optimizer_ckpt,
            steps_per_epoch,
        )
        if _best_score is not None:
            args.best_score = _best_score
        logger.info(f"Continuing training from step {args.train_steps}")

    if not memory_checkpoints:
        write_reproducibility_checkpoint(
            model_path,
            args,
            model,
            optimizer,
            num_epochs,
            reproducibility_variables,
        )

    postfix: MutableMapping[str, Any] = collections.OrderedDict(
        epoch=args.epoch, score=0.0, loss=0.0, g_norm=0.0, tokens=0
    )

    if args.best_score is not None:
        postfix["score"] = float(args.best_score[1])

    args.start_step = args.train_steps
    with (
        logging_redirect_tqdm(),
        tqdm(
            desc="Training" + (f" on {world_size} GPUs" if world_size > 1 else ""),
            total=steps_to_train,
            initial=args.start_step,
            leave=True,
            disable=not use_tqdm,
            position=0,
            mininterval=0.1,
        ) as progress_bar,
        setup_profiler_context(do_profile=do_profile, rank=rank) as prof,
    ):
        while args.train_steps < steps_to_train:
            model.train()

            # create dataloader based on hparams, world size & rank (when use_fsdp=True), seed
            dataloader = dataloader_fn(
                args,
                "train",
                True,
            )

            dataloader = ProfilingDataLoaderIterator(dataloader)
            postfix.update(epoch=args.epoch)
            args.epoch += 1

            stats = torch.zeros(4, device=device)
            stop_after_epoch = False
            for cpu_batch in dataloader:
                batch_size = first(cpu_batch.values()).size(0)

                # (1) move the batch to the desired device, non-blocking if possible
                batch = cpu_batch.to(device, non_blocking=None)

                # (2) call forward and backward using the model and batch
                loss_info = step_fn(
                    model,
                    args,
                    batch,
                    device,
                    compute_loss_context_mngr,
                    logger,
                )

                # (3) extract the loss tensor from the step function result
                if isinstance(loss_info, MutableMapping):
                    loss_tensor = loss_info.pop("loss")
                else:
                    warnings.warn(
                        "Returning just a torch.Tensor from the step function is subject to removal in the future, return a Mapping instead",
                        DeprecationWarning,
                    )
                    loss_tensor = loss_info

                # (4) in case we are profiling, abort after a certain amount of steps
                if do_profile and ((args.train_steps + 1) % 3 == 0):
                    stop_after_epoch = True
                    break

                # (5) measure training statistics
                stats[0] += loss_tensor.detach()
                stats[1] += batch_size if loss_is_batch_accumulated else 1

                args.grad_steps += 1
                args.train_steps += 1

                # (6) optimizer step, in case we did enough forward passes
                if args.grad_steps >= hparams.gradient_accumulation:

                    # (7) call the optimizer and update the lr scheduler
                    norm = train_step(model, optimizer, lr_scheduler).detach()
                    if check_nan_grad_norm and (
                        norm.isinf() or norm.isnan() or norm.isneginf()
                    ):
                        logger.warning(
                            f"Gradient norm of model is nan, inf or neginf ({norm}), aborting training ..."
                        )
                        stop_after_epoch = True
                        break

                    # (8) update gradient norm statistics
                    stats[2] += norm
                    args.grad_steps = 0

                # (9) update total tokens statistics
                if "attention_mask" in batch:
                    num_tokens = batch["attention_mask"].sum()
                    stats[3] += num_tokens

                # (10) sync statistics across devices,
                if args.train_steps % hparams.gradient_accumulation == 0:
                    if use_fsdp:
                        dist.all_reduce(stats, dist.ReduceOp.SUM)
                    local_stats = stats.clone().cpu()

                    # reset the distributed stats
                    stats.zero_()

                    accumulated_loss, num_batches_or_steps, g_norm, global_tokens = (
                        local_stats.tolist()
                    )
                    global_loss = accumulated_loss / max(1, num_batches_or_steps)
                    global_norm = g_norm / (
                        max(1, num_batches_or_steps // hparams.gradient_accumulation)
                    )

                    args.score_history["_loss"].append((args.train_steps, global_loss))
                    args.score_history["_norm"].append((args.train_steps, global_norm))

                    postfix.update(
                        loss=global_loss,
                        g_norm=global_norm,
                        tokens=postfix["tokens"] + global_tokens,
                    )

                    if isinstance(loss_info, Mapping):
                        postfix.update(loss_info)

                    if do_profile:
                        postfix.update(dl=dataloader.step_time)

                progress_bar.set_postfix(postfix, refresh=False)
                progress_bar.update(1)

                del batch
                del loss_tensor

                # (11) evaluation in case we hit a breakpoint!
                if (
                    args.train_steps % interval_to_eval_at
                ) == 0 and args.grad_steps == 0:
                    new_best_score, stop_after_epoch, new_score, patience_exceeded = (
                        do_evaluate(
                            model,
                            args,
                            evaluate_fn,
                            dataloader_fn,
                            score_comparison,
                        )
                    )

                    if new_best_score or new_checkpoint_logic:
                        checkpoint_path = Path(model_path)
                        if new_checkpoint_logic:
                            checkpoint_path = (
                                checkpoint_path / f"step-{args.train_steps}"
                            )

                        save_checkpoint(
                            args,
                            model,
                            optimizer,
                            lr_scheduler,
                            checkpoint_path.as_posix(),
                            dont_ckpt,
                            memory_checkpoints,
                        )

                        save_loss_plot(
                            args,
                            track_title,
                            checkpoint_path / "loss.pdf",
                            **plot_kwargs,
                            save_plot_copy_to_parent_dir=new_checkpoint_logic,
                        )

                        if new_best_score:
                            postfix.update(score=float(new_score))

                            if dont_ckpt:
                                logger.info(
                                    f"Found new best checkpoint ({args.hparams.optimize_for}, {new_score:.8f}), but checkpointing is disabled"
                                )
                            elif memory_checkpoints:
                                logger.info(
                                    f"Found new best checkpoint ({args.hparams.optimize_for}, {new_score:.8f}), saving in-memory"
                                )
                            else:
                                logger.info(
                                    f"Found new best checkpoint ({args.hparams.optimize_for}, {new_score:.8f}), saved to {checkpoint_path.as_posix()}"
                                )
                        elif new_checkpoint_logic:
                            logger.info(
                                f"Saving checkpoint at {args.train_steps} training steps to {checkpoint_path.as_posix()}"
                            )

                    if patience_exceeded:
                        stop_after_epoch = True
                        break

                if args.train_steps >= steps_to_train or stop_after_epoch:
                    break
            if stop_after_epoch:
                break

    if use_fsdp:
        logger.info("Waiting at save barrier")
        distributed_barrier()
    if not dont_ckpt and args.best_score is not None:
        checkpoint_path = Path(model_path)
        if new_checkpoint_logic:
            best_step = args.best_score[0]
            checkpoint_path = checkpoint_path / f"step-{best_step}"
        logger.info(f"Loading checkpoint from {checkpoint_path.as_posix()}")

        model = load_model_checkpoint(
            model,
            config,
            model_init,
            checkpoint_path,
            device,
            memory_checkpoints,
        )

        shutil.copytree(
            checkpoint_path, checkpoint_path.parent, symlinks=True, dirs_exist_ok=True
        )

    model_dir = Path(model_path)
    if not do_profile:
        test_metrics = do_test(
            args,
            model,
            evaluate_fn,
            dataloader_fn,
            device,
        )
        logger.info(test_metrics)

        save_metrics_to_file(
            test_metrics, args.score_history, model_dir / "metrics.json"
        )

        train_logger.log_test_metrics(dict(), test_metrics.to_dict())

        save_loss_plot(
            args,
            track_title,
            model_dir / "loss.pdf",
            **plot_kwargs,
        )
        return test_metrics

    else:
        save_profiler(do_profile, rank, prof, model_dir / "profiler.json")
        return None


def do_test(
    args: TrainingArgs[THparams, TDataset],
    model: TModel,
    evaluate_fn: EvaluateProtocol[TModel, TDataset, THparams, TMetrics],
    dataloader_fn: DataLoaderProtocol[TDataset, THparams],
    device: Optional[str] = None,
) -> TMetrics | MetricsDict[TMetrics]:
    if device is None:
        device = "cpu" if not torch.cuda.is_available() else "cuda:0"

    logger = logging.getLogger("test")
    logger.info("TESTING")

    if model.device != device:
        model = model.to(device)

    with (
        logging_redirect_tqdm(),
        torch.no_grad(),
    ):
        test_metrics = _call_evaluate_function(
            evaluate_fn,
            model,
            args,
            "test",
            dataloader_fn,
        )
        return test_metrics


# https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
def smooth_curve(
    scalars: torch.Tensor, weight: float
) -> torch.Tensor:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = scalars.clone()
    for pos, point in enumerate(scalars.tolist()):
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed[pos] = smoothed_val  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def save_profiler(
    do_profile: bool,
    rank: int,
    profiler: torch.profiler.profile | contextlib.nullcontext,
    output_file: Path,
):
    logger = logging.getLogger("profiler")
    if do_profile and isinstance(profiler, torch.profiler.profile) and rank == 0:
        profiler.export_chrome_trace(output_file.as_posix())
        logger.info(
            f"Saved chrome trace profiler.json to {output_file.absolute().as_posix()} - to visualize open about:tracing"
            f" in Chrome"
        )

        try:
            memory_snapshot_file = output_file.with_name("memory_snapshot.pickle")
            torch.cuda.memory._dump_snapshot(memory_snapshot_file)
            logger.info(
                f"Saved memory snapshot to {memory_snapshot_file.absolute().as_posix()} - to visualize run "
                f'"python venv/lib/python3.X/site-packages/torch/cuda/_memory_viz.py trace_plot snapshot.pickle '
                f'-o snapshot.html", more details on https://pytorch.org/blog/understanding-gpu-memory-1/'
            )
        except Exception as error:
            logger.error(f"Could not save memory snapshot {error}")


def save_loss_plot(
    args: TrainingArgs[THparams, TDataset],
    title: Optional[str],
    output_file: Path,
    **kwargs,
):
    loss_steps, loss_history = (
        torch.tensor(args.score_history["_loss"]).cpu().unbind(dim=1)
    )
    grad_steps, grad_norm_history = (
        torch.tensor(args.score_history["_norm"]).cpu().unbind(dim=1)
    )

    score_history = {
        key: torch.tensor(value).cpu()
        for key, value in args.score_history.items()
        if key in (args.scores_to_track | {"_steps"})
    }

    _save_loss_plot(
        title,
        loss_steps,
        loss_history,
        grad_steps,
        grad_norm_history,
        score_history,
        args.hparams.optimize_for,
        output_file,
        **kwargs,
    )


def _save_loss_plot(
    title: Optional[str],
    loss_steps: torch.Tensor,
    loss_history: torch.Tensor,
    grad_steps: torch.Tensor,
    grad_norm_history: torch.Tensor,
    score_history: MutableMapping[str, torch.Tensor],
    optimize_for: str,
    output_file: Path,
    **kwargs,
):
    if kwargs.get("do_pickle", True):
        torch.save(
            (
                title,
                loss_steps,
                loss_history,
                grad_steps,
                grad_norm_history,
                score_history,
                optimize_for,
                kwargs,
            ),
            output_file.with_name("loss.pt"),
        )

    if MATPLOTLIB:
        logger = logging.getLogger("matplotlib")

        score_steps = score_history.pop("_steps")

        from matplotlib.axes import Axes
        from matplotlib.colors import ListedColormap

        @lru_cache(2)
        def get_listed_colormap(name: str) -> tuple[ListedColormap, tuple]:
            colormap = plt.colormaps[name]
            if isinstance(colormap, ListedColormap):
                if isinstance(colormap.colors, tuple):
                    return colormap, colormap.colors
                raise ValueError(f"Colormap {name} does not define a tuple of colors")
            raise ValueError(f"Colormap {name} is not a listed colormap")

        cmap, colors = get_listed_colormap("Set3")
        cmap_index = 0  # , cmap_step = 0.0, 1 / (len(train_score_trend) - 1)
        loss_cmap, loss_colors = get_listed_colormap("tab20b")

        marker_style = dict(
            markersize=10,
            markeredgecolor="black",
            color="black",
        )

        def _score_name(k: str) -> str:
            k = k.replace("_", " ")
            k = " ".join(e.capitalize() for e in k.split() if e)
            return k

        score_names = {key: _score_name(key) for key in score_history.keys()}
        if optimize_for in score_names:
            score_names[optimize_for] = score_names[optimize_for] + " (*)"

        plots = kwargs.get(
            "plots",
            [
                {
                    "left": ["_loss"],
                    "left_label": "Loss",
                    "left_bounds": (None, None),
                    "left_scale": "log",
                    "left_labels": ["Loss (smoothed)"],
                    "left_fmts": [None],
                    "left_margin": None,
                    "right": list(sorted(score_history.keys())),
                    "right_label": "Metrics",
                    "right_bounds": (-0.05, 1.05),
                    "right_scale": "linear",
                    "right_labels": [
                        score_names[k] for k in sorted(score_history.keys())
                    ],
                    "right_fmts": ["*--"] * len(score_history),
                    "right_margin": None,
                    "ncol": (
                        2
                        if len(score_history) % 2 == 0
                        else (1 if len(score_history) <= 1 else 3)
                    ),
                }
            ],
        )

        fig, axis = plt.subplots(
            1,
            len(plots),
            layout="constrained",
            figsize=kwargs.get("figsize", (11.8, 5.2)),
        )

        smoothed_train_loss_history = smooth_curve(loss_history, 0.9)
        smoothed_gradient_norm_history = smooth_curve(grad_norm_history, 0.9)

        plottable_series = {
            key: (torch.stack((score_steps, value)) if value.dim() == 1 else value)
            for key, value in score_history.items()
        }
        plottable_series["_loss"] = torch.stack(
            (loss_steps, smoothed_train_loss_history)
        )
        plottable_series["_loss_raw"] = torch.stack((loss_steps, loss_history))
        plottable_series["_grad_norm"] = torch.stack(
            (grad_steps, smoothed_gradient_norm_history)
        )
        plottable_series["_grad_norm_raw"] = torch.stack(
            (grad_steps, grad_norm_history)
        )

        color_presets = kwargs.get(
            "color_presets",
            {
                "_loss": dict(c=loss_colors[12]),
                "_loss_raw": dict(c=loss_colors[14]),
                "_grad_norm": dict(c=loss_colors[1], alpha=0.4),
                "_grad_norm_raw": dict(c=loss_colors[3], alpha=0.4),
            },
        )

        def _apply_bounds(a: Axes, m, l, u):
            if l is None:
                l = a.get_ybound()[0]
            elif isinstance(l, str):
                l = plottable_series[l][1].min() * 0.99
            if u is None:
                u = a.get_ybound()[1]
            elif isinstance(u, str):
                u = plottable_series[u][1].max() * 1.01
            a.set_ybound(l, u)
            if m is not None:
                a.margins(*m)

        def _setup_legend(a: Axes, o: Optional[Axes], ncol: int):
            handles, labels = a.get_legend_handles_labels()
            if o is not None:
                score_handles, score_labels = o.get_legend_handles_labels()
                handles = handles + score_handles
                labels = labels + score_labels
            a.legend(
                handles,
                labels,
                ncol=ncol,
                loc=kwargs.get("loc", "upper center"),
                bbox_to_anchor=kwargs.get("bbox_to_anchor", (0.5, -0.1)),
            )

        num_evaluations = score_steps.numel()
        num_visible_evaluations = max(1, math.floor(num_evaluations / 10))

        def _process_metric(
            a: Axes, m: str, l: str, f: Optional[str], c: Optional[Any]
        ):
            nonlocal cmap_index

            x, y = plottable_series[m].unbind(dim=0)
            if m not in color_presets:
                x = x.numpy()[::-num_visible_evaluations]
                y = y.numpy()[::-num_visible_evaluations]
            if len(x) < 1:
                return
            args: tuple[Any, ...] = x, y

            if m not in color_presets:
                if c is None:
                    c = colors[cmap_index]
                    cmap_index += 1

                metric_kwargs = dict(
                    marker_style,
                    markerfacecolor=c,
                    markeredgecolor="black",
                    color="black",
                )

                if f is not None:
                    args = args + (f,)
            else:
                metric_kwargs = color_presets[m]

            a.plot(
                *args,
                label=l,
                **metric_kwargs,
            )

        for pos, plot in enumerate(plots):
            if not isinstance(axis, Axes):
                ax: Axes = axis[pos]
            else:
                ax = axis
            ax.set_xlabel(plot.get("xaxis_label", "Training Steps"))
            ax.set_ylabel(plot["left_label"])
            ax.set_yscale(plot["left_scale"])
            if plot.get("xaxis_formatter", None) is not None:
                ax.xaxis.set_major_formatter(plot.get("xaxis_formatter"))
                ax.xaxis.set_minor_formatter(plot.get("xaxis_formatter"))
            if plot.get("yaxis_formatter", None) is not None:
                ax.yaxis.set_major_formatter(plot.get("yaxis_formatter"))
                ax.yaxis.set_minor_formatter(plot.get("yaxis_formatter"))

            left_colors = plot.get("left_colors", [None] * len(plot["left"]))

            for metric, label, fmt, color in zip(
                plot["left"], plot["left_labels"], plot["left_fmts"], left_colors
            ):
                _process_metric(ax, metric, label, fmt, color)

            _apply_bounds(ax, plot["left_margin"], *plot["left_bounds"])

            twin_ax = None
            if plot["right"]:
                twin_ax = ax.twinx()
                twin_ax.set_ylabel(plot["right_label"])
                twin_ax.set_yscale(plot["right_scale"])
                if plot.get("twinaxis_formatter", None) is not None:
                    twin_ax.yaxis.set_major_formatter(plot.get("twinaxis_formatter"))

                right_colors = plot.get("right_colors", [None] * len(plot["right"]))

                for metric, label, fmt, color in zip(
                    plot["right"],
                    plot["right_labels"],
                    plot["right_fmts"],
                    right_colors,
                ):
                    _process_metric(twin_ax, metric, label, fmt, color)

                _apply_bounds(twin_ax, plot["right_margin"], *plot["right_bounds"])
            _setup_legend(ax, twin_ax, plot["ncol"])
            if plot.get("post_ax_fn", None) is not None:
                plot["post_ax_fn"](ax, twin_ax)

        if title is not None:
            fig.suptitle(title.replace("\\n", "\n"), fontsize=10)

        if kwargs.get("post_fig_fn", None) is not None:
            kwargs["post_fig_fn"](fig, axis)

        fig.savefig(output_file)
        if not isinstance(output_file, Path):
            output_file = Path(output_file)

        if kwargs.get("save_plot_copy_to_parent_dir", False):
            parent_file = output_file.parent.parent / output_file.name
            if score_steps.numel() <= 1:
                logger.info(
                    f'Saved train history plot to "{parent_file.absolute().as_posix()}"'
                )
            else:
                logger.info(
                    f'Saved train history plot to "{output_file.absolute().as_posix()}"'
                )
            fig.savefig(parent_file)
        else:
            logger.info(
                f'Saved train history plot to "{output_file.absolute().as_posix()}"'
            )
        plt.close(fig)
    else:
        warnings.warn(
            "Not creating loss plot since matplotlib is not installed", stacklevel=2
        )


def save_metrics_to_file(
    metrics: (
        TMetrics
        | tuple[TMetrics, ...]
        | MetricsDict[TMetrics]
        | tuple[MetricsDict[TMetrics], ...]
    ),
    score_history: Mapping[str, list[int | float | tuple[int | float, ...]]],
    file: Path,
):
    save: dict[str, Any] = dict()
    save["outcomes"] = ensure_no_tensors_in_value(score_history)

    if isinstance(metrics, tuple):
        save["test_metrics"] = ensure_no_tensors_in_value(
            [elem.to_dict() for elem in metrics]
        )
    else:
        save["test_metrics"] = ensure_no_tensors_in_value(metrics.to_dict())

    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w") as f:
        json.dump(save, f)


def ensure_no_tensors_in_value(v) -> Any:
    if isinstance(v, list) or isinstance(v, tuple):
        return type(v)(ensure_no_tensors_in_value(e) for e in v)
    elif isinstance(v, dict):
        return {k: ensure_no_tensors_in_value(e) for k, e in v.items()}
    elif isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return v.tolist()
    elif type(v) not in {str, int, float}:
        raise ValueError(v)
    else:
        return v
