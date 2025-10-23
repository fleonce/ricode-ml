import math
import warnings
from typing import Any, Callable, Mapping, Optional, TYPE_CHECKING, TypeVar

from torch.nn import Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torcheval.metrics import Mean
from torcheval.metrics.toolkit import sync_and_compute
from tqdm import tqdm

from ricode.ml.distributed import distributed_world_size
from ricode.ml.training_basics import BasicMetrics, MetricsDict
from ricode.ml.training_datasets import BasicDataset, ProxyTrainingArgs
from ricode.ml.training_types import (
    DataLoaderProtocol,
    EvaluateProtocol,
    HasOptimizerArgs,
    ModelInitProtocol,
    TConfig,
    TDataset,
    THparams,
    TModel,
    TrainingArgs,
)

try:
    import torchao

    TORCHAO = True
except ImportError:
    TORCHAO = False


def get_grouped_parameters(
    model: TModel,
    weight_decay: float,
    lr: float,
    no_decay_fields: list[str] | None = None,
):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
    no_decay += no_decay_fields or []

    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0,
            "lr": lr,
        },
    ]
    return grouped_parameters


def setup_model(
    model_class: type[TModel],
):
    def _model_init(config, **kwargs):
        return model_class(config, **kwargs)

    return _model_init


if TORCHAO:
    import torchao.prototype.quantized_training
    from torchao import quantize_

    def setup_quantized_model(
        model_class: type[TModel],
        quantization_config: str,
        quantization_filter_fn: Optional[Callable[[Module, str], bool | None]] = None,
        quantization_config_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        quantization_config_kwargs = quantization_config_kwargs or dict()

        _model_init = setup_model(model_class)

        def _quantized_init(config, **kwargs):
            model = _model_init(config, **kwargs)

            # quantize the model inplace, no modifications to the model required!
            quantize_config = getattr(
                torchao.prototype.quantized_training, quantization_config
            )
            quantize_(
                model,
                quantize_config(**quantization_config_kwargs),
                quantization_filter_fn,
            )

            return model

        return _quantized_init

else:

    def setup_quantized_model(
        model_class: type[TModel],
        quantization_config: str,
        quantization_filter_fn: Optional[Callable[[Module, str], bool | None]] = None,
        quantization_config_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        warnings.warn("torchao must be installed for quantization to work")
        return setup_model(model_class)

    def setup_float8_model(
        model_class: type[TModel],
        model_init_fn: ModelInitProtocol[TConfig, TModel] | None = None,
    ):
        warnings.warn("torchao must be installed for float8 training to work")
        return setup_model(model_class)


# partially from https://github.com/lyutyuh/ASP/blob/31ac48dfe9d85cb0b3ad22d43667104db38f2dc2/util/func.py#L399-L413
def get_scheduler_lambda(
    scheduler_type: str, warmup_steps: float | int, total_steps: int
):
    if isinstance(warmup_steps, float):
        # convert ratio to integer
        return get_scheduler_lambda(
            scheduler_type, int(warmup_steps * total_steps), total_steps
        )
    if scheduler_type == "linear":
        return get_scheduler_lambda("linear_with_warmup", 0, total_steps)
    elif scheduler_type == "linear_with_warmup":

        def lambda_rule(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            return max(
                0.0,
                float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
            )

        return lambda_rule
    elif scheduler_type == "constant":
        return lambda step: 1.0
    elif scheduler_type == "constant_with_warmup":
        return lambda step: min(1.0, float(step) / float(max(1, warmup_steps)))
    elif scheduler_type == "inverse_sqrt":
        return get_scheduler_lambda("inverse_sqrt_with_warmup", 0, total_steps)
    elif scheduler_type == "inverse_sqrt_with_warmup":

        def lambda_rule(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            timescale = warmup_steps or total_steps / 100
            shift = timescale - warmup_steps
            decay = 1.0 / math.sqrt((step + shift) / timescale)
            return decay

        return lambda_rule
    elif scheduler_type == "cosine_to10x_with_warmup":

        def lambda_rule(step):
            if step < warmup_steps:
                # warmup the learning rate from 0 to 1 over the first 'warmup_steps' steps
                return float(step) / float(max(1, warmup_steps))

            if step >= total_steps:
                return 0.1

            num_normal_steps = total_steps - warmup_steps
            effective_step = step - warmup_steps

            # cosine decay to 10 % of the original learning rate over training, following
            # https://arxiv.org/pdf/2203.15556#appendix.B, Figure A1
            # also follows NeoBERT, however, they train with a constant LR for the last 100 K model_setup steps (10 %)
            decay = math.cos(math.pi * effective_step / num_normal_steps) / 2 + 0.5
            decay = 0.1 + 0.9 * decay

            return decay

        return lambda_rule
    elif scheduler_type == "cosine_to10x":
        return get_scheduler_lambda("cosine_to10x_with_warmup", 0.0, total_steps)
    else:
        raise ValueError(f"Unknown scheduler type {scheduler_type}")


def setup_optimizers(
    model: TModel,
    args: TrainingArgs[THparams, TDataset],
    num_training_steps: int,
):
    if TYPE_CHECKING and not isinstance(args.hparams, HasOptimizerArgs):
        raise ValueError

    effective_batch_size = args.hparams.batch_size * args.hparams.gradient_accumulation

    lr = args.hparams.lr
    if args.hparams.scale_lr:
        lr = lr * effective_batch_size

    optimizer = AdamW(
        get_grouped_parameters(model, args.hparams.weight_decay, lr),
        fused=False,
        weight_decay=0.1,
        lr=1e-4,
    )
    lr_scheduler = get_scheduler_lambda(
        args.hparams.lr_scheduler, args.hparams.warmup_steps, num_training_steps
    )
    lr_scheduler = LambdaLR(optimizer, [lr_scheduler, lr_scheduler])
    return optimizer, lr_scheduler


_T_cont = TypeVar("_T_cont", contravariant=True)
_U = TypeVar("_U", bound=BasicDataset)
_V = TypeVar("_V")
_T_dataset = TypeVar("_T_dataset")
_T_hparams = TypeVar("_T_hparams")
_T_cov_metrics = TypeVar("_T_cov_metrics", bound=BasicMetrics, covariant=True)


def evaluate_multikey_split_datasets(
    func: EvaluateProtocol[_T_cont, TDataset, THparams, _T_cov_metrics]
) -> EvaluateProtocol[_T_cont, TDataset, THparams, _T_cov_metrics]:
    def inner(
        model: _T_cont,
        args: TrainingArgs[THparams, TDataset],
        split: str,
        dataloader_fn: DataLoaderProtocol[TDataset, THparams],
    ) -> _T_cov_metrics | MetricsDict[_T_cov_metrics]:
        if isinstance(args.dataset[split], dict):
            if len(args.dataset[split]) == 0:
                raise ValueError("Cannot evaluate zero datasets")

            metrics_per_key = dict()
            for key in args.dataset[split].keys():
                result = func(model, ProxyTrainingArgs(args, key), split, dataloader_fn)  # type: ignore
                metrics_per_key[key] = result

            metrics = MetricsDict(None, **metrics_per_key)
            return BasicMetrics.from_dict(metrics.to_dict())
        return func(model, args, split, dataloader_fn)

    return inner


def multistage_evaluate_function(
    *funcs: EvaluateProtocol[_T_cont, TDataset, THparams, _T_cov_metrics]
):
    if len(funcs) == 0:
        return False

    def inner(
        model: _T_cont,
        args: TrainingArgs[THparams, TDataset],
        split: str,
        dataloader_fn: DataLoaderProtocol[TDataset, THparams],
    ) -> _T_cov_metrics | MetricsDict[_T_cov_metrics]:
        metrics = None
        for func in funcs:
            func_metrics = func(
                model,
                args,
                split,
                dataloader_fn,
            )
            if metrics is None:
                metrics = func_metrics
            else:
                for key, value in func_metrics.__dict__.items():
                    setattr(metrics, key, value)
        return metrics

    return inner


@evaluate_multikey_split_datasets
def default_evaluate_function(
    model: TModel,
    args: TrainingArgs[THparams, TDataset],
    split: str,
    dataloader_fn: DataLoaderProtocol[TDataset, THparams],
) -> BasicMetrics:
    device = model.device

    avg_loss = Mean(device=device)
    for batch in tqdm(
        dataloader_fn(args, split, False),
        leave=False,
        position=0,
        desc="Evaluating loss",
    ):
        loss = model(**batch.to(device))[0]

        avg_loss.update(loss)

    if distributed_world_size() > 1:
        return BasicMetrics(loss=sync_and_compute(avg_loss).item())
    return BasicMetrics(loss=avg_loss.compute().item())
