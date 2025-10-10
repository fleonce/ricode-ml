import dataclasses


@dataclasses.dataclass()
class TrainingConfig:

    # knob whether we want to normalize the loss we obtain by the number of gradient accumulation steps
    normalize_loss_by_gradient_accumulation: bool = True

    # whether to measure gpu memory allocation
    track_cuda_statistics: bool = False
