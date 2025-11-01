import queue
from collections import OrderedDict
from queue import SimpleQueue
from threading import Thread
from time import sleep, time
from typing import Optional

import attrs
from tqdm.std import tqdm

from ricode.ml._training.utils import (
    _format_to_memory_units,
    _format_to_percentage,
    _format_to_powers_of_1000,
    format_to_energy_usage,
)
from ricode.ml.training_types import TDataset, THparams, TrainingArgs
from ricode.nvidia.nvml import setup_nvml


@attrs.define
class EnergyStatistics:
    train_energy: float
    validation_energy: float

    power: float
    memory: int
    gpu_util: int
    memory_util: int


@setup_nvml
def watcher(
    queue: SimpleQueue[Optional[EnergyStatistics]],
    args: TrainingArgs[THparams, TDataset],
    interval: float = 0.1,
):
    local_device = "cuda:0"
    if local_device is not None and local_device.startswith("cuda:"):
        from pynvml import nvmlDeviceGetHandleByIndex

        device_handle = nvmlDeviceGetHandleByIndex(int(local_device[len("cuda:") :]))
    else:
        return

    last_known_step = -1

    last_time = time()
    train_sample_steps = 0
    train_power_usage = 0
    train_energy_usage = 0
    train_time = 0
    validation_sample_steps = 0
    validation_power_usage = 0
    validation_energy_usage = 0
    validation_time = 0

    keys = ["_power", "_memory", "_gpu_util", "_memory_util", "_time", "_energy"]
    measurements = {key: [] for key in keys}

    was_inside_eval = False

    while not args.done:
        sleep(interval)

        # inside_eval = 30 < i < 40

        from pynvml import (
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetPowerUsage,
            nvmlDeviceGetUtilizationRates,
        )

        milli_watt_usage = nvmlDeviceGetPowerUsage(device_handle)
        memory = nvmlDeviceGetMemoryInfo(device_handle)
        utilization = nvmlDeviceGetUtilizationRates(device_handle)

        used_memory = memory.used
        # used_memory_share = memory.used / memory.total
        gpu_util = utilization.gpu
        memory_util = utilization.memory

        time_now = time()
        time_diff = time_now - last_time

        last_time = time_now
        is_same_step = last_known_step == args.train_steps
        last_known_step = args.train_steps
        milli_watt_hours = (milli_watt_usage / 3600) * time_diff

        if queue.qsize() < 100:
            queue.put(
                EnergyStatistics(
                    train_energy_usage,
                    validation_energy_usage,
                    milli_watt_usage,
                    used_memory,
                    gpu_util,
                    memory_util,
                )
            )
            # queue.put((train_energy_usage + validation_energy_usage, train_energy_usage, validation_energy_usage))

        if args.inside_eval:
            # mW to kW
            validation_time += time_diff
            # validation_energy_usage += milli_watt_hours
            validation_power_usage += milli_watt_usage
            validation_sample_steps += 1
            was_inside_eval = True

            for key, value in zip(
                keys,
                [
                    milli_watt_usage,
                    used_memory,
                    gpu_util,
                    memory_util,
                    time_diff,
                    milli_watt_hours,
                ],
            ):
                measurements[key].append(value)
            continue
        else:
            if was_inside_eval:
                keys = [
                    "_power",
                    "_memory",
                    "_gpu_util",
                    "_memory_util",
                    "_time",
                    "_energy",
                ]
                avg_milli_watts = sum(measurements["_power"]) / len(
                    measurements["_power"]
                )
                avg_memory_share = sum(measurements["_memory"]) / len(
                    measurements["_memory"]
                )
                avg_gpu_util = sum(measurements["_gpu_util"]) / len(
                    measurements["_gpu_util"]
                )
                avg_memory_util = sum(measurements["_memory_util"]) / len(
                    measurements["_memory_util"]
                )
                total_energy = sum(measurements["_energy"])
                validation_energy_usage += total_energy

                args.score_history["_eval_power"].append(
                    (args.train_steps, avg_milli_watts)
                )
                args.score_history["_eval_memory"].append(
                    (args.train_steps, avg_memory_share)
                )
                args.score_history["_eval_gpu_util"].append(
                    (args.train_steps, avg_gpu_util)
                )
                args.score_history["_eval_memory_util"].append(
                    (args.train_steps, avg_memory_util)
                )
                args.score_history["_eval_energy"].append(
                    (args.train_steps, validation_energy_usage)
                )
                for value in measurements.values():
                    value.clear()

            if not is_same_step:
                args.score_history["_power"].append(
                    (args.train_steps, milli_watt_usage)
                )
                args.score_history["_memory"].append((args.train_steps, used_memory))
                args.score_history["_gpu_util"].append((args.train_steps, gpu_util))
                args.score_history["_memory_util"].append(
                    (args.train_steps, memory_util)
                )
                args.score_history["_energy"].append(
                    (args.train_steps, train_energy_usage)
                )
                pass

            was_inside_eval = False
            train_energy_usage += milli_watt_hours

        train_time += time_diff
        train_energy_usage += milli_watt_hours
        train_power_usage += milli_watt_usage
        train_sample_steps += 1

    queue.put(None)
    queue.put(EnergyStatistics(train_energy_usage, validation_energy_usage, 0, 0, 0, 0))


class Watcher:
    latest: Optional[EnergyStatistics]

    def __init__(self, args: TrainingArgs[THparams, TDataset]):
        self.queue = SimpleQueue()
        self.thread = Thread(
            None,
            watcher,
            "pynvml_recorder",
            args=(
                self.queue,
                args,
            ),
            daemon=True,
        )
        self.thread.start()
        self.latest = None
        self.exhausted = False

    def poll_latest(self) -> Optional[EnergyStatistics]:
        while not self.exhausted:
            try:
                result = self.queue.get_nowait()
                if result is None:
                    self.exhausted = True
                    break
                self.latest = result
            except queue.Empty:
                break
        return self.latest

    def wait_until_finish(self) -> EnergyStatistics:
        while (_ := self.queue.get()) is not None:
            continue
        return self.queue.get()


class watcher_tqdm(tqdm):  # noqa
    def __init__(self, *args, source: Optional[Watcher], **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source

    def update(self, n=1):
        if self.source is not None and (info := self.source.poll_latest()) is not None:
            self.set_postfix(
                OrderedDict(
                    energy=format_to_energy_usage(
                        info.train_energy + info.validation_energy
                    ),
                    power=_format_to_powers_of_1000(
                        info.power, ["mW", "W", "kW", "MW"]
                    ),
                    memory=_format_to_memory_units(info.memory),
                    gpu_util=_format_to_percentage(info.gpu_util),
                    memory_util=_format_to_percentage(info.memory_util),
                ),
                refresh=False,
            )
        super().update(n)
