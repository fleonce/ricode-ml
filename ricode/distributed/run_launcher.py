import functools
import os
import subprocess
import sys
from typing import Mapping, Sequence

from with_argparse import with_argparse
from with_argparse.configure_argparse import WithArgparse


def _on_help(args: WithArgparse):
    args.argparse.print_help()


def _screen_name(pattern, i):
    return pattern.format(i)


def _dispatch_command(screen_bin, session_name, cmd):
    proc = subprocess.run([screen_bin, "-S", session_name, "-X", "stuff", cmd + "\n"])
    assert proc.returncode == 0, f"Could not dispatch command '{cmd}' to {session_name}"


def _parse_gpus(gpus: str) -> list[int]:
    gpu_ids: list[int] = []
    for part in gpus.split(","):
        if "-" in part:
            start, end = part.split("-")
            gpu_ids.extend(range(int(start), int(end) + 1))
        else:
            gpu_ids.append(int(part))
    return list(sorted(map(lambda x: x - 1, gpu_ids)))


def _check_screen_exists():
    proc = subprocess.run("which screen".split(), capture_output=True)
    if proc.returncode == 0:
        return proc.stdout.decode("utf-8").strip()
    return None


def _check_screens_exist(
    screen_bin, screen_pattern: str, device_ids: Sequence[int]
) -> Sequence[str]:
    existing = []
    for rank, device_id in enumerate(device_ids):
        device_screen_name = _screen_name(screen_pattern, device_id + 1)
        screen_list = subprocess.run(f"{screen_bin} ls {device_screen_name}".split())
        if screen_list.returncode != 0:
            existing.append(device_screen_name)
    return existing


def _run_proc(args: Sequence[str], env: Mapping[str, str] | None = None):
    return subprocess.run(
        args,
        shell=False,
        cwd=os.getcwd(),
        capture_output=False,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )


@with_argparse(
    partial_parse=True, partial_parse_pass_remaining_args=True, on_help=_on_help
)
def launch(
    train_script: str = "train.py",
    gpus: str = "1",
    screen_pattern: str = "gpu{}",
    master_address: str = "localhost",
    master_port: str = "12345",
    # internal argument:
    _help: bool = False,
    _args: list[str] = None,
):
    screen_bin = _check_screen_exists()

    train_path = os.path.join(os.getcwd(), train_script)
    if os.path.exists(train_path):
        command_to_run = [sys.executable, train_path, *_args]

        if _help:
            command_to_run.append("-h")
            process = _run_proc(command_to_run)
            return process.returncode
        else:
            if screen_bin is None:
                print("screen is unavailable, falling back to single-gpu mode")
                gpus = "1"

            device_ids = _parse_gpus(gpus)
            world_size = len(device_ids)
            is_distributed = len(device_ids) > 1

            if not is_distributed:
                os.environ["RANK"] = "0"
                os.environ["WORLD_SIZE"] = "1"
                return _run_proc(command_to_run).returncode
            else:
                existing_screens = _check_screens_exist(
                    screen_bin, screen_pattern, device_ids
                )
                if len(existing_screens) > 0:
                    print(
                        "The following screens already exist, "
                        "change the pattern or remove the screens: ",
                        *list(map(repr, existing_screens)),
                    )
                    return 1

                import torch.cuda

                if len(device_ids) > torch.cuda.device_count():
                    print(
                        f"Using more gpus ({len(device_ids)}) than available "
                        f"on this system ({torch.cuda.device_count()})"
                    )
                    return 1

                for rank, device_id in enumerate(device_ids):
                    device_screen_name = _screen_name(screen_pattern, device_id + 1)

                    dispatch = functools.partial(
                        _dispatch_command,
                        screen_bin,
                        device_screen_name,
                    )

                    process: subprocess.CompletedProcess = subprocess.run(
                        [screen_bin, "-dmS", device_screen_name, "bash"]
                    )
                    assert (
                        process.returncode == 0
                    ), f"Could not create screen for {device_screen_name}"

                    dispatch("export TMOUT=0")
                    if is_distributed:
                        dispatch(
                            f"export WORLD_SIZE={world_size} RANK={rank} CUDA_LOCAL_DEVICE={device_id}"
                        )
                        dispatch(
                            f"export MASTER_ADDR={master_address} MASTER_PORT={master_port}"
                        )
                    else:
                        dispatch(f"export CUDA_VISIBLE_DEVICES={device_id}")
                    dispatch("cd " + os.getcwd())
                    dispatch(" ".join(command_to_run))
                return 0
    else:
        print("Cannot find", train_path, "aborting ...", file=sys.stderr)
        return 1
