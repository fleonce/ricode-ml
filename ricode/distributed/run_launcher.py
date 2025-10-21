import os
import subprocess
import sys

from with_argparse import with_argparse
from with_argparse.configure_argparse import WithArgparse


def _on_help(args: WithArgparse):
    args.argparse.print_help()


def _screen_name(pattern, i):
    return pattern.format(i)


def _dispatch_command(session_name, cmd):
    proc = subprocess.run(
        ["/usr/bin/screen", "-S", session_name, "-X", "stuff", cmd + "\n"]
    )
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


@with_argparse(partial_parse=True, on_help=_on_help)
def launch(
    train_script: str = "train.py",
    gpus: str = "1",
    screen_pattern: str = "gpu{}",
    master_address: str = "localhost",
    master_port: str = "12345",
    # internal argument:
    _help: bool = False,
):
    train_path = os.path.join(os.getcwd(), train_script)
    if os.path.exists(train_path):
        command_to_run = [sys.executable, train_path, *sys.argv[1:]]

        if _help:
            process = subprocess.run(
                command_to_run,
                shell=False,
                cwd=os.getcwd(),
                capture_output=False,
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            return process.returncode
        else:
            device_ids = _parse_gpus(gpus)
            world_size = len(device_ids)
            is_distributed = len(device_ids) > 1

            # prepare environment variables
            # prepare screens (?)
            # calc. gpu available vs. gpu desired
            # setup distributed etc.
            for rank, device_id in enumerate(device_ids):
                device_screen_name = _screen_name(screen_pattern, device_id + 1)

                process: subprocess.CompletedProcess = subprocess.run(
                    ["/usr/bin/screen", "-dmS", device_screen_name, "bash"]
                )
                assert (
                    process.returncode == 0
                ), f"Could not create screen for {device_screen_name}"

                _dispatch_command(device_screen_name, "export TMOUT=0")
                if is_distributed:
                    _dispatch_command(
                        device_screen_name,
                        f"export WORLD_SIZE={world_size} RANK={rank} CUDA_LOCAL_DEVICE={device_id}",
                    )
                    _dispatch_command(
                        device_screen_name,
                        f"export MASTER_ADDR={master_address} MASTER_PORT={master_port}",
                    )
                else:
                    _dispatch_command(
                        device_screen_name, f"export CUDA_VISIBLE_DEVICES={device_id}"
                    )
                _dispatch_command(device_screen_name, "cd " + os.getcwd())
                _dispatch_command(device_screen_name, " ".join(command_to_run))

    else:
        print("Cannot find", train_path, "aborting ...", file=sys.stderr)
        return 1
