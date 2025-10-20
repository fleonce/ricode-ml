import os
import sys

from with_argparse import with_argparse


@with_argparse(partial_parse=True)
def launch(train_script: str = "train.py"):
    train_path = os.path.join(os.getcwd(), train_script)
    if os.path.exists(train_path):
        # prepare environment variables
        # prepare screens (?)
        # calc. gpu available vs. gpu desired
        # setup distributed etc.
        print("Found training script", train_path)
    else:
        print("Cannot find", train_path, "aborting ...", file=sys.stderr)
        exit(1)
