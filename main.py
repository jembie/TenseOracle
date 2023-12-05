import os
import sys

from Utilities.parsers import parse_config, parse_args, parse_task_config
from Utilities.comet import CometExperiment
from Utilities.preprocessing import load_dataset_from_config
import copy
import torch

def main():
    (train, test) = load_dataset_from_config(task_config, config=config, args=args)
    return


if __name__ == '__main__':
    args = parse_args()
    config = parse_config(args)
    task_config = parse_task_config(args)
    experiment = CometExperiment(args, config, task_config)

    # Some Back-Ups to make sure that config values aren't changed
    args_backup = copy.deepcopy(args)
    config_backup = copy.deepcopy(config)
    task_config_backup = copy.deepcopy(task_config)

    if torch.cuda.is_available():
        cuda = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        print(f"cuda available, using one of those: {cuda}")
    else:
        if not args.gpu_optional:
            raise Exception("No GPU Found, If no GPU required please set --gpu_optional")

    main()
    sys.exit(0)
