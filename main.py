import os
import sys

from Utilities.parsers import parse_config, parse_args, parse_task_config
from Utilities.comet import CometExperiment
from Utilities.preprocessing import load_dataset_from_config
from Utilities.active_learning import load_model, load_query_strategy, load_active_learner, initialize_active_learner
from Utilities.active_learning import perform_active_learning
import copy
import torch

def main():
    (train, test) = load_dataset_from_config(task_config, config=config, args=args)

    num_classes = len(set(train.y).union(test.y))
    clf_factory = load_model(config, num_classes)

    query_strategy = load_query_strategy(strategy_name=args.strategy_name,
                                         filter_name=args.filter_strategy_name,
                                         config=config,
                                         num_classes=num_classes)

    # Init Learner & Seed Set
    active_learner = load_active_learner(clf_factory, query_strategy, train)
    indices_labeled = initialize_active_learner(active_learner, train.y, config)

    indices_labeled = perform_active_learning(
        active_learner=active_learner,
        train=train,
        indices_labeled=indices_labeled,
        config=config
    )

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
