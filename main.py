import sys
import time

from Utilities.parsers import parse_config, parse_args, parse_task_config
from Utilities.comet import CometExperiment
from Utilities.preprocessing import load_dataset_from_config
from Utilities.active_learning import (
    load_model,
    load_query_strategy,
    load_active_learner,
    initialize_active_learner,
)
from Utilities.active_learning import perform_active_learning
from Utilities.evaluation import assess_dataset_quality
import copy
import torch
import numpy as np
from Utilities.general import set_random_seed, log_failed_attempts
import pprint


def main():
    set_random_seed(args.random_seed)
    (train, test) = load_dataset_from_config(task_config, config=config, args=args)

    num_classes = len(set(train.y).union(test.y))
    # Load Factory for models such that a new model can be trained each iteration
    clf_factory = load_model(config, num_classes)

    # Load Acquisition Function and Wrap Filter around
    query_strategy = load_query_strategy(
        strategy_name=args.strategy_name,
        filter_name=args.filter_strategy_name,
        config=config,
        args=args,
        num_classes=num_classes,
    )

    # Init Learner & Seed Set
    active_learner = load_active_learner(clf_factory, query_strategy, train)
    indices_labeled = initialize_active_learner(active_learner, train.y, config)

    indices_labeled = perform_active_learning(
        config=config,
        args=args,
        active_learner=active_learner,
        train=train,
        test=test,
        indices_labeled=indices_labeled,
        experiment=experiment,
    )
    # Extract all Identified Samples from Filter
    indices_htl = (
        active_learner.query_strategy.indices_htl
    )  # htl - Hard To Learn (i.e. Outlier)

    filter_strategies = experiment.filter_strategy_name.split()
    pprint.pprint(indices_htl)

    for filter_strategy in filter_strategies:

        indices_used = np.concatenate((indices_labeled, indices_htl[filter_strategy]), axis=0)
        indices_unused = np.setdiff1d(np.arange(len(train.y)), indices_used)

        # Evaluate whether avoided samples indeed hurt performance
        set_performance = assess_dataset_quality(
            active_learner=active_learner,
            train=train,
            indices_labeled=indices_labeled,
            indices_unlabeled=indices_unused,
            indices_htl=indices_htl,
            test=test,
            config=config,
            args=args,
            experiment=experiment,
        )

        # Log Results to Comet
        metrics_to_log = {
            "avg_duration": sum(tt := active_learner.query_strategy.time_tracker) / len(tt),
            **set_performance,
        }
        experiment.log_metrics(metrics_to_log)
        experiment.log_results(
            np.array(active_learner.query_strategy.time_tracker), "durations"
        )

    return


if __name__ == "__main__":
    # Read Arguments
    args = parse_args()
    config = parse_config(args)
    task_config = parse_task_config(args)
    # Setup Comet if available to logg results
    experiment = CometExperiment(args, config, task_config)

    # Check for GPU - Crash if not available (except if --gpu_optional)
    if torch.cuda.is_available():
        experiment.log_parameters({"GPU": True})
        cuda = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        print(f"cuda available, using one of those: {cuda}")
    else:
        experiment.log_parameters({"GPU": False})
        if not args.gpu_optional:
            print(torch.version.cuda)
            if torch.cuda.device_count() == 0:
                log_failed_attempts(
                    args.random_seed,
                    args.task_config,
                    args.filter_strategy_name,
                    config["SHARED_CACHE_ADR"],
                )
                raise Exception(
                    "No GPU Found, If none required please set --gpu_optional"
                )

    main()
    sys.exit(0)
