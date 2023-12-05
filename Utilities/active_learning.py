import torch
from small_text import TransformerBasedClassificationFactory, TransformerModelArguments
from Strategies import acquisition_functions, filters
from small_text import QueryStrategy, PoolBasedActiveLearner, random_initialization_balanced
import numpy as np
import gc


def load_model(config: dict, num_classes):
    if torch.cuda.is_available():
        kwargs = {'device': 'cuda:0'}
    else:
        kwargs = {}
    kwargs["cache_dir"] = config["SHARED_CACHE_ADR"]
    kwargs["validation_set_size"] = 0.1
    kwargs["mini_batch_size"] = config["TRAIN_BATCH_SIZE"]
    kwargs["num_epochs"] = config["NUM_EPOCHS"]
    kwargs["validations_per_epoch"] = 1
    kwargs["lr"] = config["LR"]
    clf_factory = TransformerBasedClassificationFactory(
        TransformerModelArguments(config["MODEL_NAME"]),
        num_classes,
        kwargs=kwargs
    )
    return clf_factory


class HTLOverseer(QueryStrategy):
    """
    A Wrapper for a QueryStrategy that simply uses A certain FilterStrategy
    to keep the QueryStrategy from Sampling Stupid samples from which
    the Overseer doesn't think they will help
    It tracks those samples and at the end we'll compare performance
    with and without those stupid samples to find out whether our FilterStrategy Worked.
    """

    def __init__(self, filter_strategy: filters.FilterStrategy, query_strategy: QueryStrategy):
        super().__init__()
        self.filter_strategy = filter_strategy
        self.query_strategy = query_strategy
        self.htl_tracker = []  # Here is where I'd put my HTL samples if I had any

    def query(self, clf, _dataset, indices_unlabeled, indices_labeled, y, n=10, ssl_step=False):
        unlabeled_pool = np.setdiff1d(indices_unlabeled, np.array(self.htl_tracker))
        chosen_samples, confidence = self.query_strategy.query(clf, _dataset, unlabeled_pool, indices_labeled, y, n=n)
        if not self.filter_strategy:
            # If no Filter Strategy in use just return samples as is
            return chosen_samples
        htl_mask = self.filter_strategy(indices_chosen=chosen_samples,
                                        confidence=confidence,
                                        indices_already_avoided=self.htl_tracker,
                                        clf=clf,
                                        dataset=_dataset,
                                        indices_unlabeled=indices_unlabeled,
                                        indices_labeled=indices_labeled,
                                        y=y,
                                        n=n)

        # Add HTL samples to HTL tracker
        self.htl_tracker += list(chosen_samples[htl_mask])

        return chosen_samples[~htl_mask]

    def __repr__(self):
        return f"HTLOverseer({str(self.filter_strategy)}, {str(self.query_strategy)})"

    @property
    def indices_htl(self) -> np.ndarray:
        return np.array(list(set(self.htl_tracker)))


def load_query_strategy(strategy_name, filter_name, config, num_classes) -> HTLOverseer:
    """
    Loads a QueryStrategy (also called AcquisitionFunction)
    and Wraps a filter around it that is supposed to
    double-check the requested samples for usability
    :param strategy_name: Name of the actual AcquisitionFunction
    :param filter_name: The name of the filter that shall be used
    :param config:
    :param num_classes:
    :return:
    """
    query_strategy = getattr(acquisition_functions, strategy_name)()

    kwargs = {
        "clf_factory": load_model(config, num_classes),
    }
    if filter_name != "None":
        filter_strategy = getattr(filters, filter_name)(**kwargs)
    else:
        filter_strategy = None
    query_strategy = HTLOverseer(filter_strategy=filter_strategy, query_strategy=query_strategy)

    return query_strategy


def load_active_learner(clf_factory, query_strategy, train):
    """A Method to load different Active Learners with different Seed Set / Training strategies"""
    return PoolBasedActiveLearner(clf_factory, query_strategy, train)


def initialize_active_learner(active_learner, y_train, config):
    """
    Select SEED_SIZE many samples at random, label those
    and at them to the labeled dataset to initialize the AL algorithm
    :param active_learner:
    :param y_train:
    :param config:
    :return:
    """
    # Select Samples
    indices_initial = random_initialization_balanced(y_train, n_samples=config['SEED_SIZE'])
    # Imagine Human Expert Providing Label here
    y_initial = y_train[indices_initial]
    # Initialize Learner
    active_learner.initialize_data(indices_initial, y_initial.astype(np.int64))

    return indices_initial


def perform_active_learning(active_learner: PoolBasedActiveLearner,
                            config,
                            indices_labeled,
                            train,
                            ):
    for i in range(config["ITERATIONS"]):
        # Clear Memory
        gc.collect()
        torch.cuda.empty_cache()

        indices_queried = active_learner.query(num_samples=config["QUERY_BATCH_SIZE"])

        # Simulate Oracle
        y = train.y[indices_queried]

        # Pass labels to active learner and retrain model
        active_learner.update(y)

        indices_labeled = np.concatenate([indices_labeled, indices_queried])

        # When evaluating we ignore Pseudo-labeled data because we don't know whether it is true
        print('Iteration #{:d} ({} train samples)'.format(i, len(active_learner.indices_labeled)))

    return indices_labeled
