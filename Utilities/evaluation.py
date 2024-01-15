import small_text
from Utilities.general import set_random_seed
from Utilities.comet import CometExperiment
import numpy as np
from small_text import PoolBasedActiveLearner
import copy
from tqdm.auto import tqdm
import deepsig
from sklearn.metrics import f1_score
from Utilities.general import SmallTextCartographer
from Strategies.dsm_filters import calc_probs


def evaluate(active_learner, test):
    y_pred_test = active_learner.classifier.predict(test)
    f1 = f1_score(y_pred_test, test.y, average='micro')

    print('Test accuracy(on {} samples): {:.2f}'.format(len(test), f1))
    print('---')
    return f1


def compare_datasets(active_learner,
                     train,
                     test,
                     indices_labeled,
                     indices_unlabeled,
                     indices_htl,
                     iterations,
                     random_seed,
                     ) -> dict:
    """
    Combines Labeled, found HTL, and unlabeled data into 3 variants.
    Labeled Data, Labeled Data with HTL Samples,
    Labeled Data with randomly chosen replacements for the HTL Samples.
    Goal: Finding out whether the HTL samples indeed decrease Performance
    They hurt performance if the Labeled data performs worse than the labeled data + HTL dataset
    Low Bar: HTL samples are low value i.e. Labeled Data + Random Replacements
    leads to better performance than Labeled Data + HTL Samples
    :param active_learner:
    :param train:
    :param test:
    :param indices_labeled:
    :param indices_unlabeled:
    :param indices_htl:
    :param iterations:
    :param random_seed:
    :return:
    """
    results = {}
    unused_budget = len(indices_htl)

    def replace_htl_with_random(indices_unlabeled):
        random_replacement_for_htl = np.random.choice(indices_unlabeled, unused_budget, replace=False).astype(np.int64)
        indices_labeled_backup = np.concatenate((indices_labeled, random_replacement_for_htl), axis=0, dtype=np.int64)
        return indices_labeled_backup

    for experiment_name in ["no_htl", "htl", "random"]:
        print(experiment_name)
        if experiment_name == "no_htl":
            # Evaluate without any HTL (i.e. fewer samples but higher quality we assume)
            indices_labeled_backup = copy.deepcopy(indices_labeled)
        elif experiment_name == "random":
            # Low Bar: Evaluate with random replacement for HTL
            # We Assume: Same size as with HTL but higher quality
            # Random Replacement is done after each iteration as well
            indices_labeled_backup = replace_htl_with_random(indices_unlabeled)
        elif experiment_name == "htl":
            # The dataset as requested if Filter not active (probably)
            # We assume: Worse due to HTL samples
            indices_labeled_backup = np.concatenate((indices_labeled, indices_htl), axis=0)
        else:
            raise NotImplementedError(f"Experiment with name {experiment_name} is not yet implemented")

        results[experiment_name] = []
        for i in tqdm(range(iterations)):
            # Bring in diversity by setting diff seed each time
            set_random_seed(random_seed + i)
            # Shuffle Dataset each time to get better evaluation
            indices_labeled_ = copy.copy(indices_labeled_backup).astype(np.int64)  # make copy to not shuffle original
            np.random.shuffle(indices_labeled_)

            y_initial = train.y[indices_labeled_].astype(np.int64)
            active_learner.initialize_data(indices_labeled_, y_initial, retrain=True)
            r = evaluate(active_learner, test)
            results[experiment_name].append(r)

            # Use different replacements for HTL in next test if in "random" mode
            if experiment_name == "random":
                indices_labeled_backup = replace_htl_with_random(indices_unlabeled)

    return results


def evaluate_dataset(active_learner: small_text.PoolBasedActiveLearner,
                     train,
                     test,
                     indices_labeled,
                     indices_unlabeled,
                     indices_htl,
                     iterations,
                     random_seed,
                     ) -> (list, SmallTextCartographer):
    """
    Combines Labeled, found HTL, and unlabeled data into 3 variants.
    Labeled Data, Labeled Data with HTL Samples,
    Labeled Data with randomly chosen replacements for the HTL Samples.
    Goal: Finding out whether the HTL samples indeed decrease Performance
    They hurt performance if the Labeled data performs worse than the labeled data + HTL dataset
    Low Bar: HTL samples are low value i.e. Labeled Data + Random Replacements
    leads to better performance than Labeled Data + HTL Samples
    :param active_learner:
    :param train:
    :param test:
    :param indices_labeled:
    :param indices_unlabeled:
    :param indices_htl:
    :param iterations:
    :param random_seed:
    :return:
    """
    results = {}

    indices_labeled_backup = copy.deepcopy(indices_labeled)

    results = []
    ds = train[indices_labeled_backup].clone()
    ds.y = train.y[indices_labeled_backup].astype(np.int64)
    cartographer = SmallTextCartographer(dataset=ds, outputs_to_probabilities=calc_probs)
    for i in tqdm(range(iterations)):
        # Bring in diversity by setting diff seed each time
        set_random_seed(random_seed + i)
        # Shuffle Dataset each time to get better evaluation
        indices_labeled_ = copy.copy(indices_labeled_backup).astype(np.int64)  # make copy to not shuffle original
        np.random.shuffle(indices_labeled_)

        y_initial = train.y[indices_labeled_].astype(np.int64)
        active_learner.initialize_data(indices_labeled_, y_initial, retrain=True, callback=cartographer)
        r = evaluate(active_learner, test)
        results.append(r)

    return results, cartographer


def assess_dataset_quality(active_learner: PoolBasedActiveLearner,
                           args,
                           config,
                           train,
                           indices_labeled: np.ndarray,
                           indices_unlabeled: np.ndarray,
                           indices_htl: np.ndarray,
                           test,
                           experiment
                           ):
    '''
    Retrains and evaluates model multiple times with the same set because we don't trust
    a single evaluation to represent the quality of a dataset, and therefore we can't trust
    that this represents the quality of the strategy
    '''
    total_budget = config["ITERATIONS"] * config["QUERY_BATCH_SIZE"] + config["SEED_SIZE"]
    unused_budget = total_budget - len(indices_labeled)
    # We assume that budget was only lost due to HTL avoidance nothing else
    assert unused_budget == 0

    print("Start Queried DS Evaluation")
    (results, cartographer) = evaluate_dataset(active_learner=active_learner,
                               train=train,
                               test=test,
                               indices_labeled=indices_labeled,
                               indices_unlabeled=indices_unlabeled,
                               indices_htl=indices_htl,
                               iterations=config["SET_EVAL_ITERATIONS"],
                               random_seed=args.random_seed)


    experiment.log_results(results, "f1s")

    cartographer: SmallTextCartographer = cartographer
    experiment.log_results(cartographer.correctness, "correctness")
    experiment.log_results(cartographer.confidence, "confidence")
    experiment.log_results(cartographer.variability, "variability")
    experiment.log_results(cartographer.gold_labels_probabilities, "gold_probs")


    # Collect all Statistics
    mean = np.mean(np.array(results))

    final_results = {
        "mean_f1": mean,
        "Avoided Samples Count": len(indices_htl),
        "mean correctness": np.mean(cartographer.correctness),
        "mean confidence": np.mean(cartographer.confidence),
        "mean variability": np.mean(cartographer.variability),
        "THTL Count": np.sum(cartographer.correctness < 0.3),
    }

    # TODO Commit all results to Comet for later in depth eval

    return final_results


def domination_test(active_learner: PoolBasedActiveLearner,
                   train,
                   test,
                   config,
                   args,
                   step,
                   experiment: CometExperiment):
    indices_labeled = active_learner.indices_labeled
    indices_htl = active_learner.query_strategy.indices_htl
    indices_unlabeled = np.setdiff1d(np.arange(len(train)), np.concatenate((indices_htl, indices_labeled)))
    results = compare_datasets(active_learner=active_learner,
                               train=train,
                               test=test,
                               indices_labeled=indices_labeled,
                               indices_unlabeled=indices_unlabeled,
                               indices_htl=indices_htl,
                               iterations=config["DOM_EVAL_ITERATIONS"],
                               random_seed=args.random_seed)

    metrics_to_log = {}
    for experiment_name in results.keys():
        metrics_to_log[f"{experiment_name}_d"] = np.median(results[experiment_name])

    experiment.log_metrics(metrics_to_log, step)

    return metrics_to_log
