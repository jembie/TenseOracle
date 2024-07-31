from Utilities.general import set_random_seed
from Utilities.comet import CometExperiment
import numpy as np
from small_text import PoolBasedActiveLearner
import copy
from tqdm.auto import tqdm
import deepsig
from sklearn.metrics import f1_score
from typing import Dict, List, DefaultDict
from collections import defaultdict


def evaluate(active_learner, test):
    y_pred_test = active_learner.classifier.predict(test)
    f1 = f1_score(y_pred_test, test.y, average="micro")

    print("Test accuracy(on {} samples): {:.2f}".format(len(test), f1))
    print("---")
    return f1

def replace_htl_with_random(
        indices_labeled,
        indices_unlabeled,
        htl_mask_indices,
        unused_budget
):
    # TODO remove HTL samples
    indices_labeled_no_htl = np.setdiff1d(indices_labeled, htl_mask_indices)

    random_replacement_for_htl = np.random.choice(
        indices_unlabeled, unused_budget, replace=False
    ).astype(np.int64)
    indices_labeled_backup = np.concatenate(
        (indices_labeled_no_htl, random_replacement_for_htl), axis=0, dtype=np.int64
    )
    assert len(indices_labeled_backup) == len(indices_labeled)
    return indices_labeled_backup

def evaluate_dataset_version(
        iterations: int,
        random_seed: int,
        train : np.ndarray,
        active_learner: PoolBasedActiveLearner,
        results: Dict[str, Dict[str, List]],
        strategy: str,
        experiment_name: str,
        test: np.ndarray,
        indices_labeled_backup,
        indices_labeled,
        indices_unlabeled,
        htl_mask_indices,
) -> Dict[str, Dict[str, List]]:
    # Use different replacements for HTL in next test if in "random" mode

    for i in tqdm(range(iterations)):
        # Bring in diversity by setting diff seed each time
        set_random_seed(random_seed + i)
        # Shuffle Dataset each time to get better evaluation
        indices_labeled_ = copy.copy(indices_labeled_backup).astype(
            np.int64
        )  # make copy to not shuffle original
        np.random.shuffle(indices_labeled_)

        y_initial = train.y[indices_labeled_].astype(np.int64)
        active_learner.initialize_data(indices_labeled_, y_initial, retrain=True)
        f1_score = evaluate(active_learner, test)
        results[strategy][experiment_name].append(f1_score)  # TODO: CHANGE IT TO 1D DICT
        # We replace HTL Samples with new random samples to better gauge the mean value of random samples for AL
        if experiment_name == "random": # TODO: CHECK
            indices_labeled_backup = replace_htl_with_random(indices_labeled=indices_labeled, indices_unlabeled=indices_unlabeled, htl_mask_indices=htl_mask_indices,unused_budget=len(htl_mask_indices))


    return results

def compare_datasets(
    active_learner,
    train,
    test,
    indices_labeled,
    indices_htl,
    iterations,
    random_seed,
) -> Dict[str, Dict[str, List]]:
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
    
    results = defaultdict(lambda: defaultdict(list))
    for experiment_name in ["no_htl", "htl", "random"]:
        print(experiment_name)

        if experiment_name == "no_htl":
            for strategy, htl_mask_indices in indices_htl.items():
                indices_unlabeled = np.setdiff1d(np.arange(len(train.y)), indices_labeled)

                print(f"In compare datasets, with Strategy: {strategy} for \n {indices_htl}")

                # Remove THTLs to  Evaluate without any of the marked HTL (i.e. fewer samples but higher quality we assume)
                indices_labeled_backup = np.setdiff1d(indices_labeled, htl_mask_indices)
                assert len(indices_labeled_backup) + len(htl_mask_indices) == len(indices_labeled)
                results = evaluate_dataset_version(
                    iterations=iterations,
                    random_seed=random_seed,
                    train=train,
                    active_learner=active_learner,
                    results=results,
                    strategy=strategy,
                    experiment_name=experiment_name,
                    test=test,
                    indices_labeled_backup=indices_labeled_backup,
                    indices_labeled=indices_labeled,
                    indices_unlabeled=indices_unlabeled,
                    htl_mask_indices=htl_mask_indices
                )

        elif experiment_name == "random":
            # Low Bar: Evaluate with random replacement for HTL
            # We Assume: Same size as with HTL but higher quality
            # Random Replacement is done after each iteration as well
            for strategy, htl_mask_indices in indices_htl.items():
                indices_unlabeled = np.setdiff1d(np.arange(len(train.y)), indices_labeled)

                print(f"In compare datasets, with Strategy: {strategy} for \n {indices_htl}")

                # Remove THTLs to  Evaluate without any of the marked HTL (i.e. fewer samples but higher quality we assume)
                indices_labeled_backup = np.setdiff1d(indices_labeled, htl_mask_indices)
                assert len(indices_labeled_backup) + len(htl_mask_indices) == len(indices_labeled)
                results = evaluate_dataset_version(
                    iterations=iterations,
                    random_seed=random_seed,
                    train=train,
                    active_learner=active_learner,
                    results=results,
                    strategy=strategy,
                    experiment_name=experiment_name,
                    test=test,
                    indices_labeled_backup=indices_labeled_backup,
                    indices_labeled=indices_labeled,
                    indices_unlabeled=indices_unlabeled,
                    htl_mask_indices=htl_mask_indices
                )

        elif experiment_name == "htl":
            # The dataset as requested if Filter not active
            # We assume: Worse due to HTL samples
            indices_labeled_backup = copy.deepcopy(indices_labeled)
            indices_unlabeled = np.setdiff1d(np.arange(len(train.y)), indices_labeled)

            results = evaluate_dataset_version(
                iterations=iterations,
                random_seed=random_seed,
                train=train,
                active_learner=active_learner,
                results=results,
                strategy="htl",
                experiment_name=experiment_name,
                test=test,
                indices_labeled_backup=indices_labeled_backup,
                indices_labeled=indices_labeled,
                indices_unlabeled=indices_unlabeled,
                htl_mask_indices=[]
            )
        else:
            raise NotImplementedError(
                f"Experiment with name {experiment_name} is not yet implemented"
            )
    return results


def assess_dataset_quality(
    active_learner: PoolBasedActiveLearner,
    args,
    config,
    train,
    indices_labeled: np.ndarray,
    indices_htl: np.ndarray,
    test,
    experiment,
):
    """
    Retrains and evaluates model multiple times with the same set because we don't trust
    a single evaluation to represent the quality of a dataset, and therefore we can't trust
    that this represents the quality of the strategy
    """
    # total_budget = (
    #     config["ITERATIONS"] * config["QUERY_BATCH_SIZE"] + config["SEED_SIZE"]
    # )
    # unused_budget = total_budget - len(indices_labeled)
    # We assume that budget was only lost due to HTL avoidance nothing else
    # assert unused_budget == len(indices_htl) or (
    #     args.use_up_entire_budget and unused_budget == 0
    # )

    print("Start Queried DS Evaluation")
    results = compare_datasets(
        active_learner=active_learner,
        train=train,
        test=test,
        indices_labeled=indices_labeled,
        indices_htl=indices_htl,
        iterations=config["SET_EVAL_ITERATIONS"],
        random_seed=args.random_seed,
    )
    experiment.log_results(results["htl"]["htl"], "HTL")

    final_results = {}
    for strategy, experiments in results.items():
        if strategy == "htl":
            continue
        for outlier_experiment in experiments:
            experiment.log_results(results[strategy][outlier_experiment], f"{strategy}_{outlier_experiment}")

        # Collect all Statistics
        median_no_htl = np.median(np.array(results[strategy]["no_htl"]))
        median_with_htl = np.median(np.array(results["htl"]["htl"]))
        median_replacement = np.median(np.array(results[strategy]["random"]))

        tmp = {
                "avgF1 (No HTL)": sum(results[strategy]["no_htl"]) / len(results[strategy]["no_htl"]),
                "avgF1 (With HTL)": sum(results["htl"]["htl"]) / len(results["htl"]["htl"]),
                "avgF1 (random replacement)": sum(results[strategy]["random"]) / len(results[strategy]["random"]),
                "medF1 (No HTL)": median_no_htl,
                "medF1 (With HTL)": median_with_htl,
                "medF1 (random replacement)": median_replacement,
                "HTL Count": len(indices_htl[strategy]),
                "ASO-Sig[1]": deepsig.aso(
                    results[strategy]["no_htl"], results["htl"]["htl"], seed=args.random_seed
                ),
                "ASO-Sig[2]": deepsig.aso(
                    results[strategy]["random"], results["htl"]["htl"], seed=args.random_seed
                ),
                "HTL_harms_median": median_no_htl - median_with_htl,
                "HTL_low_val_median": median_replacement - median_with_htl,
        }

        final_results[strategy] = tmp

    # TODO Commit all results to Comet for later in depth eval

    return final_results


def domination_test(
    active_learner: PoolBasedActiveLearner,
    train,
    test,
    config,
    args,
    step,
    experiment: CometExperiment,
):
    indices_labeled = active_learner.indices_labeled
    indices_htl = active_learner.query_strategy.indices_htl
    indices_unlabeled = np.setdiff1d(
        np.arange(len(train)), np.concatenate((indices_htl, indices_labeled))
    )
    results = compare_datasets(
        active_learner=active_learner,
        train=train,
        test=test,
        indices_labeled=indices_labeled,
        indices_unlabeled=indices_unlabeled,
        indices_htl=indices_htl,
        iterations=config["DOM_EVAL_ITERATIONS"],
        random_seed=args.random_seed,
    )

    metrics_to_log = {}
    for experiment_name in results.keys():
        metrics_to_log[f"{experiment_name}_d"] = np.median(results[experiment_name])

    experiment.log_metrics(metrics_to_log, step)

    return metrics_to_log
