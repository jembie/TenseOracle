from Utilities.general import set_random_seed
import numpy as np
from small_text import PoolBasedActiveLearner
import copy
from tqdm.auto import tqdm
import deepsig
from sklearn.metrics import f1_score


def evaluate(active_learner, test):
    y_pred_test = active_learner.classifier.predict(test)
    f1 = f1_score(y_pred_test, test.y, average='micro')

    print('Test accuracy(on {} samples): {:.2f}'.format(len(test), f1))
    print('---')
    return f1


def assess_dataset_quality(active_learner: PoolBasedActiveLearner,
                           args,
                           config,
                           train,
                           indices_labeled: np.ndarray,
                           indices_unlabeled: np.ndarray,
                           indices_htl: np.ndarray,
                           test,
                           num_iterations=4,
                           ):
    '''
    Retrains and evaluates model multiple times with the same set because we don't trust
    a single evaluation to represent the quality of a dataset, and therefore we can't trust
    that this represents the quality of the strategy
    '''
    total_budget = config["ITERATIONS"] * config["QUERY_BATCH_SIZE"] + config["SEED_SIZE"]
    unused_budget = total_budget - len(indices_labeled)
    # We assume that budget was only lost due to HTL avoidance nothing else
    assert unused_budget == len(indices_htl)

    print("Start Queried DS Evaluation")
    results = {}

    def replace_htl_with_random(indices_unlabeled):
        random_replacement_for_htl = np.random.choice(indices_unlabeled, unused_budget, replace=False).astype(
            np.int64)
        indices_labeled_backup = np.concatenate((indices_labeled, random_replacement_for_htl), axis=0)
        return indices_labeled_backup

    for experiment_name in ["no_htl", "htl", "random"]:  # , "original"]:
        print(experiment_name)
        if experiment_name == "no_htl":
            # Evaluate without any HTL (i.e. less samples but higher quality we assume)
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
        for i in tqdm(range(config["SET_EVAL_ITERATIONS"])):
            # Bring in diversity by setting diff seed each time
            set_random_seed(args.random_seed + i)
            # Shuffle Dataset each time to get better evaluation
            indices_labeled_ = copy.copy(indices_labeled_backup)  # make copy to not shuffle original
            np.random.shuffle(indices_labeled_)

            y_initial = train.y[indices_labeled_].astype(np.int64)
            active_learner.initialize_data(indices_labeled_, y_initial, retrain=True)
            r = evaluate(active_learner, test)
            results[experiment_name].append(r)

            # Use different replacements for HTL in next test if in "random" mode
            if experiment_name == "random":
                indices_labeled_backup = replace_htl_with_random(indices_unlabeled)

    # Collect all Statistics
    median_no_htl = np.median(np.array(results["no_htl"]))
    median_with_htl = np.median(np.array(results["htl"]))
    median_replacement = np.median(np.array(results["random"]))

    final_results = {
        "avgF1 (No HTL)": sum(results["no_htl"]) / len(results["no_htl"]),
        "avgF1 (With HTL)": sum(results["htl"]) / len(results["htl"]),
        "avgF1 (random replacement)": sum(results["random"]) / len(results["random"]),
        "medF1 (No HTL)": median_no_htl,
        "medF1 (With HTL)": median_with_htl,
        "medF1 (random replacement)": median_replacement,
        "HTL Count": len(indices_htl),
        "ASO-Sig[1]": deepsig.aso(results["no_htl"], results["htl"], seed=args.random_seed),
        "ASO-Sig[2]": deepsig.aso(results["random"], results["htl"], seed=args.random_seed),
        "HTL_harms_median": median_no_htl - median_with_htl,
        "HTL_low_val_median": median_replacement - median_with_htl
    }

    # TODO Commit all results to Comet for later in depth eval

    return final_results
