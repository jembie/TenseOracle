"""
Filters That use Dataset Maps to detect and Avoid HTL Samples

Idea:
    We allow each sample to choose for themselves a label
    i.e. the Winning criterium
    (They win if the model can always call the label with enough confidence when it sees the Sample again)
    Most samples that have a clear true label should be able to pull that of.
    However, the HTL ones on the other hand
    those might not have a Real True Label or multiple.
    Therefore, The model will be exceptionally uncertain on them regardless of the label they chose.
    Trivia: Loser short for Losers Can't Cheat, because samples are allowed to choose labels
    i.e. set their own rules (we call this "Cheating"). HTL samples are expected to still Lose
"""
import torch.cuda

from Strategies.filters import FilterStrategy
from collections import defaultdict
from small_text import TransformerBasedClassificationFactory, Dataset, Classifier
import numpy as np
from Utilities.general import SmallTextCartographer
from scipy.special import softmax


def calc_probs(predictions):
    softmax_scores = softmax(predictions, axis=1)
    return softmax_scores


class LoserFilter_SSL_Variety(FilterStrategy):
    """
    A Loser Can't Cheat Filter with the Following Adjustments:

    - Uses Vote Based Pseudo Labeling to extend labeled Train Set for Data set maps
    - Trains 10 Dataset maps on different random subsets of pseudo labeled data and co

    """
    def __init__(self, **kwargs):
        '''
        :param kwargs: requires argument with a TransformerBasedClassificationFactory (kwargs["clf_factory"])
        :return:
        '''
        self.clf_factory: TransformerBasedClassificationFactory = kwargs["clf_factory"]
        self.predictions_over_time = []
        self.already_labeled_htl = defaultdict(int)
        self.last_labeled_set_size = 0
        self.current_iteration = 0

    def __call__(self,
                 indices_chosen: np.ndarray,
                 indices_already_avoided: list,
                 confidence: np.ndarray,
                 clf: Classifier,
                 dataset: Dataset,
                 indices_unlabeled: np.ndarray,
                 indices_labeled: np.ndarray,
                 y: np.ndarray,
                 n=10,
                 iteration=0) -> np.ndarray:
        if self.current_iteration < iteration:
            # Track Predictions over Iteration for Pseudo Labels Later on
            self.current_iteration = iteration
            probabilities = clf.predict_proba(dataset)
            self.predictions_over_time.append(probabilities)

        if len(self.predictions_over_time) < 6:
            # Do Nothing In Early Epochs because not ready
            return np.zeros_like(indices_chosen, dtype=bool)

        # convert tracked Probas to np.array + skip first 2 iters because garbage
        predictions_over_time = np.array(self.predictions_over_time[2:])
        votes = np.argmax(predictions_over_time, axis=2)
        pseudo_labels = [np.argmax(np.bincount(votes[:, i])) for i in indices_chosen]

        # Entire Labelled & Pseudo Labeled dataset for Training
        diverse_labelled_dataset = dataset[np.concatenate([indices_chosen, indices_labeled])].clone()
        diverse_labelled_dataset.y = np.concatenate([pseudo_labels, y], axis=0, dtype=np.int64)

        # Label Samples with the Label that has the most Votes (Mark uncertain samples with -1)
        pseudo_labels_all = np.array([np.argmax(np.bincount(votes[:, i])) if np.max(np.bincount(votes[:, i])) >= np.sum(np.bincount(votes[:, i]))*0.8 else -1 for i in range(votes.shape[1])])
        # Replace Labels for samples that were labeled already by oracle
        pseudo_labels_all[indices_labeled] = y
        # If Chosen Samples were marked as uncertain reset to official Pseudo Label
        pseudo_labels_all[indices_chosen] = pseudo_labels

        # Which Samples weren't marked as uncertain?
        safe_pseudo_indices = np.argwhere(pseudo_labels_all >= 0).flatten()
        # Create Train set of Certain Samples
        extended_labelled_train_set = dataset[safe_pseudo_indices].clone()
        extended_labelled_train_set.y = np.array(pseudo_labels_all[safe_pseudo_indices], dtype=np.int64)

        # Create Cartographer Callback that tracks Chosen Samples and Oracle Labeled Data (To establish baseline)
        cartographer = SmallTextCartographer(dataset=diverse_labelled_dataset, outputs_to_probabilities=calc_probs)
        # Create Placeholder Val Set (We don't care about it)
        val_set = diverse_labelled_dataset[:10].clone()
        val_set.y = np.array(diverse_labelled_dataset.y[:10], dtype=np.int64)

        # Train Model
        for i in range(10):
            # Create model & Add Cartographer
            torch.cuda.empty_cache()
            pseudo_clf = self.clf_factory.new()
            pseudo_clf.num_epochs = 5
            pseudo_clf.callbacks.append(cartographer)
            # Sample Random Subset for Training
            indices_rand = np.random.choice(np.arange(len(extended_labelled_train_set)), replace=False, size=len(extended_labelled_train_set)//4)
            train_set = extended_labelled_train_set[indices_rand].clone()
            train_set.y = np.array(train_set.y, dtype=np.int64)
            # Train CLF
            pseudo_clf.fit(train_set=train_set, validation_set=val_set)

        # Calculate Outlier Threshold as mean - 2 * std
        m = np.mean(cartographer.correctness)
        std = np.std(cartographer.correctness)
        threshold = m - 2*std
        # Which of the chosen Samples are exceptionally hard to predict?
        htl_mask = cartographer.correctness[:len(indices_chosen)] <= threshold

        return htl_mask


class LoserFilter_Plain(FilterStrategy):
    '''
    A Losers can't Cheat Filter (CodeName DSM-LevelUp-Part1)

    - Use no extra train data except chosen samples
    - Weight recent predictions stronger for calculating pseudo labels
    - Doesn't skip first 2 Iters for Pseudo Labeling
    - Delays start by 5 Iters
    '''
    def __init__(self, **kwargs):
        '''
        :param kwargs: requires argument with a TransformerBasedClassificationFactory (kwargs["clf_factory"])
        :return:
        '''
        self.clf_factory: TransformerBasedClassificationFactory = kwargs["clf_factory"]
        self.predictions_over_time = []
        self.already_labeled_htl = defaultdict(int)
        self.last_labeled_set_size = 0
        self.current_iteration = 0

    def __call__(self,
                 indices_chosen: np.ndarray,
                 indices_already_avoided: list,
                 confidence: np.ndarray,
                 clf: Classifier,
                 dataset: Dataset,
                 indices_unlabeled: np.ndarray,
                 indices_labeled: np.ndarray,
                 y: np.ndarray,
                 n=10,
                 iteration=0) -> np.ndarray:
        if self.current_iteration < iteration:
            # Track Predictions over Iteration for Pseudo Labels Later on
            self.current_iteration = iteration
            probabilities = clf.predict_proba(dataset)
            self.predictions_over_time.append(probabilities)
        if len(self.predictions_over_time) < 5:
            # Do Nothing In Early Epochs because not ready
            return np.zeros_like(indices_chosen, dtype=bool)

        # convert to np.array
        predictions_over_time = np.array(self.predictions_over_time)
        votes = np.argmax(predictions_over_time, axis=2)
        pseudo_labels = [np.argmax(np.bincount(votes[:, i], weights=np.arange(1, votes.shape[0]+1))) for i in indices_chosen]


        # Entire Labelled & Pseudo Labeled dataset for Training
        diverse_labelled_dataset = dataset[np.concatenate([indices_chosen, indices_labeled])].clone()
        diverse_labelled_dataset.y = np.concatenate([pseudo_labels, y], axis=0, dtype=np.int64)

        # We also track already labeled data to establish a baseline of what we consider weird
        cartographer = SmallTextCartographer(dataset=diverse_labelled_dataset, outputs_to_probabilities=calc_probs)
        val_set = diverse_labelled_dataset[:10].clone()
        val_set.y = np.array(diverse_labelled_dataset.y[:10], dtype=np.int64)
        # Train Model
        for i in range(10):
            torch.cuda.empty_cache()
            pseudo_clf = self.clf_factory.new()
            pseudo_clf.num_epochs = 5
            pseudo_clf.callbacks.append(cartographer)
            # We don't care about validation set at all, but we need to pass something so the smallest DS we have avail.
            pseudo_clf.fit(train_set=diverse_labelled_dataset, validation_set=val_set)
        # Calculate Outlier Threshold as mean - 2 * std
        m = np.mean(cartographer.correctness)
        std = np.std(cartographer.correctness)
        threshold = m - 2*std
        # Which of the chosen Samples are exceptionally hard to predict?
        htl_mask = cartographer.correctness[:len(indices_chosen)] <= threshold

        return htl_mask
