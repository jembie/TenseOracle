"""
Filters That don't fit neatly into the other big categories
"""
import torch.cuda

from Strategies.filters import FilterStrategy
from collections import defaultdict
from small_text import TransformerBasedClassificationFactory, Dataset, Classifier
import numpy as np
from Utilities.general import SmallTextCartographer
from scipy.special import softmax
from scipy.stats import entropy
from torch import nn
import copy
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class SingleStepEntropy(FilterStrategy):
    '''
    Idea: We allow each chosen sample to choose (for themselves) a label (i.e. PseudoLabel)
    We train model again with each pseudo labeled sample i.e. 1 Step only
    If Entropy increases on unlabeled data we consider the sample HTL
    '''
    def __init__(self, **kwargs):
        self.entrs = []
        self.last_labeled_set_size = 0
        self.predictions_over_time = None

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
        if self.last_labeled_set_size < len(indices_labeled):
            self.last_labeled_set_size = len(indices_labeled)
            probabilities = clf.predict_proba(dataset)
            probabilities = probabilities.reshape((probabilities.shape[0], 1, probabilities.shape[1]))
            if self.predictions_over_time is not None:
                self.predictions_over_time = np.concatenate([self.predictions_over_time, probabilities], axis=1)
            else:
                self.predictions_over_time = probabilities
        if self.predictions_over_time.shape[1] < 5:  # We don't trust first 2 iters & want at least 3 voters
            return np.zeros_like(indices_chosen, dtype=bool)
        pseudo_labels = np.argmax(np.average(self.predictions_over_time[:, -3:], axis=1), axis=1)
        # Create validation set for performance reasons
        validation_indices = np.random.choice(np.arange(len(dataset)), replace=False, size=min(len(dataset), 1000))
        validation_set = dataset[validation_indices].clone()
        validation_set.y = np.zeros_like(validation_indices)
        proba = clf.predict_proba(validation_set)
        entropies = entropy(proba, axis=1)
        entropy_original = np.average(entropies)
        REPETITIONS = 5
        new_entropies = []
        for idx in indices_chosen:
            scores = []
            for r in range(REPETITIONS):
                clf_ = copy.deepcopy(clf)
                clf_.num_epochs = 1
                # Create Train set consisting of only the sample of interest
                sample = dataset[np.array([idx])].clone()
                sample.y = pseudo_labels[np.array([idx])]
                # Train for a single step
                clf_.fit(sample, validation_set=sample)
                # Calc Entropy
                proba = clf_.predict_proba(validation_set)
                entropies = entropy(proba, axis=1)
                scores.append(np.average(entropies))
                del clf_
                torch.cuda.empty_cache()
            new_entropies.append(np.average(scores))
        # TODO Only remove samples that are FAR off
        new_entropies = np.array(new_entropies)
        differences = new_entropies - entropy_original
        mean = np.mean(differences)
        std = np.std(differences)
        htl_mask = differences > max(mean + 1.9 * std, 0)  # new_entropy must be larger than old and if many are larger we only avoid the worst
        return htl_mask