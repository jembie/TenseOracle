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
    Idea: We allow each sample to choose for themselves a label
    We train model again with each pseudo labeled sample i.e. 1 Step only
    If Entropy increases on unlabeled data we consider the sample HTL
    '''
    def __init__(self, **kwargs):
        self.entrs = []
        self.current_iteration = 0
        self.predictions_over_time = []

    def harsh_crowd(self, probabilities):
        """
        Pseudo Labeling via a crowd of the last 7 predictions
        Parameters
        ----------
        probabilities
        Returns
        -------
        """
        start = -7
        consensus = np.average(probabilities[start:], axis=0)
        return np.argmax(consensus, axis=1)

    def heavy_crowd(self, probabilities):
        """
        Pseudo Labeling with weighted Crowd,
        slightly better than the unweighted version (harsh_crowd)
        But struggles in first few iterations with rapid switches alot more
        Parameters
        ----------
        probabilities
        Returns
        -------
        """
        start = -11
        consensus = np.average(probabilities[start:], axis=0,
                               weights=np.arange(1, min(-start, probabilities.shape[0]) + 1))
        return np.argmax(consensus, axis=1)

    def mixed_crowd(self, probabilities):  # 5 0.905863881072342 + Outlier Avoidance
        if probabilities.shape[0] < 5:
            return self.harsh_crowd(probabilities)
        else:
            return self.heavy_crowd(probabilities)

    def pseudo_label_uncertainty_clipping(self, probabilities):
        start = -14
        mask = np.ones(probabilities.shape[1])
        consensus = np.average(probabilities[start:], axis=0,
                               weights=np.arange(1, min(-start, probabilities.shape[0]) + 1))
        entropies = entropy(consensus, axis=1)
        mask[entropies > np.mean(entropies)] = 0
        return mask


    def calculate_pseudo_labels(self, probabilities):
        pseudo_labels = self.mixed_crowd(probabilities)
        mask = self.pseudo_label_uncertainty_clipping(probabilities)
        return pseudo_labels, mask

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
        # Track class distribution for each sample over iterations
        if self.current_iteration < iteration:
            self.current_iteration = iteration
            probabilities = clf.predict_proba(dataset)
            self.predictions_over_time.append(probabilities)

        # Delay Start by 5 iterations as we don't trust the first 2
        # and prefer having at least 3 voters for Pseudo Labels
        if len(self.predictions_over_time) < 5:
            return np.zeros_like(indices_chosen, dtype=bool)

        # Calculate Pseudo Labels
        pseudo_labels, mask = self.calculate_pseudo_labels(np.array(self.predictions_over_time))

        # For this to work we will need to check alot of models on how uncertain they are on a given DS
        # Fortunately: we don't require Labeled data for this therefore the unlabeled pool provides us a val set
        # Unfortunately: This gets very expensive very fast therefore we need to down sample the val set
        validation_indices = np.random.choice(np.arange(len(dataset)), replace=False, size=min(len(dataset), 1000))
        validation_set = dataset[validation_indices].clone()
        validation_set.y = np.zeros_like(validation_indices)
        # Test Current Model to set Baseline i.e. what is the avg. entr. of the current model on the val set
        proba = clf.predict_proba(validation_set)
        entropies = entropy(proba, axis=1)
        entropy_original = np.average(entropies)

        REPETITIONS = 5
        new_entropies = []
        # Train copies of the current model for a single step on a single sample
        # If Average Entropy increases then we consider it HTL
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
                # Clean Up
                del clf_
                torch.cuda.empty_cache()
            # Write down average entropy for current sample over multiple runs
            new_entropies.append(np.average(scores))

        new_entropies = np.array(new_entropies)
        # Measure whether each sample has improved or worsen performance
        differences = new_entropies - entropy_original
        # Calculate threshold to remove samples that performed especially bad
        mean = np.mean(differences)
        std = np.std(differences)
        threshold = mean + 2 * std
        # Remove samples that made model worse (0)
        # or if many worsen perf we only remove samples that perform especially bad (as unlikely that all samples HTL)
        htl_mask = differences > max(threshold, 0)
        return htl_mask




class SingleStepEntropy_SimplePseudo(SingleStepEntropy):
    def calculate_pseudo_labels(self, probabilities):
        return np.argmax(np.average(probabilities[-3:], axis=0), axis=1), None