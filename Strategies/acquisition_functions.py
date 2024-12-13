from small_text.query_strategies.strategies import QueryStrategy, RandomSampling, ConfidenceBasedQueryStrategy
import AE_filters
from scipy.stats import entropy
from abc import ABC, abstractmethod
import numpy as np
import torch


class UncertaintyClipping(QueryStrategy):
    """A base class for confidence-based querying.

    To use this class, create a subclass and implement `get_confidence()`.
    """

    def __init__(self, lower_is_better=False, **kwargs):
        self.lower_is_better = lower_is_better
        self.scores_ = None

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        confidence, proba, embeddings = self.score(clf, dataset, indices_unlabeled, indices_labeled, y)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        lowest_5_percent = int(len(indices_unlabeled) * 0.05)
        indices_partitioned = np.argsort(confidence[indices_unlabeled])[lowest_5_percent : lowest_5_percent + n]
        # return confidence as well to save time
        return (
            np.array([indices_unlabeled[i] for i in indices_partitioned]),
            confidence,
            proba,
            embeddings,
        )

    def score(self, clf, dataset, indices_unlabeled, indices_labeled, y):
        """Assigns a confidence score to each instance.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : small_text.data.datasets.Dataset
            A text dataset.
        indices_unlabeled : np.ndarray[int]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : np.ndarray[int]
            Indices (relative to `dataset`) for the labeled data.
        y : np.ndarray[int] or csr_matrix
            List of labels where each label maps by index position to `indices_labeled`.

        Returns
        -------
        confidence : np.ndarray[float]
            Array of confidence scores in the shape (n_samples, n_classes).
            If `self.lower_is_better` the confiden values are flipped to negative so that
            subsequent methods do not need to differentiate maximization/minimization.
        """

        confidence, proba, embeddings = self.get_confidence(clf, dataset, indices_unlabeled, indices_labeled, y)
        self.scores_ = confidence
        if not self.lower_is_better:
            confidence = -confidence

        return confidence, proba, embeddings

    @abstractmethod
    def get_confidence(self, clf, dataset, indices_unlabeled, indices_labeled, y):
        """Computes a confidence score for each of the given instances.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : small_text.data.datasets.Dataset
            A text dataset.
        indices_unlabeled : np.ndarray[int]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : np.ndarray[int]
            Indices (relative to `dataset`) for the labeled data.
        y : np.ndarray[int] or csr_matrix
            List of labels where each label maps by index position to `indices_labeled`.
        Returns
        -------
        confidence : ndarray[float]
            Array of confidence scores in the shape (n_samples, n_classes).
        """
        pass

    def __str__(self):
        return 'ConfidenceBasedQueryStrategy()'

class ReconstructionLossClipping(QueryStrategy):
    """A base class for confidence-based querying.

    To use this class, create a subclass and implement `get_confidence()`.
    """

    def __init__(self, lower_is_better=False, args = None, **kwargs):
        self.lower_is_better = lower_is_better
        self.scores_ = None
        self.device = "cpu:0" if torch.cuda.device_count() == 0 else "cuda:0"
        self.iter_counter = 0
        self.autoencoder = AE_filters.AutoFilter_Chen_Like(device=self.device, percentile=args.percentile)
        self.htl_mask = []

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self.iter_counter += 1
        self._validate_query_input(indices_unlabeled, n)

        confidence, proba, embeddings = self.score(clf, dataset, indices_unlabeled, indices_labeled, y)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        self.htl_mask = self.autoencoder(
            indices_chosen=indices_unlabeled, indices_unlabeled=indices_unlabeled, indices_labeled=indices_labeled, indices_already_avoided=[], confidence=confidence, embeddings=embeddings, probas=proba, clf=clf, dataset=dataset, n=len(indices_unlabeled), y=y, iteration=self.iter_counter
            )
        
        unlabeled_pool = indices_unlabeled[~self.htl_mask]
        confidence = confidence[unlabeled_pool]
        indices_partitioned = np.argsort(confidence)[:n]
        # return confidence as well to save time
        return (
            np.array([indices_unlabeled[i] for i in indices_partitioned]),
            confidence,
            proba,
            embeddings,
        )


    def score(self, clf, dataset, indices_unlabeled, indices_labeled, y):
        """Assigns a confidence score to each instance.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : small_text.data.datasets.Dataset
            A text dataset.
        indices_unlabeled : np.ndarray[int]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : np.ndarray[int]
            Indices (relative to `dataset`) for the labeled data.
        y : np.ndarray[int] or csr_matrix
            List of labels where each label maps by index position to `indices_labeled`.

        Returns
        -------
        confidence : np.ndarray[float]
            Array of confidence scores in the shape (n_samples, n_classes).
            If `self.lower_is_better` the confiden values are flipped to negative so that
            subsequent methods do not need to differentiate maximization/minimization.
        """

        confidence, proba, embeddings = self.get_confidence(clf, dataset, indices_unlabeled, indices_labeled, y)
        self.scores_ = confidence
        if not self.lower_is_better:
            confidence = -confidence

        return confidence, proba, embeddings

    @abstractmethod
    def get_confidence(self, clf, dataset, indices_unlabeled, indices_labeled, y):
        """Computes a confidence score for each of the given instances.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : small_text.data.datasets.Dataset
            A text dataset.
        indices_unlabeled : np.ndarray[int]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : np.ndarray[int]
            Indices (relative to `dataset`) for the labeled data.
        y : np.ndarray[int] or csr_matrix
            List of labels where each label maps by index position to `indices_labeled`.
        Returns
        -------
        confidence : ndarray[float]
            Array of confidence scores in the shape (n_samples, n_classes).
        """
        pass

    def __str__(self):
        return 'ConfidenceBasedQueryStrategy()'


class PredictionEntropyUncertaintyClipped(UncertaintyClipping):
    """Selects instances with the largest prediction entropy [HOL08]_."""

    def __init__(self, **kwargs):
        super().__init__(lower_is_better=False)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        # proba_ = clf.predict_proba(dataset)  # Average Dur 88s
        embeddings, proba = clf.embed(dataset, return_proba=True, embedding_method="cls")  # average dur 84s
        return np.apply_along_axis(lambda x: entropy(x), 1, proba), proba, embeddings

    def __str__(self):
        return "PredictionEntropy()"


class PredictionEntropy(ConfidenceBasedQueryStrategy):
    """Selects instances with the largest prediction entropy [HOL08]_."""
    def __init__(self, **kwargs):
        super().__init__(lower_is_better=False)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        #proba_ = clf.predict_proba(dataset)  # Average Dur 88s
        embeddings, proba = clf.embed(dataset, return_proba=True, embedding_method="cls")  # average dur 84s
        return np.apply_along_axis(lambda x: entropy(x), 1, proba), proba, embeddings

    def __str__(self):
        return 'PredictionEntropy()'