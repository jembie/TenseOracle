from small_text.query_strategies.strategies import QueryStrategy, RandomSampling, PredictionEntropy
from scipy.stats import entropy
import numpy as np


class UncertaintyClipping(QueryStrategy):
    """A base class for confidence-based querying.

    To use this class, create a subclass and implement `get_confidence()`.
    """

    def __init__(self, lower_is_better=False):
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


class PredictionEntropyUncertaintyClipped(UncertaintyClipping):
    """Selects instances with the largest prediction entropy [HOL08]_."""

    def __init__(self):
        super().__init__(lower_is_better=False)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        # proba_ = clf.predict_proba(dataset)  # Average Dur 88s
        embeddings, proba = clf.embed(dataset, return_proba=True, embedding_method="cls")  # average dur 84s
        return np.apply_along_axis(lambda x: entropy(x), 1, proba), proba, embeddings

    def __str__(self):
        return "PredictionEntropy()"
