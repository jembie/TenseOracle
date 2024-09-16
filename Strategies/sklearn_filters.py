from numpy import ndarray

import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import HDBSCAN

from small_text.classifiers.classification import Classifier
from small_text.data.datasets import Dataset
from Strategies.filters import FilterStrategy


def detect_outliers(filter_strategy: FilterStrategy, train_data: ndarray, outliers_to_check: ndarray) -> ndarray:
    outlier_classifier = filter_strategy.fit(train_data)

    score_samples = outlier_classifier.decision_function(train_data)
    mean = np.mean(score_samples)
    std = np.std(score_samples)

    threshold = (
        mean - 2 * std
    )  # The lower, the more abnormal. Negative scores represent outliers, positive scores represent inliers. That's why mean - 2 * std instead of mean + 2 * std

    boolean_mask = outlier_classifier.decision_function(outliers_to_check) < threshold

    return boolean_mask


class IsolationForestFilter(FilterStrategy):
    def __init__(self, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

    def __call__(
        self,
        indices_chosen: ndarray,
        confidence: ndarray,
        embeddings: ndarray,
        probas: ndarray,
        indices_already_avoided: list,
        clf: Classifier,
        dataset: Dataset,
        indices_unlabeled: ndarray,
        indices_labeled: ndarray,
        y: ndarray,
        n=10,
        iteration=0,
    ) -> ndarray:
        outliers_to_check = embeddings[indices_chosen]
        # train_data = np.delete(embeddings, indices_chosen)

        boolean_mask = detect_outliers(
            filter_strategy=IsolationForest(n_estimators=400, random_state=self.seed, max_samples=len(embeddings)),
            train_data=embeddings,
            outliers_to_check=outliers_to_check,
        )

        return boolean_mask


class LocalOutlierFactorFilter(FilterStrategy):
    def __init__(self, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

    def __call__(
        self,
        indices_chosen: ndarray,
        confidence: ndarray,
        embeddings: ndarray,
        probas: ndarray,
        indices_already_avoided: list,
        clf: Classifier,
        dataset: Dataset,
        indices_unlabeled: ndarray,
        indices_labeled: ndarray,
        y: ndarray,
        n=10,
        iteration=0,
    ) -> ndarray:
        outliers_to_check = embeddings[indices_chosen]
        train_data = np.delete(embeddings, indices_chosen, axis=0)

        boolean_mask = detect_outliers(
            filter_strategy=LocalOutlierFactor(metric="cosine", novelty=True), train_data=train_data, outliers_to_check=outliers_to_check
        )

        return boolean_mask


class HDBScanFilter(FilterStrategy):
    def __init__(self, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

    def __call__(
        self,
        indices_chosen: ndarray,
        confidence: ndarray,
        embeddings: ndarray,
        probas: ndarray,
        indices_already_avoided: list,
        clf: Classifier,
        dataset: Dataset,
        indices_unlabeled: ndarray,
        indices_labeled: ndarray,
        y: ndarray,
        n=10,
        iteration=0,
    ) -> ndarray:
        hdb = HDBSCAN(metric="cosine")
        hdb.fit(embeddings)
        labels = hdb.labels_ < 0
        print(f"Total Outliers detected for: HDBScan {sum(labels) / len(labels)}")

        return labels[indices_chosen]
