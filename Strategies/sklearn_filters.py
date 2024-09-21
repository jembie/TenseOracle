from numpy import ndarray

import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import HDBSCAN

from small_text.classifiers.classification import Classifier
from small_text.data.datasets import Dataset
from Strategies.filters import FilterStrategy


def detect_outliers(filter_strategy: FilterStrategy, train_data: ndarray, outliers_to_check: ndarray) -> ndarray:
    prediction = filter_strategy.fit_predict(train_data)
    boolean_mask = (prediction[outliers_to_check]) == -1
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
        boolean_mask = detect_outliers(
            filter_strategy=IsolationForest(n_estimators=400, random_state=self.seed, max_samples=len(embeddings), n_jobs=-1),
            train_data=embeddings,
            outliers_to_check=indices_chosen,
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
        boolean_mask = detect_outliers(
            filter_strategy=LocalOutlierFactor(n_neighbors=20, metric="cosine", novelty=False, n_jobs=-1),
            train_data=embeddings, 
            outliers_to_check=indices_chosen
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
        hdb = HDBSCAN(metric="cosine", n_jobs=-1, min_cluster_size=20)
        hdb.fit(embeddings)
        labels = hdb.labels_ == -1
        print(f"Total Outliers detected for: HDBScan {sum(labels) / len(labels)}")

        return labels[indices_chosen]
