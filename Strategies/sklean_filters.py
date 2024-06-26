from numpy import ndarray

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import HDBSCAN

from small_text.classifiers.classification import Classifier
from small_text.data.datasets import Dataset
from Strategies.filters import FilterStrategy


def detect_outliers(
    filter_strategy: FilterStrategy, embeddings: ndarray, indices_chosen: ndarray
) -> ndarray:
    outlier_classifier = filter_strategy.fit(embeddings)
    prediction = outlier_classifier.predict(embeddings)

    boolean_mask = prediction < 0
    print(f"Total Outliers detected for: {filter_strategy.__class__} {sum(boolean_mask) / len(boolean_mask)}")

    return boolean_mask[indices_chosen]


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
            filter_strategy=IsolationForest(random_state=self.seed),
            embeddings=embeddings,
            indices_chosen=indices_chosen,
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

        lof = LocalOutlierFactor()
        data = lof.fit_predict(embeddings)
        chosen_data = data[indices_chosen]

        boolean_mask = chosen_data < 0
        print(f"Total Outliers detected for: LocalOutlierFactor {sum(boolean_mask) / len(boolean_mask)}")

        return boolean_mask


class SGDOneClassSVMFilter(FilterStrategy):

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
            filter_strategy=SGDOneClassSVM(random_state=self.seed),
            embeddings=embeddings,
            indices_chosen=indices_chosen,
        )

        return boolean_mask


class OneClassSVMFilter(FilterStrategy):

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
            filter_strategy=OneClassSVM(),
            embeddings=embeddings,
            indices_chosen=indices_chosen,
        )

        return boolean_mask


class EllipticEnvelopeFilter(FilterStrategy):

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
            filter_strategy=EllipticEnvelope(random_state=self.seed),
            embeddings=embeddings,
            indices_chosen=indices_chosen,
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

        hdb = HDBSCAN()
        hdb.fit(embeddings)
        labels = (hdb.labels_ < 0)
        print(f"Total Outliers detected for: HDBScan {sum(labels) / len(labels)}")

        return labels[indices_chosen]
