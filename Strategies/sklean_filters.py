from numpy import ndarray
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from small_text.classifiers.classification import Classifier
from small_text.data.datasets import Dataset
from Strategies.filters import FilterStrategy


class IsolationForestFilter(FilterStrategy):
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
        isolation_forest = IsolationForest(random_state=0).fit(embeddings)
        chosen_data = embeddings[indices_chosen]
        prediction = isolation_forest.predict(chosen_data)

        boolean_prediction = (prediction == 1)
        return boolean_prediction


class LocalOutlierFactorFilter(FilterStrategy):
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
        pass


class SGDOneClassSVMFilter(FilterStrategy):
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
        pass


class OneClassSVMFilter(FilterStrategy):
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
        pass


class EllipticEnvelopeFilter(FilterStrategy):
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
        pass
