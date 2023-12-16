import numpy as np
from abc import ABC, abstractmethod
from small_text.classifiers import Classifier
from small_text.data.datasets import Dataset


class FilterStrategy(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self,
                 indices_chosen: np.ndarray,
                 confidence: np.ndarray,
                 indices_already_avoided: list,
                 clf: Classifier,
                 dataset: Dataset,
                 indices_unlabeled: np.ndarray,
                 indices_labeled: np.ndarray,
                 y: np.ndarray,
                 n=10,
                 iteration=0) -> np.ndarray:
        '''
        In Short: Takes in np.ndarray of chosen sample indices and returns a boolean mask (np.ndarray)
        with False on all the samples that are ok (not HTL) and True on all the samples that are considered HTL.

        Has access to same Parameters as Acquisition Strategy
        + Indices the mentioned Strategy has chosen this epoch
        + Indices this Filter has already Avoided
        :param indices_chosen: Indices of Samples That shall be queried this iteration
        :param indices_already_avoided: Indices that this filter has already avoided in previous iterations
        :param clf: Current Classifier Already Trained on Labeled set
        :param dataset: entire dataset without Labels
        :param indices_unlabeled: numpy array of indices that remain unlabeled (includes the ones that were found HTL)
        :param indices_labeled: numpy array of indices that were already labeled by the Oracle
        :param y: np.ndarray[int] or csr_matrix
                  List of labels where each label maps by index position to `indices_labeled`.
        :param n: (int) Number of samples to query.
        :return: boolean mask (np.ndarray)
        '''


class RandomFilter(FilterStrategy):
    """
    Chooses which samples are considered HTL at Random (for Debugging Only)
    """

    def __call__(self,
                 indices_chosen: np.ndarray,
                 confidence: np.ndarray,
                 indices_already_avoided: list,
                 clf: Classifier,
                 dataset: Dataset,
                 indices_unlabeled: np.ndarray,
                 indices_labeled: np.ndarray,
                 y: np.ndarray,
                 n=10) -> np.ndarray:
        a = np.zeros_like(indices_chosen, dtype=bool)
        a[:len(indices_chosen)//3] = True
        np.random.shuffle(a)
        return a