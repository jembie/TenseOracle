import torch
import numpy as np
import random
from small_text import TransformersDataset, Classifier
from typing import Callable
from transformers import TrainerCallback

def set_random_seed(seed):
    # set random seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



class SmallTextCartographer():
    def __init__(self, dataset: TransformersDataset,
                 outputs_to_probabilities: Callable = None,
                 ):
        """
        :param dataset: Dataset. Usually, as the paper suggests, this is the training dataset. It should be:
             - Non-shuffled, so each iteration over the dataset should yield samples in the same order
             - Labels should be sparse encoded i.e. as integer not one-hot
             - label-column shall be called 'label' and only contain a single label per row
        :param outputs_to_probabilities: Callable to convert model's output to probabilities. Use this if the model
            outputs logits, dictionary or any other form which is not a vector of probabilities.
        """
        self.dataset = dataset
        self.super_dataset = None

        # Stores the probabilities for the gold labels after each epoch,
        # e.g. self._gold_labels_probabilities[i] == gold label probabilities at the end of epoch i
        self._gold_labels_probabilities = []
        self._gold_label_hits = []
        self.outputs2probabilities = outputs_to_probabilities


    def process_predictions(self, probabilities):
        # Convert outputs to probabilities if necessary
        if self.outputs2probabilities is not None:
            probabilities = self.outputs2probabilities(probabilities)

        if isinstance(self.dataset, TransformersDataset):
            y = self.dataset.y
        else:
            y = self.dataset["label"]
        gold_probabilities = probabilities[np.arange(probabilities.shape[0]), y]
        gold_probabilities = np.expand_dims(gold_probabilities, axis=-1)
        self._gold_labels_probabilities.append(gold_probabilities)

        gold_hits = 1 * (np.argmax(probabilities, axis=1) == y)
        gold_hits = np.expand_dims(gold_hits, axis=-1)
        self._gold_label_hits.append(gold_hits)
        return


    @property
    def gold_labels_probabilities(self) -> np.ndarray:
        """
        Gold label predicted probabilities. With the shape of ``(n_samples, n_epochs)`` and ``(i, j)`` is the
        probability of the gold label for sample ``i`` at epoch ``j``
        :return: Gold label probabilities
        """
        return np.hstack(self._gold_labels_probabilities)

    @property
    def confidence(self) -> np.ndarray:
        """
        Average true label probability across epochs
        :return: Confidence
        """
        return np.mean(self.gold_labels_probabilities, axis=-1)

    @property
    def variability(self) -> np.ndarray:
        """
        Standard deviation of true label probability across epochs
        :return: Variability
        """
        return np.std(self.gold_labels_probabilities, axis=-1)

    @property
    def correctness(self) -> np.ndarray:
        """
        Proportion of correct predictions made across epochs.
        :return: Correctness
        """
        return np.mean(np.hstack(self._gold_label_hits), axis=-1)


    def on_epoch_end(self, clf: Classifier):
        predictions = clf.predict(self.dataset, return_proba=True)
        probabilities = predictions[1]
        self.process_predictions(probabilities=probabilities)
        return