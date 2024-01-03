"""
Filters That use Learning Entropy to detect HTL samples

Idea:
    Summary: If Class distribution CHANGES in a wierd way it is HTL
    Thesis1: With knowledge gained over epochs One Class (i.e. the true class of a sample)
    should get more and more clearly (i.e. it's probability should increase after each epoch)
    All other classes should Generally decrease with every iteration
    That is we only look at the CHANGE of the class distribution over iterations instead of the actual
    distribution itself.
    Thesis2: HTL samples as they have no true label or true label that the model can identify won't
    behave that way i.e. some classes will become more likely just to become less likely and vice versa
    Therefore: If both Thesis are true then we could identify HTL samples by how easily we can identify
    the class that gets more likely after each iteration, i.e. if one gets clearly more likely with each
    iteration while all others degenerate with each iteration then it is NOT HTL but if we are especially
    unsure about it then we consider as HTL
    Alternative Reason why helpful: if Learning Entropy high that means previously labeled samples either send
    mixed signals or none in either case this sample is so different from the ones that came before
    (and therefore probably also the ones that follow)  s.t. the model wouldn't profit from a label
     for this sample as it is likely unrepresentative
"""
import torch.cuda

from Strategies.filters import FilterStrategy
from collections import defaultdict
from small_text import TransformerBasedClassificationFactory, Dataset, Classifier
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy


class TeachingFilter(FilterStrategy):
    '''
    Codename: LE-Clean
    Idea: We track over multiple iterations a learning metric e.g. Entropy and how much it changes
    If it improved less over the iterations than most others then we assume HTL
    Assumption: If the sample didn't learn from all the ones that came before
    then the others won't learn from it as well because they are too dissimilar
    '''

    def __init__(self, **kwargs):
        self.probs = []
        self.current_iteration = 0

    def moving_average(self, a, n=3, axis=2):
        """Simple Moving Average Formular"""
        # Swap the target axis with last axis
        a_swapped = np.swapaxes(a, axis, -1)

        # Apply MA along axis
        ret = np.cumsum(a_swapped, axis=-1, dtype=float)
        ret[..., n:] = ret[..., n:] - ret[..., :-n]
        divisor = np.arange(1, ret.shape[-1] + 1)
        divisor[divisor > n] = n
        ret = ret / divisor

        # Swap axes ack
        return np.swapaxes(ret, axis, -1)

    def learning_entropy(self, matrix):
        '''
        Assumption: if learning a sample 1 classes probability should increase with every step and all other should fall
        We calculate how monotone (Note: Not how steep) a classes probability rises
        Then we compare it to the behaviour of the other classes via softmax
        Finally we apply entropy to see with which certainty we can find a rising class
        if 0 -> 1 class rises monotone all others fall monotone
        if large either all fall, multiple rise, or just chaotic
        Conclusion we only want samples that have a low entropy
        as they should have similarities with previously sampled data as indicated by low learning entropy
        (knowledge was gathered from previous samples that helped with this sample)
        Therefore we might assume that this sample could help with the others too
        High LearningEntropy might indicate that the sample is so different that the others didn't affect this one and vice versa
        :param matrix:
        :return:
        '''
        # Moving average to reduce random fluctuations
        smooth_matrix = self.moving_average(matrix, n=3, axis=0)
        # Collect Votes at different distances
        # Is the second larger than the first -> +1 if not -1, and so on i.e. 3 larger than second
        # Is the third larger than first if so +1 else -1
        # Note: The third stands for "The expected probability of a certain class for a certain sample in the third iteration"
        # IOW: We simply compare class distributions for a sample over iterations,each class gets a +1 if its probability
        # has increased compared to a previous iteration and a -1 if it has decreased
        results = np.zeros(smooth_matrix.shape[1:])
        for n in range(1, smooth_matrix.shape[0]):
            r = ((smooth_matrix[n:, :, :] - smooth_matrix[:-n, :, :]) > 0)
            results += np.sum(r, axis=0) - np.sum(~r, axis=0)
        # normalize Results
        results_norm = results / sum(range(smooth_matrix.shape[0]))
        # Convert scores into probabilities via softmax
        # i.e. how likely is a class the class we search (i.e. the one which's prob increases after each iteration)
        # Use entropy to find out how certain we are that we found the right one
        return entropy(softmax(results_norm, axis=1), axis=1)

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
        if self.current_iteration < iteration:
            # Track class distribution for each sample over iterations
            self.current_iteration = iteration
            probabilities = clf.predict_proba(dataset)
            self.probs.append(probabilities)

        # Skip first 3 iters as distributions still behave randomly
        if len(self.probs) < 3:
            return np.zeros_like(indices_chosen, dtype=bool)

        # Calculate Learning Entropy
        entropies = np.array(self.probs)
        learning_entropy = self.learning_entropy(entropies)
        # Define Threshold
        std_entropy = np.std(learning_entropy)
        mean_entropy = np.mean(learning_entropy)
        threshold = mean_entropy + 2 * std_entropy
        # Marks all (assumed) HTL samples in ENTIRE dataset
        absolute_mask = learning_entropy > threshold
        return absolute_mask[indices_chosen]