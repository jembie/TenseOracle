"""
Filters That use AutoEncoders to detect and Avoid HTL Samples (or rather Outliers)

Idea:
    AutoEncoders learn to compress and decompress data
    They can be trained using all data as no label is required
    Therefore, AEs won't have a problem with regular data but when lots of
    information gets lost when compressing and decompressing then this sample doesn't
    represent the data and is probably an outlier.
    Learning this sample won't help as it is unrepresentative and maybe even broken
    e.g. wrong language, wrong genre, many words missing
    i.e. Labels have to be assigned at random if Oracle has not the required background knowledge
    e.g. The language skills or knowing what the author tried to express when creating this sample text
"""
import torch.cuda

from Strategies.filters import FilterStrategy
from collections import defaultdict
from small_text import TransformerBasedClassificationFactory, Dataset, Classifier
import numpy as np
from Utilities.general import SmallTextCartographer
from scipy.special import softmax
from torch import nn
import copy
from torch import optim
from torch.utils.data import DataLoader


class AutoFilter_Chen_Like(FilterStrategy):
    '''
    Idea: Use an Auto-encoder Ensemble to detect outliers
    '''

    def __init__(self, device, **kwargs):
        """

        :param device:
        :param kwargs:
        """
        self.outlier_scores = None
        self.current_iteration = 0
        self.EPOCH_COUNT = 10
        self.criterion = None
        self.device = device
        self.predictions = []

    def create_autoencoder(self, current_dim, alpha=0.5, min_nodes=3, max_layers=3):
        """
        Create an AutoEncoder Model mostly as described in "Outlier Detection with Autoencoder Ensembles"
        :param current_dim:
        :param alpha:
        :param min_nodes:
        :param max_layers:
        :return:
        """
        layers = []

        # Building encoder and decoder layers
        layer_dims = []
        while current_dim > min_nodes and max_layers > 0:
            next_dim = max(int(alpha * current_dim), min_nodes)
            layer_dims.append((current_dim, next_dim))
            current_dim = next_dim
            max_layers -= 1

        # Decoder is same in reversed order
        layer_dims = layer_dims + [(input, output) for (output, input) in layer_dims[::-1]]

        # Creating layers with random connection removal
        for i, dims in enumerate(layer_dims):
            layer = nn.Linear(*dims)
            layers.append(layer)
            layers.append(nn.Sigmoid() if (
                    i == 0 or i + 1 == len(layer_dims)) else nn.ReLU())  # First and last sigmoid all others RELU

        return nn.Sequential(*layers)

    def drop_neurons(self, model_in: nn.Module, connection_drop_rate=0.1, device="cuda:0"):
        """
        Set weight of randomly chosen neurons to 0
        i.e. Freeze them
        :param model_in:
        :param connection_drop_rate:
        :param device:
        :return:
        """
        # Clone the model to avoid modifying the original model
        model_out = copy.deepcopy(model_in)

        for layer in model_out.modules():
            if isinstance(layer, nn.Linear):
                with torch.no_grad():
                    mask = (torch.rand(layer.weight.size()) > connection_drop_rate).float()
                    layer.weight = nn.Parameter(layer.weight * mask.to(device))

        return model_out

    def pretrain_layer(self, model: nn.Module, train_loader, layer_index, criterion, device="cuda:0", epochs=5):
        """
        Pretrain a specific layer of the autoencoder.
        Short:
        Freezes all layers except 2 and replaces all in between with a placeholder
        Trains Placeholder and the 2 and moves on to the next 2 (Throws placeholder away)
        From the outside to the inside

        :param model: The autoencoder model.
        :param train_loader: DataLoader for the training data.
        :param layer_index: The index of the layer to be pretrained.
        :param criterion: Loss function.
        :param optimizer: Optimizer.
        :param epochs: Number of epochs for training.
        """
        params = list(model.children())
        # Make copy of original to check if it worked as intended at end
        old_model = copy.deepcopy(model)
        # Create preprocessor i.e. outer layers:
        preprocessor = nn.Sequential(*params[:layer_index * 2])
        # Create a shallow model for the moment with only the 2 layers that shall be trained + 1 Placeholder
        encoder = params[layer_index * 2]
        decoder = params[-(layer_index + 1) * 2]
        placeholder = nn.Linear(encoder.out_features, decoder.in_features)
        module_tmp = nn.Sequential(encoder, placeholder, decoder).to(device=device)

        optimizer = optim.Adam(module_tmp.parameters(), lr=0.001)
        # Train shallow model for a few epochs
        for epoch in range(epochs):
            for data in train_loader:
                # Forward pass through the layers up to the one we're training
                x = data.to(device=device)
                with torch.no_grad():
                    x = preprocessor(x)

                decoded = module_tmp(x)

                # We only try to reconstruct how it looked before it went through the current layers of interest
                loss = criterion(decoded, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Check Condition: Only 4 Layers may be updated at once (2 in encoder & 2 in decoder)
        # All others should have remained unchanged
        old_weights = [p.detach().to("cpu") for p in old_model.parameters()]
        new_weights = [p.detach().to("cpu") for p in model.parameters()]
        num_consitent_layers = sum([torch.all(o == n) for (o, n) in zip(old_weights, new_weights)])
        assert len(old_weights) - num_consitent_layers == 4
        return None  # Model was trained in place

    def pretrain_autoencoder(self, model, train_loader, criterion, optimizer, epochs=5):
        """
        Pretrain all layers of the autoencoder.

        :param model: The autoencoder model.
        :param train_loader: DataLoader for the training data.
        :param criterion: Loss function.
        :param optimizer: Optimizer.
        :param epochs: Number of epochs for training.
        """
        # Assuming the model has symmetric encoder and decoder
        num_layers = len(list(model.parameters())) // 2

        for layer_index in range(num_layers):
            print(f"Pretraining layer {layer_index + 1}/{num_layers}")
            self.pretrain_layer(model=model, train_loader=train_loader, layer_index=layer_index, criterion=criterion,
                                device="cuda:0", epochs=epochs)

    def detect_outliers(self, ensemble: list[nn.Module], dataset):
        """
        Use ensemble of autoencoders to detect outliers in the given dataset
        :param ensemble: list of auto encoders
        :param dataset:
        :return:
        """
        losses = []
        # Loop over all models in ensemble
        for m in ensemble:
            # Move Model & data to device
            m.to(self.device)
            embedding = torch.Tensor(dataset).to(self.device)
            # Make Prediction
            predictions = m(embedding)
            loss = []
            # How well did compression and decompression go?
            for target, pred in zip(embedding, predictions):
                loss.append(self.criterion(pred, target).item())
            loss = np.array(loss)
            # Standardize Losses to make them comparable btwn. models
            loss_standardized = (loss - np.mean(loss)) / np.std(loss)
            losses.append(loss_standardized)
            # Clean up
            m.to("cpu:0")
            torch.cuda.empty_cache()

        # Stack Losses
        losses = np.array(losses)
        # Use median loss for each sample
        outlier_scores = np.median(losses, axis=0)
        # Classical Outlier Detection mean + s standard deviations as thresholds
        htl_mask = outlier_scores > (np.mean(outlier_scores) + 2 * np.std(outlier_scores))

        return htl_mask

    def train_ensemble(self, embeddings):
        """
        Train An AutoEncoder Ensemble (mostly) as described in
        "Outlier Detection with Autoencoder Ensembles" by Chen et al.

        :param embeddings:
        :return:
        """
        # Create the (untrained) model
        base_model = self.create_autoencoder(embeddings[0].shape[0])

        # Make Preparations
        data_loader = DataLoader([x for x in embeddings], batch_size=16, shuffle=True)
        self.criterion = nn.MSELoss()
        optimizer = optim.Adam(base_model.parameters(), lr=0.001)
        base_model = base_model.to(device=self.device)

        # Pretrain a base model (something like a template)
        self.pretrain_autoencoder(base_model, data_loader, self.criterion, optimizer, epochs=3)

        ENSEMBLE_SIZE = 30
        ensemble = []
        for i in range(ENSEMBLE_SIZE):
            # Create a copy of base model with some neurons frozen (Like a Stable Dropout)
            model = self.drop_neurons(base_model, 0.1)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            # Train Copy
            for e in range(self.EPOCH_COUNT):
                for data in data_loader:
                    optimizer.zero_grad()
                    # Forward pass
                    input = data.to(device=self.device)
                    output = model(input)

                    loss = self.criterion(output, input)
                    loss.backward()
                    for layer in model.modules():
                        if isinstance(layer, nn.Linear):
                            # Avoid Updating Frozen Neurons
                            with torch.no_grad():
                                mask = (layer.weight != 0).float()
                                layer.weight.grad *= mask
                    optimizer.step()
            ensemble.append(model)

        return ensemble

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
        # Delay start because depends on CLF Embedding which gets (currently) finetuned on labeled data
        # TODO (FUTURE WORK): unsupervised/self supervised finetuning should suffice
        # E.g. MLM, NSP, ClozeQuestions or SetFit Style training could allow usage already in first iteration
        if iteration < 6:
            return np.zeros(indices_chosen.shape, dtype=bool)

        # Only need to calculate outliers scores once
        if self.outlier_scores is None:
            # Embed data (we try to find outliers with weird embeddings)
            embeddings = clf.embed(dataset, embedding_method="cls")

            # Train Autoencoders
            ensemble = self.train_ensemble(embeddings)

            # Use ensemble to map each sample to an outlier score then threshold it to HTL or Not (Binary label)
            self.outlier_scores = self.detect_outliers(ensemble, embeddings)

        return self.outlier_scores[indices_chosen]
