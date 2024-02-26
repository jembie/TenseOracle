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
from scipy.stats import entropy
from torch import nn
import copy
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import RobertaTokenizer


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
                 embeddings: np.ndarray,
                 probas: np.ndarray,
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
            #embeddings = clf.embed(dataset, embedding_method="cls")

            # Train Autoencoders
            ensemble = self.train_ensemble(embeddings)

            # Use ensemble to map each sample to an outlier score then threshold it to HTL or Not (Binary label)
            self.outlier_scores = self.detect_outliers(ensemble, embeddings)

        return self.outlier_scores[indices_chosen]


class Autoencoder(nn.Module):
    """
    A simple LSTM Based AutoEncoder
    """
    def __init__(self, vocab_size, embed_dim):
        super(Autoencoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Encoder
        self.encoder_lstm1 = nn.LSTM(embed_dim, 256, 3, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(256, 128, 3, batch_first=True)
        self.encoder_linear = nn.Linear(128, 64)

        # Decoder
        self.decoder_lstm1 = nn.LSTM(64, 128, 3, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(128, 256, 3, batch_first=True)
        self.decoder_linear = nn.Linear(256, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.encoder_lstm1(x)
        x, _ = self.encoder_lstm2(x)
        x = self.encoder_linear(x)  # Consider the last output of the sequence

        x, _ = self.decoder_lstm1(x)
        x, _ = self.decoder_lstm2(x)
        x = self.decoder_linear(x)
        x = self.softmax(x)

        return x

class AutoFilter_LSTM(FilterStrategy):
    '''
    Idea: Use Auto-encoders to detect outliers
    Codename: AE-Class-Certain-T1
    Features:
    - LSTM based Autoencoders
    - Use 500 samples with currently highest entropy to define baseline
    - Consider only the 1% (i.e. about 5 Samples) with the highest Loss among the 500 as HTL
    - Train Ensemble
    - Create Diversity by pseudo labeling data and splitting dataset by class label
    '''
    def __init__(self, tokenizer: RobertaTokenizer, device, **kwargs):
        '''
        :param kwargs: requires argument with a TransformerBasedClassificationFactory (kwargs["tokenizer"])
        :return:
        '''
        self.tokenizer = tokenizer
        self.committee = []
        self.committee_size = None
        self.EPOCH_COUNT = 10
        self.criterion = None
        self.device = device
        self.threshold = None
        self.scores = None
        self.predictions = []

    def train_model(self, dataset)->nn.Module:
        """
        Create an LSTM model
        Train it on the given dataset and return it
        :param dataset:
        :return:
        """
        # Initialize model, loss, optimizer
        model = Autoencoder(self.tokenizer.vocab_size, 768)
        self.criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model = model.to(device=self.device)

        EPOCH_COUNT = self.EPOCH_COUNT

        # Create Dataloader
        data_loader = DataLoader([x[0] for x in dataset.x], batch_size=16, shuffle=True)
        # Train Loop
        for epoch in range(EPOCH_COUNT):
            for idx, line in enumerate(data_loader):
                line_ = line.to(self.device)
                optimizer.zero_grad()
                output = model(line_)
                loss = self.criterion(torch.transpose(output,dim0=1,dim1=2), line_)
                loss.backward()
                optimizer.step()
                if idx % 500 == 0:
                    print(f'Epoch [{epoch + 1}/{EPOCH_COUNT}], Loss: {loss.item():.4f}')
        # Finish Training
        model.eval()
        return model

    def calculate_loss(self, text, model: nn.Module):
        """
        Takes in a model and an example
        Let's the model compress and decompress the sample
        :param text:
        :param model:
        :return: Loss achieved during reconstruction
        """
        tokens = text[0].to(self.device)
        output = model(tokens)
        loss = self.criterion(output, tokens)
        return loss.item()


    def detect_outliers(self, dataset, indices_chosen: np.ndarray):
        def calc_threshold(model):  # self.scores is None:
            # Sample the 500 samples with the highest Entropy to calc mean and std for threshold
            indices_high_entr = np.argsort(self.last_confidence)[:500]
            # Calculate the loss of the given model on each chosen sample
            scores = []
            for idx in tqdm(indices_high_entr):
                scores.append(self.calculate_loss(dataset.x[idx], model))
            self.scores = np.array(scores)

            # Calculate threshold as the 99th percentile of the losses,
            # i.e. the 1% with the highest losses will be considered HTL
            return np.percentile(self.scores, 99)

        masks = []
        # Loop over all models in the ensemble
        for m in self.committee:
            # Prepare Model
            m.to(self.device)
            # Let model vote for what it considers HTL
            threshold = calc_threshold(m)
            losses = [self.calculate_loss(dataset.x[idx], model=m) for idx in indices_chosen]
            # Collect Votes
            masks.append(np.array(losses) > threshold)
            # Clean Up afterwards
            m.to("cpu:0")
            torch.cuda.empty_cache()

        # Every sample that gets a majority vote is considered HTL
        htl_mask = np.sum(np.stack(masks), axis=0) > (len(self.committee)//2)

        return htl_mask

    def __call__(self,
                 indices_chosen: np.ndarray,
                 indices_already_avoided: list,
                 confidence: np.ndarray,
                 embeddings: np.ndarray,
                 probas: np.ndarray,
                 clf: Classifier,
                 dataset: Dataset,
                 indices_unlabeled: np.ndarray,
                 indices_labeled: np.ndarray,
                 y: np.ndarray,
                 n=10,
                 iteration=0) -> np.ndarray:
        predictions = probas  # clf.predict_proba(dataset)
        self.predictions.append(predictions)
        self.last_confidence = confidence

        if iteration < 6:  # Skip first 6 iterations because CLF not useful
            return np.zeros(indices_chosen.shape, dtype=bool)

        torch.cuda.empty_cache()
        if self.committee_size is None:
            pred = np.array(self.predictions)
            # Use Average prediction of the last 4 Epochs as those are usually the most reliable
            pred_avg = np.average(pred[-4:], axis=0)
            # Calculate Certainty of average distribution (i.e. did they all agree/disagree were they all uncertain?)
            entr = entropy(pred_avg, axis=1)
            # Which Samples have exceptionally high entropy
            entr_mask = entr < np.mean(entr)+2*np.std(entr)
            # Assign the most likely class to each sample based on averaged distributions
            classes = np.argmax(pred_avg, axis=1)

            # Split Dataset by Class Labels (only consider Labels we are relatively certain about)
            splits = []
            for c in set(classes):
                splits.append(np.argwhere((classes == c) & entr_mask).flatten())

            # 3 models per class
            self.committee_size = 3 * len(splits)

            # Train Models
            for i in range(self.committee_size):
                # Select one of the splits created earlier
                train_indices = copy.copy(splits[i%len(splits)])
                np.random.shuffle(train_indices)
                train_set = dataset[train_indices]
                # Train model only on samples of one class to get sufficient diversity
                model = self.train_model(train_set)
                # "Save" trained Model
                self.committee.append(model)
                # Clean Up again
                model.to("cpu:0")
                torch.cuda.empty_cache()
        htl_mask = self.detect_outliers(dataset, indices_chosen)
        return htl_mask


class AutoFilter_LSTM_SIMPLE(AutoFilter_LSTM):
    """
    Idea: Use Auto-encoders to detect outliers
    Codename: AE-THR-WOW-T!
    Features:
    - LSTM based Autoencoder
    - Use 500 samples with currently highest entropy to define baseline
    - Consider only the 1% (i.e. about 5 Samples) with the highest Loss among the 500 as HTL
    - Train Single LSTM Model
    - Doesn't need Labeled data so can start immediately
    """
    def __init__(self, tokenizer: RobertaTokenizer, device, **kwargs):
        '''
        :param kwargs: requires argument with a TransformerBasedClassificationFactory (kwargs["tokenizer"])
        :return:
        '''
        self.tokenizer = tokenizer
        self.model = None
        self.criterion = None
        self.device = device
        self.threshold = None
        self.scores = None
        self.EPOCH_COUNT = 10


    def detect_outliers(self, dataset, indices_chosen: np.ndarray):
        """
        Finds HTL Samples, by
        Calculating threshold based on forced ranking of the 500 samples with currently highest Entropy
        :param dataset:
        :param indices_chosen:
        :return: Mask of which of the chosen samples it thinks is HTL
        """
        # Sample 500 samples with the highest Entropy to calc mean and std for threshold
        indices_high_entr = np.argsort(self.last_confidence)[:500]
        scores = []
        for idx in tqdm(indices_high_entr):
            scores.append(self.calculate_loss(dataset.x[idx], self.model))
        self.scores = np.array(scores)

        # Forced Ranking: We assume that 1% of samples with the highest Loss in those 500 are HTL
        self.threshold = np.percentile(self.scores, 99)
        # Calculate Reconstruction Loss of chosen samples
        losses = [self.calculate_loss(dataset.x[idx], self.model) for idx in indices_chosen]
        # Return Mask
        return np.array(losses) > self.threshold

    def __call__(self,
                 indices_chosen: np.ndarray,
                 indices_already_avoided: list,
                 confidence: np.ndarray,
                 embeddings: np.ndarray,
                 probas: np.ndarray,
                 clf: Classifier,
                 dataset: Dataset,
                 indices_unlabeled: np.ndarray,
                 indices_labeled: np.ndarray,
                 y: np.ndarray,
                 n=10,
                 iteration=0) -> np.ndarray:
        # Doesn't require any Labels to run so can start immediately
        if self.model is None:
            self.model = self.train_model(dataset)
        self.last_confidence = confidence
        htl_mask = self.detect_outliers(dataset, indices_chosen)
        return htl_mask