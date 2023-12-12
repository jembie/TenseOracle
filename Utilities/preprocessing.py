import datasets as ds
import transformers as tr
from pathlib import Path
import numpy as np
from typing import Tuple, Iterable
from scipy.sparse import csr_matrix
from small_text.utils.annotations import experimental
from small_text.utils.labels import csr_to_list
from small_text import TransformersDataset

def load_tokenizer(model_name: str, cache_dir: str):
    return tr.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


def load_from_hub(task_config: dict, seed = 42):
    # Some DS are grouped together (Glue) if we want one of those we need a subset of the Group
    subset = task_config["subset"] if "subset" in task_config else None
    # Some tasks require additionally local data
    data_dir = task_config["load_from_dir"] if "load_from_dir" in task_config else None

    # Load Data from hub
    dataset = ds.load_dataset(task_config["dataset_path"],
                              name=subset,
                              data_dir=data_dir,
                              keep_in_memory=True)

    if dataset["train"].num_rows > 20000:
        dataset["train"] = dataset["train"].train_test_split(train_size=20000, shuffle=True, seed=seed)["train"]

    # Some Datasets have only labeled validation (and no test) sets we use those for testing
    dataset["test"] = dataset[task_config["test_set_name"]]

    return dataset


def load_dataset_from_config(task_config: dict, config: dict, args) -> Tuple[TransformersDataset, TransformersDataset]:
    '''
    Loads a Dataset with respect to the given Configurations.
    :returns a tuple of a (Train, Test) Set in TransformerDataset Format
    '''
    tokenizer = load_tokenizer(config["MODEL_NAME"], config["SHARED_CACHE_ADR"])
    dataset = load_from_hub(task_config, seed=args.random_seed)

    match task_config:
        case {  # Simple Text Classification from Huggingface
            "task_class": "text_classification",
            "input_text_column": input_column,
            "label_column": label_column,
        }:

            # Small Text Requires a Special Format
            train, test = convert_to_transformer_ds(dataset,
                                                    tokenizer,
                                                    input_column,
                                                    label_column,
                                                    )

        case {  # Text Pair Classification
            "task_class": "text_pair_classification",
            "input_text_1": input_column1,
            "input_text_2": input_column2,
            "label_column": label_column,
        }:
            train, test = convert_to_transformer_ds(dataset,
                                                    tokenizer,
                                                    (input_column1, input_column2),
                                                    label_column,
                                                    )

        case _:
            raise ValueError(f"The Dataset config {task_config} seems to be invalid")

    # Check whether train is larger than queried samples else AL gets useless
    assert dataset["train"].num_rows > config["SEED_SIZE"] + (config["ITERATIONS"] * config["QUERY_BATCH_SIZE"])
    print(dataset)

    return (train, test)


def get_label_set(dataset: ds.DatasetDict, label_col) -> list:
    '''Takes in a dataset Dict and returns a list of all used Labels'''
    label_set = list(set(ds.concatenate_datasets([dataset["train"], dataset["test"]])[label_col]))
    assert len(label_set) > 1  # i.e. at least 2 classes
    return label_set


def sparse_encode_label(old_labels: list, label_set) -> np.ndarray:
    # TODO Optimize
    return np.array([label_set.index(l) for l in old_labels])


def convert_to_transformer_ds(dataset: ds.DatasetDict,
                              tokenizer,
                              input_col,
                              label_col
                              ) -> Tuple[TransformersDataset, TransformersDataset]:
    '''
    Takes in a DatasetDict and converts it to a TransformersDataset
    :param dataset:
    :param tokenizer:
    :param input_col: str if single_text classification or tuple if text_pair_classification task
    :param label_col:
    :param use_better_truncation:
    :return:
    '''
    # ============================= #
    # Convert DS to Transformer DS  #
    # ============================= #
    label_set = get_label_set(dataset, label_col)
    if isinstance(input_col, str):  # Single Sentence Classification
        train = BetterTransformersDataset.from_arrays(
            texts=dataset["train"][input_col],
            y=sparse_encode_label(dataset["train"][label_col], label_set),
            tokenizer=tokenizer,
            truncation=512,
            target_labels=label_set
        )
        test = BetterTransformersDataset.from_arrays(
            texts=dataset["test"][input_col],
            y=sparse_encode_label(dataset["test"][label_col], label_set),
            tokenizer=tokenizer,
            truncation=512,
            target_labels=label_set
        )
    else:  # Text Pair Classification
        train = BetterTransformersDataset.from_arrays(
            texts=zip(dataset["train"][input_col[0]], dataset["train"][input_col[1]]),
            y=sparse_encode_label(dataset["train"][label_col], label_set),
            tokenizer=tokenizer,
            truncation=512,
            target_labels=label_set
        )
        test = BetterTransformersDataset.from_arrays(
            texts=zip(dataset["test"][input_col[0]], dataset["test"][input_col[1]]),
            y=sparse_encode_label(dataset["test"][label_col], label_set),
            tokenizer=tokenizer,
            truncation=512,
            target_labels=label_set
        )

    assert np.all(train.y == sparse_encode_label(dataset["train"][label_col], label_set))
    assert np.all(test.y == sparse_encode_label(dataset["test"][label_col], label_set))

    return (train, test)


class BetterTransformersDataset(TransformersDataset):
    @classmethod
    @experimental
    def from_arrays(cls,
                    texts: Iterable[str] | Iterable[Tuple[str, str]],
                    y: np.ndarray | csr_matrix,
                    tokenizer: tr.RobertaTokenizer,
                    target_labels=None,
                    truncation=512):
        """
        Constructs a new TransformersDataset from the given text and label arrays.

        Parameters
        ----------
        texts : list of str or np.ndarray[str] or zip[Tuple[str, str]] (For Sentence Pairs)
            List of text documents.
        y : np.ndarray[int] or scipy.sparse.csr_matrix
            List of labels where each label belongs to the features of the respective row.
            Depending on the type of `y` the resulting dataset will be single-label (`np.ndarray`)
            or multi-label (`scipy.sparse.csr_matrix`).
        tokenizer : tokenizers.Tokenizer
            A huggingface tokenizer.
        target_labels : numpy.ndarray[int] or None, default=None
            List of possible labels. Will be directly passed to the datset constructor.
        truncation : int
            Maximum sequence length in number of tokens

        Returns
        -------
        dataset : TransformersDataset
            A dataset constructed from the given texts and labels.

        .. versionadded:: 1.1.0
        """
        data_out = []

        multi_label = isinstance(y, csr_matrix)
        if multi_label:
            y = csr_to_list(y)

        tokenizer_arguments = {
            "add_special_tokens": True,
            "padding": 'max_length',
            "return_attention_mask": True,
            "return_tensors": 'pt',
            "truncation": 'longest_first',
            "max_length": truncation,
        }
        for i, doc in enumerate(texts):
            if isinstance(doc, tuple):  # Encode Text Pair
                encoded_dict = tokenizer.encode_plus(
                    text=doc[0],
                    text_pair=doc[1],
                    **tokenizer_arguments
                )
            else:  # Encode Single Text
                encoded_dict = tokenizer.encode_plus(
                    text=doc,
                    **tokenizer_arguments,
                )

            # Combine Parts and add sample to final collection
            if multi_label:
                data_out.append((encoded_dict['input_ids'],
                                 encoded_dict['attention_mask'],
                                 np.sort(y[i])))
            else:
                data_out.append((encoded_dict['input_ids'],
                                 encoded_dict['attention_mask'],
                                 y[i]))

        return TransformersDataset(data_out, multi_label=multi_label, target_labels=target_labels)
