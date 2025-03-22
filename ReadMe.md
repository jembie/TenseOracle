# Tense Oracle Project

This project contains the code utilized in the research paper "..."[\<cite>] and is made publicly available to ensure reproducibility.
In this project we motivate a filtering-based approach for mitigating the impact of 'Too Hard to Learn Samples' (i.e., outliers) during Active Learning (AL) training.

It has been shown [\<cite>], that outlier samples provide minimal value to machine learning algorithms, yet they are frequently chosen by AL strategies. AL strategies prioritize labeling instances where the model is currently performing the weakest.
The reason for that is, that the model assumes that it can gain the most information from samples on which it currently performs poorly compared to samples where the model is already performing well.
However, this assumption does not hold for outliers, as these samples often remain 'unlearnable' and thus lead to an inefficient allocation of labeling resources.

## Avoidance of Too Hard to Learn Samples during Active Learning - In a Nutshell

AL is a technique designed to reduce the cost of labeling large datasets for machine learning through selectively labeling only the most informative data samples.
One of the most widely used methods within AL is pool-based uncertainty sampling, which follows an iterative loop:

1. A small subset of labeled data and a large pool of unlabeled data are initialized.
2. A model is trained on the labeled data and then the trained model is used to make predictions on the unlabeled pool.
3. Samples for which the model has the highest uncertainty are chosen, based on the assumption that the model can learn the most from them.
4. These chosen samples are then sent to an oracle (typically a human annotator) for labeling.
5. Then newly labeled data is incorporated into the training set, and the model is retrained.
6. This cycle repeats until the predefined labeling budget is exhausted.

The premise of AL, is, that through this iterative process we achieve a more effective dataset compared to random sampling of labeled instances.

<img src="res/AL-Loop.png" width="400" style="border-radius: 10px;">

However, this methodology starts struggling when the model is confronted with datasets that contain a significant number (>= 5%) of outliers. Since models consistently perform poorly on outliers, they tend to be repeatedly chosen for labeling.Consequently, labeling these samples wastes resources and may even degrade model performance.

To address this issue, we introduce a filtering mechanism that prevents the selection of such outliers for labeling. Our approach integrates a filter capable of vetoing specific samples, ensuring that AL resources are allocated more effectively.

<img src="res/Filtered-AL-Loop.png" width="400" style="border-radius: 10px;">

## Usage Instructions

After Cloning the project one can run a trial via the following command:\

```py
python main.py --task_config Configs/Tasks/<task_config>.json --filter_strategy_name <filter_class_name>
```

_Note:_

- The available filtering strategies for `<filter_class_name>` can be found in the `Strategies` directory. The `Strategies/__init__.py` file contains a list of all available filters.

- If no GPU is available, the `--gpu_optional` flag must be set. Otherwise, execution will terminate immediately, as GPU usage is highly recommended for most datasets.

**Example Execution:**

- with GPU available

```py
python main.py --task_config ./Configs/Tasks/dbpedia.json --filter_strategy_name AutoFilter_Chen_Like
```

- without GPU available

```py
python main.py --task_config ./Configs/Tasks/dbpedia.json --filter_strategy_name AutoFilter_Chen_Like --gpu_optional
```

**Experiment Configuration:**

Most hyperparameters that should remain constant across multiple runs (to ensure comparability) are specified in a separate configuration file. By default, the system utilizes the `./Configs/standard.json` file. If one wants to choose another configuration file, then this can achieved through adding `--experiment_config <config>` in the execution.
