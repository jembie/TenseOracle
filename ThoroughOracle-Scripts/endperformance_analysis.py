from pathlib import Path
from collections import defaultdict
from typing import DefaultDict, List, Dict
import numpy as np
import pandas as pd
import constants


def load_asset_data_baseline(ASSET_PATHS: Path) -> DefaultDict[str, List]:
    """
    Load baseline asset data from a directory, calculating mean F1 scores.

    This function traverses the given directory to find files ending with 'f1s.npy',
    loads their content, and calculates the mean value for each task.

    Args:
        ASSET_PATHS (Path): Directory path containing the asset files

    Returns:
        DefaultDict[str, List]: A dictionary mapping task names to lists of mean F1 scores
    """
    TASK_ASSET_MAP = defaultdict(list)
    for path in ASSET_PATHS.glob("**/*"):
        if path.name.endswith("f1s.npy"):
            task_name = path.parent.parent.name
            if task_name != ASSET_PATHS.name:
                data = np.load(path)
                if data.size == 0:
                    print(f"{path} is empty")
                TASK_ASSET_MAP[task_name].append(data.mean())

    return TASK_ASSET_MAP


def collect_asset_paths(ASSET_PATHS: Path) -> DefaultDict[str, List]:
    """
    Collect paths to all asset files ending with '_f1s.npy' from a directory.

    This function traverses the given directory to find files ending with '_f1s.npy'
    and organizes them by task name.

    Args:
        ASSET_PATHS (Path): Directory path to search for asset files

    Returns:
        DefaultDict[str, List]: A dictionary mapping task names to lists of file paths
    """
    TASK_ASSET_MAP = defaultdict(list)
    for path in ASSET_PATHS.glob("**/*"):
        if path.name.endswith("_f1s.npy"):
            task_name = path.parent.parent.name
            if task_name != ASSET_PATHS.name:
                TASK_ASSET_MAP[task_name].append(path)

    return TASK_ASSET_MAP


def load_asset_data(workspace_data: DefaultDict[str, List[Path]]) -> Dict[str, pd.DataFrame]:
    """
    Load and organize asset data from file paths into pandas DataFrames.

    This function processes a collection of file paths, loads the numpy arrays,
    and organizes them by filter strategy name into DataFrames.

    Args:
        workspace_data (DefaultDict[str, List[Path]]): Dictionary mapping task names
                                                      to lists of file paths

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping task names to DataFrames containing
                                the processed data
    """
    results = {}
    for task_name, assets in workspace_data.items():
        collected_dfs = defaultdict(list)

        for asset_path in assets:
            data: np.ndarray = np.load(asset_path)
            if data.size == 0:
                print(f"{asset_path} is empty")

            filter_strategy_name = asset_path.name.replace("_f1s.npy", "")
            filter_strategy_name = filter_strategy_name.removesuffix("Filter")
            collected_dfs[filter_strategy_name].append(data.mean())

        df = pd.DataFrame.from_dict(collected_dfs, orient="index")
        df = df.transpose()
        results[task_name] = df
    return results


def transform_into_experimental_data(data):
    """
    Transform the data structure for experimental analysis.

    This function explodes the DataFrames and concatenates them into a single
    DataFrame for further analysis.

    Args:
        data (Dict[str, pd.DataFrame]): Dictionary of DataFrames to transform

    Returns:
        pd.DataFrame: A single concatenated DataFrame with exploded series
    """
    results = []
    for df in data.values():
        df_transformed = df.apply(pd.Series.explode).reset_index(drop=True)
        results.append(df_transformed)

    return pd.concat(results, axis=0, ignore_index=True)


def baseline(random_sampling: str, prediction_entropy_uncertainty: str, prediction_entropy: str, filter_strategies: str) -> None:
    """
    Calculate and compare baseline metrics for different filtering strategies.

    This function loads data for different sampling and filtering strategies,
    calculates rankings for each strategy across tasks, and prints the final rankings.

    Args:
        random_sampling (str): Directory name for random sampling assets
        prediction_entropy_uncertainty (str): Directory name for prediction entropy uncertainty assets
        prediction_entropy (str): Directory name for prediction entropy assets
        filter_strategies (str): Directory name for filter strategies assets

    Returns:
        None: Results are printed to standard output
    """
    random_baseline_data = load_asset_data_baseline(ASSET_PATHS=Path(constants.BASE_PATH, "cache", "assets", random_sampling))
    prediction_entropy_uncertainty_data = load_asset_data_baseline(
        ASSET_PATHS=Path(constants.BASE_PATH, "cache", "assets", prediction_entropy_uncertainty)
    )
    prediction_entropy_data = load_asset_data_baseline(ASSET_PATHS=Path(constants.BASE_PATH, "cache", "assets", prediction_entropy))
    actual_experiments = load_asset_data(collect_asset_paths(ASSET_PATHS=Path(constants.BASE_PATH, "cache", "assets", filter_strategies)))

    categories_ranking = {
        "Random Sampling": [],
        "Prediction Entropy Uncertainty Clipped": [],
        "Prediction Entropy": [],
        "SimpleDSM": [],
        "SemanticAE": [],
        "SimpleSS": [],
        "HDBScan": [],
        "IsolationForest": [],
        "LocalOutlierFactor": [],
    }

    for task, df in actual_experiments.items():
        df["Random Sampling"] = random_baseline_data[task]
        df["Prediction Entropy Uncertainty Clipped"] = prediction_entropy_uncertainty_data[task]
        df["Prediction Entropy"] = prediction_entropy_data[task]

        df_transformed = df.apply(pd.Series.explode).reset_index(drop=True)
        category_order = df_transformed.mean().sort_values(ascending=False).index.tolist()

        for category in categories_ranking.keys():
            categories_ranking[category].append(category_order.index(category) + 1)

    results = {}
    for category, rankings in categories_ranking.items():
        results[category] = sum(rankings) / len(actual_experiments.keys())

    ranking_df = pd.DataFrame.from_dict(data=[results])
    ranking_df.index = ["Ranking"]
    for col in ranking_df.columns:
        print(f"Rank for {col}: {ranking_df.at['Ranking', col]}")


if __name__ == "__main__":
    """
    Defines experiment parameters and runs the baseline comparison.
    Change the values for each key to match to your project's folder, i.e. inside the `cache/assets` directory.
    """
    kwargs = {
        "random_sampling": "final-experiment-random-sampling-all-filters",
        "prediction_entropy": "final-experiment-prediction-entropy-no-filters",
        "prediction_entropy_uncertainty": "final-experiment-no-filters",
        "filter_strategies": "final-experiment-all-filters",
    }

    baseline(**kwargs)
