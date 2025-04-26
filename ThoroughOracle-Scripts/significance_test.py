import comet_ml as comet
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, DefaultDict, Optional
from collections.abc import Callable
import numpy as np
from collections import defaultdict
import constants
import deepsig
import logging


def load_asset_data(workspace_data: DefaultDict[str, List[Path]]) -> Dict[str, pd.DataFrame]:
    """
    Load asset data from the given workspace directory paths.

    Args:
        workspace_data (DefaultDict[str, List[Path]]): A dictionary mapping task names to lists of asset directory paths.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary mapping task names to concatenated DataFrames containing asset data.
    """
    asset_data = {}
    for task_name, asset_dirs in workspace_data.items():
        collected_assets = []
        for asset_dir in asset_dirs:
            asset_paths = asset_dir.glob("*")
            collected_dfs = []
            for asset_path in asset_paths:
                data = np.load(asset_path)
                df = pd.DataFrame(data={asset_path.name[:-4]: [data]})
                if df.empty:
                    print(f"Careful! For '{task_name}', the '{asset_path}' file is empty.")
                else:
                    collected_dfs.append(df)

            collected_assets.append(pd.concat(collected_dfs, axis=1))

        asset_data[task_name] = pd.concat(collected_assets, axis=0, ignore_index=True)

    return asset_data


def collect_asset_paths(ASSET_PATHS: Optional[Path] = None) -> DefaultDict[str, List]:
    """
    Collect asset directory paths from the given base asset path.

    Args:
        ASSET_PATHS (Optional[Path]): The base path where assets are stored. Defaults to computed path.

    Returns:
        DefaultDict[str, List]: A dictionary mapping task names to lists of directory paths containing assets.
    """
    if not ASSET_PATHS:
        ASSET_PATHS = Path(constants.BASE_PATH, "cache", "assets", constants.COMET_WORKSPACE)

    TASK_ASSET_MAP = defaultdict(list)
    for dir_path in ASSET_PATHS.glob("**/*"):
        if dir_path.is_dir() and dir_path.name.isdigit():
            task_name = dir_path.parent.name
            if task_name != constants.COMET_WORKSPACE:
                TASK_ASSET_MAP[task_name].append(dir_path)

    return TASK_ASSET_MAP


def filter_no_htl(asset_name: str) -> bool:
    """
    This is a helper function for the `create_comparison_df()` function. It is used to compare `No HTL` with `HTL`, so the filter condition looks for assets that contain `_no_htl` in their name.

    Args:
        asset_name (str): The asset to check for.

    Returns:
        bool: Returns `True` if the asset ends with `_no_htl`, else defaults to `False`.
    """
    return asset_name.endswith("_no_htl")


def filter_random(asset_name: str) -> bool:
    """
    This is a helper function for the `create_comparison_df()` function. It is used to compare `Random (Filled Up)` with `HTL`, so the filter condition looks for assets that contain `_random` in their name.

    Args:
        asset_name (str): The asset to check for.

    Returns:
        bool: Returns `True` if the asset ends with `_random`, else defaults to `False`.
    """
    return asset_name.endswith("_random")


def clean_up_asset_name(asset_name: str) -> str:
    """
    Remove suffixes and map to cleaned names.

    This function removes the `_no_htl` and `_random` extension from the asset.
    For example: `AutoFilter_Chen_Like_no_htl` would be transformed into `AutoFilter_Chen_Like`.

    Args:
        asset_name: The name of the asset to be (potentially) changed

    Returns:
        The cleaned asset name
    """
    if asset_name.endswith("_no_htl") or asset_name.endswith("_random"):
        asset_name = asset_name[:-7]

    cleaned_asset = constants.FILTER_NAMES_CLEAN[asset_name]
    return cleaned_asset


def helper_function(
    significance_test_data: Dict[str, pd.DataFrame], filter_condition: Callable[[str], bool]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Process and filter significance test data based on the given condition.

    Args:
        significance_test_data (Dict[str, pd.DataFrame]): The significance test data.
        filter_condition (Callable[[str], bool]): A function to filter asset names.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: A dictionary mapping task names to filtered asset data.
    """
    results = {}
    for task_name, asset_df in significance_test_data.items():
        redundant_columns = [col for col in asset_df.columns if col.endswith("Marked_Samples")]
        asset_df = asset_df.drop(columns=redundant_columns)

        collected_assets = {}
        for col in asset_df.columns:
            if filter_condition(col):
                cleaned_asset = clean_up_asset_name(col)
                flattened_array = np.concatenate(asset_df[col].values)
                if flattened_array.size == 0:
                    print(f"Empty array found for: {col} in {task_name}")
                else:
                    collected_assets[cleaned_asset] = flattened_array
            elif col.endswith("HTL"):
                collected_assets[col] = np.concatenate(asset_df[col].values)

        results[task_name] = collected_assets

    return results


def prepare_significance_test_data(filter_condition: Callable[[str], bool]) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for significance testing.

    Args:
        filter_condition (Callable[[str], bool]): The filtering condition function.

    Returns:
        Dict[str, pd.DataFrame]: Processed data ready for significance testing.
    """
    workspace_data = collect_asset_paths()
    asset_data = load_asset_data(workspace_data=workspace_data)
    return helper_function(asset_data, filter_condition)


def signifance_test(comparison_data: Dict[str, pd.DataFrame], file_name: str):
    """
    Perform significance testing and save results.

    Args:
        comparison_data (Dict[str, pd.DataFrame]): Data for the comparison.
        file_name (str): The output file name for results.
    """
    results = []
    for task_name, assets_dict in comparison_data.items():
        task_aso = []
        htl_data = assets_dict["HTL"]
        for filter_strategy, data in assets_dict.items():
            if filter_strategy != "HTL":
                if (len(data) != constants.SEED_COUNT * 30) or (len(htl_data) != constants.SEED_COUNT * 30):
                    logging.error(
                        f"Error! In {task_name} the filterstrategy '{filter_strategy}' has '{len(data)}' Data, and '{len(htl_data)}' HTL Data but expected were: '{constants.SEED_COUNT * 30}'"
                    )

                better = deepsig.aso(data, htl_data, num_bootstrap_iterations=10_000, num_jobs=-1, seed=42)
                task_aso.append(better)

        assets_dict.pop("HTL", None)
        results.append(pd.DataFrame(data=[task_aso], columns=assets_dict.keys(), index=[f"{task_name}_{file_name}"]))

    df = pd.concat(results)
    df.to_csv(f"./img/minimal-difference/{constants.COMET_WORKSPACE}_{file_name}.csv")


def no_htl_vs_htl():
    return prepare_significance_test_data(filter_no_htl)


def random_vs_htl():
    return prepare_significance_test_data(filter_random)


def visualize_results():
    sns.set_theme()
    sns.xkcd_palette
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))

    random_better = pd.read_csv(
        f"{constants.BASE_PATH}/img/minimal-difference{constants.COMET_WORKSPACE}_random_is_better.csv", index_col=0
    )
    random_better.index = [updated_index.split("_")[0][1::] for updated_index in list(random_better.index)]

    rename_strategies = {
        "Simple SS": "SSE",
        "Semantic AE": "AE",
        "Simple DSM": "DSM",
        "HDBScan": "HDBSCAN",
        "LocalOutlierFactor": "LOF",
        "IsolationForest": "IF",
    }
    random_better = random_better.rename(columns=rename_strategies)
    random_better = random_better[["DSM", "SSE", "LOF", "HDBSCAN", "IF", "AE"]]

    annotations = random_better.copy().astype(str)
    eps_min_threshold = 0.20
    for index in random_better.index:
        for col in random_better.columns:
            value = random_better.at[index, col]
            if value < eps_min_threshold:
                annotations.loc[index, col] = f"$\\bf{{{value:.2f}}}$"
            else:
                annotations.loc[index, col] = round(value, 2)

    colors = sns.color_palette("YlGnBu", as_cmap=True)  # Stairwell-style color palette

    sns.heatmap(
        cmap=colors,
        data=random_better,
        annot=annotations,
        fmt="",
        annot_kws={"size": 20},
        linewidths=0.5,
        linecolor="grey",
        cbar_kws={"shrink": 0.5},
        square=True,
    )
    axes.tick_params(axis="y", rotation=0)
    axes.set_xticklabels(axes.get_xticklabels(), fontsize=15, rotation=45, ha="center")
    axes.set_yticklabels(axes.get_yticklabels(), fontsize=15)
    axes.set_title("Upper bound to the violation ratios ($\\epsilon_{min}$)", fontsize=18)

    plt.savefig(f"{constants.BASE_PATH}/img/minimal-difference/sigtest_random_vs_htl.pdf", format="pdf", dpi=300)


def main():
    logging.basicConfig(
        filename=f"{constants.BASE_PATH}/app.log", filemode="a", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    signifance_test(random_vs_htl(), "random_is_better")
    logging.info("Done with random_is_better")


if __name__ == "__main__":
    # main()
    visualize_results()
