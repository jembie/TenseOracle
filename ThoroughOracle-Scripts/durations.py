import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, DefaultDict, Optional, Callable
import numpy as np
from collections import defaultdict
import constants
from constants import COMET_WORKSPACE
import deepsig
import logging


class DurationMeasure:
    """
    Class for analyzing experimental data, comparing filtered vs unfiltered approaches.
    Processes asset data, calculates statistics, and generates visualization heatmaps.
    """

    rename = {
        "SingleStepEntropySimplePseudo": "SSE",
        "AutoFilterChenLike": "AE",
        "LoserFilterPlain": "DSM",
        "HDBScanFilter": "HDBSCAN",
        "LocalOutlierFactorFilter": "LOF",
        "IsolationForestFilter": "IF",
    }

    def clean_asset_name(self, asset_name: str) -> str:
        return self.rename["".join(asset_name.split("_")[2::])[:-4]]

    def collect_asset_paths(self, asset_paths: Optional[Path] = None) -> DefaultDict[str, List]:
        """
        Collect asset paths organized by task name.

        Args:
            asset_paths: Path to the directory containing assets.
                         Defaults to BASE_PATH/cache/assets/COMET_WORKSPACE.

        Returns:
            A dictionary mapping task names to lists of asset paths.
        """
        if asset_paths is None:
            asset_paths = Path(constants.BASE_PATH, "cache", "assets", "")

        TASK_ASSET_MAP = defaultdict(list)
        for path in asset_paths.glob("**/*"):
            if path.name.endswith(".npy") and "durations" in path.name:
                strategy_name = self.clean_asset_name(path.name)
                if strategy_name != asset_paths.name:
                    data = np.load(path)
                    TASK_ASSET_MAP[strategy_name].append(data)

        return TASK_ASSET_MAP

    def calculate_median_duration(self, asset_data: Dict[str, pd.DataFrame]) -> DefaultDict[str, DefaultDict[str, np.float64]]:
        """
        Calculate average values for each asset across all samples.

        Args:
            asset_data: Dictionary mapping task names to DataFrames with asset data

        Returns:
            Nested dictionary with task names, asset names, and their average values
        """
        results = {}
        for strategy, durations in asset_data.items():
            results[strategy] = np.median(durations)

        return results

    def prepare_data(self):
        """
        Load and prepare data for analysis.

        Returns:
            Summarized data with average values for each asset and task
        """
        workspace_data = self.collect_asset_paths()
        return workspace_data

    def run(self, minimal_diff: Optional[bool] = False):
        """
        Main execution method. Prepares data, creates visualizations, and saves results.
        """
        summarised_data = self.prepare_data()

        median_durations = self.calculate_median_duration(summarised_data)

        if minimal_diff:
            print("========= Experiment Minimal Difference =========")
        else:
            print("========= Experiment Endperformance =========")

        for strategy, median_duration in median_durations.items():
            print(f"{strategy} has: {median_duration:.5f}s median execution time")


if __name__ == "__main__":
    experiment = DurationMeasure()
    experiment.run()
