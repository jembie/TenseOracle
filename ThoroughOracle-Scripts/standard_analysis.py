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


class StandardExperiment:
    """
    Class for analyzing experimental data, comparing filtered vs unfiltered approaches.
    Processes asset data, calculates statistics, and generates visualization heatmaps.
    """

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
            asset_paths = Path(constants.BASE_PATH, "cache", "assets", constants.COMET_WORKSPACE)

        task_asset_map = defaultdict(list)
        for dir_path in asset_paths.glob("**/*"):
            if dir_path.is_dir() and dir_path.name.isdigit():
                task_name = dir_path.parent.name
                if task_name != COMET_WORKSPACE:
                    task_asset_map[task_name].append(dir_path)

        return task_asset_map

    def load_asset_data(self, workspace_data: DefaultDict[str, List[Path]]) -> Dict[str, pd.DataFrame]:
        """
        Load asset data from files into pandas DataFrames.

        Args:
            workspace_data: Dictionary mapping task names to lists of asset directory paths.

        Returns:
            Dictionary mapping task names to DataFrames containing loaded asset data.
        """
        asset_data = {}
        for task_name, asset_dirs in workspace_data.items():
            collected_assets = []
            for asset_dir in asset_dirs:
                asset_paths = asset_dir.glob("*")
                collected_dfs = []
                for asset_path in asset_paths:
                    data = np.load(asset_path)
                    # Remove .npy extension from name
                    column_name = asset_path.name[:-4]
                    df = pd.DataFrame(data={column_name: [data]})
                    if df.empty:
                        print(f"Careful! For '{task_name}', the '{asset_path}' file is empty. Not appending the file to results.")
                    else:
                        collected_dfs.append(df)

                if collected_dfs:
                    collected_assets.append(pd.concat(collected_dfs, axis=1))

            if collected_assets:
                asset_data[task_name] = pd.concat(collected_assets, axis=0, ignore_index=True)

        return asset_data

    def clean_up_asset_name(self, asset_name: str) -> str:
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

    def calculate_averages(self, asset_data: Dict[str, pd.DataFrame]) -> DefaultDict[str, DefaultDict[str, np.float64]]:
        """
        Calculate average values for each asset across all samples.

        Args:
            asset_data: Dictionary mapping task names to DataFrames with asset data

        Returns:
            Nested dictionary with task names, asset names, and their average values
        """
        summarised_data = defaultdict(lambda: defaultdict(dict))

        for task_name, asset_df in asset_data.items():
            # Remove Marked_Samples from this Analysis
            redundant_columns = [col for col in asset_df.columns if col.endswith("Marked_Samples")]
            asset_df = asset_df.drop(columns=redundant_columns)

            for asset_name in asset_df.columns:
                arrays_for_col = asset_df[asset_name].values
                flattened_array = np.concatenate(arrays_for_col)
                if flattened_array.size == 0:
                    print(f"Empty array found for: {asset_name} in {task_name}")
                else:
                    mean = flattened_array.mean()
                    summarised_data[task_name][asset_name] = mean

        return summarised_data

    def prepare_data(self):
        """
        Load and prepare data for analysis.

        Returns:
            Summarized data with average values for each asset and task
        """
        workspace_data = self.collect_asset_paths()
        asset_data = self.load_asset_data(workspace_data=workspace_data)
        summarised_data = self.calculate_averages(asset_data)
        return summarised_data

    def transform_into_mean_difference(self, asset_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to show percentage differences relative to HTL baseline.

        Args:
            asset_df: DataFrame containing asset values

        Returns:
            DataFrame with values transformed to percentage differences
        """
        for col in asset_df.columns:
            if col != "HTL":
                asset_df[col] = (asset_df[col] - asset_df["HTL"]).mul(100)

        return asset_df

    def filter_no_htl(self, asset_name: str) -> bool:
        """
        Check if an asset name has the '_no_htl' suffix.

        Used to compare `No HTL` with `HTL`.

        Args:
            asset_name: The asset name to check

        Returns:
            True if the asset ends with `_no_htl`, False otherwise
        """
        return asset_name.endswith("_no_htl")

    def filter_random(self, asset_name: str) -> bool:
        """
        Check if an asset name has the '_random' suffix.

        Used to compare `Random (Filled Up)` with `HTL`.

        Args:
            asset_name: The asset name to check

        Returns:
            True if the asset ends with `_random`, False otherwise
        """
        return asset_name.endswith("_random")

    def create_comparison_df(
        self,
        data: DefaultDict[str, DefaultDict[str, np.float64]],
        filter_condition: Callable[[str], bool],
    ) -> pd.DataFrame:
        """
        Create a DataFrame comparing filtered vs unfiltered approaches.

        Args:
            data: Nested dictionary with task names, asset names, and their values
            filter_condition: Function to determine which assets to include

        Returns:
            DataFrame with mean F1-Score differences between filtered and unfiltered approaches
        """
        dfs = []
        for task_name, asset_dict in data.items():
            collected_assets = {}
            for asset_name, asset_value in asset_dict.items():
                if filter_condition(asset_name):
                    cleaned_asset = self.clean_up_asset_name(asset_name)
                    collected_assets[cleaned_asset] = asset_value
                elif asset_name.endswith("HTL"):
                    collected_assets[asset_name] = asset_value

            if collected_assets:
                asset_df = pd.DataFrame(data=collected_assets, index=[task_name])
                asset_df = self.transform_into_mean_difference(asset_df=asset_df)
                dfs.append(asset_df)

        if dfs:
            merged_df = pd.concat(dfs)
            return merged_df
        else:
            return pd.DataFrame()

    def create_heatmap(self, data: pd.DataFrame, ax, title: str):
        """
        Create a heatmap visualization of comparison data.

        Args:
            data: DataFrame containing comparison data
            ax: Matplotlib axis to plot on
            title: Title for the heatmap
        """
        sns.heatmap(
            data=data.drop(columns="HTL", errors="ignore"),
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},  # Font size for annotations
            linewidths=0.5,
            linecolor="grey",  # Grey borders to define the stairwell look
            cbar_kws={"shrink": 0.5},  # Shrink color bar for fitting
            square=True,
            ax=ax,
        )
        ax.set_title(title)

    def save_visualization(self, filename: str, format: str = "pdf", dpi: int = 300):
        """
        Save visualization to file.

        Args:
            filename: Base filename for the output
            format: File format (pdf, png, etc.)
            dpi: Resolution in dots per inch
        """
        output_path = Path(constants.BASE_PATH) / "img" / "minimal-difference" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path, format=format, dpi=dpi)
        print(f"Saved plot at: {output_path}")

    def run(self):
        """
        Main execution method. Prepares data, creates visualizations, and saves results.
        """
        summarised_data = self.prepare_data()

        data_filtered = self.create_comparison_df(summarised_data, self.filter_no_htl)
        data_unfiltered = self.create_comparison_df(summarised_data, self.filter_random)

        sns.set_theme()
        fig, axes = plt.subplots(1, 2, figsize=(17, 11))

        self.create_heatmap(data_filtered, axes[0], "Filtered vs Unfiltered")
        self.create_heatmap(data_unfiltered, axes[1], "Random(Filled Up) vs Unfiltered")

        self.save_visualization("filtered_vs_unfiltered_vs_random2.pdf")


if __name__ == "__main__":
    experiment = StandardExperiment()
    experiment.run()
