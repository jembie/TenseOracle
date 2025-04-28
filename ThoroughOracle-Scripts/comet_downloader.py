import comet_ml as comet
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv, find_dotenv
import constants
from constants import (
    METRICS,
    TASK_NAMES,
    BASE_PATH,
    COMET_WORKSPACE,
)
from utils.bcolors import bcolors

import concurrent.futures


class DownloadCometData:
    """Downloads the data from Comet experiments"""

    def extract_used_metrics(self, experiment_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract a list of unique metrics from the given experiment metrics that are also present in the `metrics` list.

        Parameters
            experiment_metrics : List[Dict]
                A list of dictionaries where each dictionary represents a metric with various attributes, including `metricName`.

        Returns
            List[dict]
                A list of unique metric dictionaries where `metricName` exists in the global `metrics` list.
        """
        metrics_used: List[Dict] = []
        for metric in experiment_metrics:
            if metric["metricName"] in METRICS:
                metrics_used.append(metric)
        return metrics_used

    def extract_paremeter_value(self, parameters_used: List[Dict[str, Any]], parameter_name: str) -> str | float:
        """
        Extracts the current value of a specified parameter from a list of parameters.

        Parameters
            parameters_used : list of dict
                A list of dictionaries containing parameter information.
            parameter_name : str
                The name of the parameter to extract.

        Returns
            `str | float`
                The current value of the specified parameter.
        """
        parameters_dict = [entry for entry in parameters_used if entry.get("name") == parameter_name]
        return parameters_dict[0]["valueCurrent"]

    def load_experiment_data(self, experiment: comet.APIExperiment, endperformance_experiment: Optional[bool] = False) -> None:
        """
        Loads and organizes experiment data, including metrics, parameters, and assets.

        Parameters
            experiment : comet.APIExperiment
                A Comet APIExperiment object containing the data from the experiment.
            endperformance_experiment : bool, optional, default: `False`
                If `True`, includes filter strategy information in the data extraction.

        Returns
            `None`
        """
        experiment_parameters = experiment.get_parameters_summary()
        task = self.extract_paremeter_value(experiment_parameters, "task")
        seed = self.extract_paremeter_value(experiment_parameters, "seed")

        filter_strategy_name = self.extract_paremeter_value(experiment_parameters, "filter_strategy_name")
        # If there was no strategy used, the default return is 'None', we then update it to be '' so naming of assets becomes (e.g.) durations.npy instead of None_durations.npy
        filter_strategy_name = "" if filter_strategy_name == "None" else filter_strategy_name
        if not filter_strategy_name and not endperformance_experiment:
            error_message = bcolors.fail(
                f"ERROR! Attempted extracting filter_strategy_name returned '{filter_strategy_name}' "
                f"from the current workspace ('{constants.COMET_WORKSPACE}') in {task}.\n"
                f"Perhaps you forgot to specify the correct 'COMET_WORKSPACE' or forgot to set "
                f"{bcolors.bold('endperformance_experiment to True?')}\n"
                f"{bcolors.fail('Aborting...')}\n"
            )
            raise AttributeError(error_message)

        kwargs = {"task": task, "seed": seed, "filter_strategy_name": filter_strategy_name}

        self.download_assets(experiment, **kwargs)

    def download_assets(self, experiment: comet.APIExperiment, task: str, seed: str, filter_strategy_name: Optional[str] = "") -> None:
        """
        Downloads and saves the assets of an experiment, filtering out unnecessary files.

        Parameters
            experiment : comet.APIExperiment
                A Comet APIExperiment object containing the experiment data.
            task : str
                The task name associated with the experiment.
            seed : str
                The seed value associated with the experiment.
            filter_strategy_name : str, optional, default: ""
                The filter strategy name, if applicable.

        Returns
            `None`
        """
        assets = experiment.get_asset_list()
        filtered_assets = [asset for asset in assets if not asset["fileName"].endswith(".py")]

        asset_ids = []
        for asset in filtered_assets:
            asset_ids.append((asset["fileName"], asset["assetId"]))

        for file_name, idx in asset_ids:
            asset_data = experiment.get_asset(idx)

            if filter_strategy_name:
                asset_path = Path(f"./{BASE_PATH.name}/cache/assets/{COMET_WORKSPACE}/{task}/{seed}/{filter_strategy_name}_{file_name}")
            else:
                asset_path = Path(f"./{BASE_PATH.name}/cache/assets/{COMET_WORKSPACE}/{task}/{seed}/{file_name}")

            asset_path.parent.mkdir(parents=True, exist_ok=True)
            with open(asset_path, "wb") as f:
                f.write(asset_data)

    def download_workspace_data(self, task_name: str, endperformance_experiment: Optional[bool] = False) -> None:
        """
        Loads experiment data for a specific project from the Comet workspace.

        Parameters
            task_name : str
                The name of the task to load data from.
            endperformance_experiment : bool, optional, default: False
                If `True`, indicates that filter strategies were used during the experiment.

        Returns
            `None`
        """
        experiments = API.get(workspace=COMET_WORKSPACE, project_name=task_name)
        for exp in experiments:
            self.load_experiment_data(exp, endperformance_experiment=endperformance_experiment)

    def get_data(self, endperformance_experiment: bool) -> None:
        """
        Starts the process of downloading the workspace task data for the tasks defined in `constants.COMET_WORKSPACE` and `constants.TASK_NAMES`.

        Parameters
            endperformance_experiment : bool
                Must be set to either `True` or `False`. `True` indicates that the endperformance experiment was performed. `False` indicates that we are parsing the minimal difference experiment.

        Returns
            `None`
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_download_workspace_data = {
                executor.submit(self.download_workspace_data, task_name, endperformance_experiment=endperformance_experiment): task_name
                for task_name in TASK_NAMES
            }
            with tqdm(total=len(TASK_NAMES), desc="Downloading task data...", unit="task") as pbar:
                for future in concurrent.futures.as_completed(future_to_download_workspace_data):
                    task_name = future_to_download_workspace_data[future]
                    try:
                        future.result()
                    except AttributeError as error:
                        raise error
                    else:
                        print(bcolors.ok(f"Download successfully completed for '{task_name}'"))
                    finally:
                        pbar.update()


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    API = comet.API()
    experimental_data = DownloadCometData()
    experimental_data.get_data(endperformance_experiment=False)
