import os
import comet_ml
import numpy as np
from pathlib import Path


class CometExperiment:
    def __init__(self, args, config, task_config):
        self.args = args
        self.config = config
        self.task_config = task_config
        self.workspace = args.comet_workspace
        self.api_key = args.comet_api_key
        self.project_name = (
            cpn
            if (cpn := args.comet_project_name) is not None
            else task_config["task_name"]
        )
        self.cache_adr = config["CACHE_ADR"]
        self.filter_strategy_name = None

        if (not self.api_key) != (not self.workspace):
            raise Exception(
                f"""
            Comet API Key is {'not ' if (not self.api_key) else ''}set,
            but the workspace is {'not ' if (not self.api_key) else ''}.
            If you don't want to use comet please set neither 
            or if you do want to use it set both.
            """
            )
        if self.api_key:
            self.EXPERIMENT = self.setup_comet()
        else:
            self.EXPERIMENT = None

    def setup_comet(self) -> comet_ml.Experiment:
        # 🤗 creates its own experiment if not set to 'DISABLED'
        os.environ["COMET_MODE"] = "DISABLED"

        experiment = comet_ml.Experiment(
            api_key=self.api_key,
            project_name=self.project_name,
            workspace=self.workspace,
        )

        comet_ml.config.set_global_experiment(experiment)

        self.filter_strategy_name = (
            ' '.join(self.args.filter_strategy_name)
            if isinstance(self.args.filter_strategy_name, list)
            else self.args.filter_strategy_name
        )

        experiment.add_tag(
            self.filter_strategy_name + ("-full-budget" if self.args.use_up_entire_budget else "")
        )
        # Commit Task Name to differentiate Task when all is sent to the same project
        experiment.log_parameter("task", self.task_config["task_name"])
        experiment.log_parameter("config", self.args.experiment_config)
        experiment.log_parameter("seed", self.args.random_seed)
        experiment.log_parameter("strategy_name", self.args.strategy_name)
        experiment.log_parameter("filter_strategy_name", self.filter_strategy_name)
        experiment.log_parameter("eval_iter", self.config["SET_EVAL_ITERATIONS"])
        experiment.log_parameter("model", self.config["MODEL_NAME"])

        return experiment

    def log_metrics(self, dictionary: dict, step=None):
        if self.EXPERIMENT is None:
            pass
        else:
            self.EXPERIMENT.log_metrics(dictionary, step=step)

    def log_parameters(self, dictionary: dict):
        if self.EXPERIMENT is None:
            pass
        else:
            self.EXPERIMENT.log_parameters(dictionary)

    def log_results(self, f1_scores: list, name: str) -> None:
        """
        Converts a list of f1_scores into a numpy array
        Saves this numpy array to a .npy file
        Uploads the file to comet as an asset
        :param f1_scores:
        :return:
        """
        if self.EXPERIMENT is None:
            pass
        else:
            f1_scores = np.array(f1_scores)
            path = Path(f"{self.cache_adr}/{name}.npy")
            np.save(path, f1_scores)
            self.EXPERIMENT.log_asset(path, file_name=f"{name}.npy")

    def log_marked_samples(self, marked_samples: list, name: str) -> None:
        if self.EXPERIMENT is None:
            pass
        else:
            avoided_samples = np.array(marked_samples)
            path = Path(f"{self.cache_adr}/{name}.npy")
            np.save(path, avoided_samples)
            self.EXPERIMENT.log_asset(path, file_name=f"{name}.npy")