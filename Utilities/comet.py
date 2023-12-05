import os
import comet_ml


class CometExperiment():
    def __init__(self, args, config, task_config):
        self.args = args
        self.config = config
        self.task_config = task_config
        self.workspace = args.comet_workspace
        self.api_key = args.comet_api_key
        self.project_name = cpn if (cpn := args.comet_project_name) is not None else task_config["task_name"]

        if (not self.api_key) != (not self.workspace):
            raise Exception(f"""
            Comet API Key is {'not ' if (not self.api_key) else ''}set,
            but the workspace is {'not ' if (not self.api_key) else ''}.
            If you don't want to use comet please set neither 
            or if you do want to use it set both.
            """)
        if self.api_key:
            self.EXPERIMENT = self.setup_comet()
        else:
            self.EXPERIMENT = None

    def setup_comet(self):
        # 🤗 creates its own experiment if not set to 'DISABLED'
        os.environ['COMET_MODE'] = 'DISABLED'

        experiment = comet_ml.Experiment(
            api_key=self.api_key,
            project_name=self.project_name,
            workspace=self.workspace
        )

        comet_ml.config.set_global_experiment(experiment)

        experiment.add_tag(self.args.filter_strategy_name)

        # Commit Task Name to differentiate Task when all is sent to the same project
        experiment.log_parameter("task", self.task_config["task_name"])
        experiment.log_parameter("config", self.args.experiment_config)
        experiment.log_parameter("seed", self.args.random_seed)
        experiment.log_parameter("strategy_name", self.args.strategy_name)
        experiment.log_parameter("filter_strategy_name", self.args.filter_strategy_name)
        experiment.log_parameter("eval_iter", self.config["SET_EVAL_ITERATIONS"])
        experiment.log_parameter("model", self.config["MODEL_NAME"])

        return experiment

    def log_metrics(self, dictionary: dict):
        if self.EXPERIMENT is None:
            pass
        else:
            self.EXPERIMENT.log_metrics(dictionary)
