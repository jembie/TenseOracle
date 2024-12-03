import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        A script to test and evaluate Filter Strategies, 
        to avoid querying Outliers also called (Too) Hard To Learn (HTL) Samples
        during Active Learning.
        HTL samples waste the time of the Oracle (Human Expert) 
        and often even harm performance of models that try to learn them
        """)

    parser.add_argument("--task_config",
                        type=str,
                        required=True,
                        help="Path to the Config File for the demanded Task"
                        )

    parser.add_argument("--experiment_config",
                        type=str,
                        default="./Configs/standard.json",
                        help="""
                        NOTE: You might wanna keep the Default Here to ALLOW Consistency
                        Path to a Config File with Parameters 
                        that should be consistent btwn. experiments e.g. MODEL_NAME, BUDGET, etc.
                        """)

    parser.add_argument(
        '--strategy_name',
        type=str,
        default="PredictionEntropy",
        help='Name of the active learning strategy to employ'
    )

    parser.add_argument(
        '--filter_strategy_name',
        type=str,
        default="None",
        help='Name of the Filter Strategy to employ'
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='''The Seed That shall be used during the experiments to make them more comparable'''
    )

    parser.add_argument(
        '--dom_test',
        action="store_true",
        help='''
                A Strategy dominates another if it is better at any point in time than the other.
                If set the model will be evaluated on the test set after each iteration
                The test set is usually much larger then train set therefore that will 
                take a while.
                '''
    )

    parser.add_argument(
        '--use_up_entire_budget',
        action="store_true",
        help='''Add some extra iterations at end to use up entire saved budget'''
    )

    parser.add_argument(
        '--gpu_optional',
        action="store_true",
        help="If set then don't raise exception if no GPU found"
    )

    parser.add_argument('--comet_api_key',
                        type=str,
                        required=False,
                        help='''Your API-Key to push to comet.ml''')

    parser.add_argument('--comet_project_name',
                        type=str,
                        required=False,
                        help='The name of the Comet project if none given loads from task config.')

    parser.add_argument('--comet_workspace',
                        type=str,
                        required=False,
                        help='The name of the Comet workspace.')

    return parser.parse_args()


def parse_config(args):
    """
    Loads Experiment Config from File to keep HP consistent btwn. Experiments
    and changes Cache address to avoid interference btwn. experiments
    :param args: CLI arguments
    :return:
    """
    with open(Path(args.experiment_config)) as file:
        config = json.load(file)

    # Create An Experiment Specific Cache to limit interference
    config["CACHE_ADR"] = f'{config["CACHE_ADR"]}/{args.comet_workspace}/{args.strategy_name}_{args.random_seed}'
    Path(config["CACHE_ADR"]).mkdir(parents=True, exist_ok=True)
    return config


def parse_task_config(args):
    """
    Loads Task Config from File to load task specific HPs
    e.g. Address of the Dataset,
    does it have a Test set or do we need to create one, etc.
    :param args: CLI arguments
    :return:
    """
    with open(Path(args.task_config)) as file:
        config = json.load(file)

    return config
