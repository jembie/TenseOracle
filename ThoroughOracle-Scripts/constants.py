"""
Constants Module for Outlier Detection Experiment Framework

This module contains all configuration constants and mappings used throughout the outlier detection
experiments. It defines experiment settings, filter strategy names, metrics, parameters, task names,
and file paths essential for running experiments and processing results.

Constants include:
- Experiment configuration (workspace, flags, seed count)
- Filter strategy definitions and mappings
- Metrics configuration with expected types
- Parameter definitions
- Dataset task names and their configurations
- File paths for accessing configuration files and data

This centralized approach ensures consistency across the experimental framework and simplifies
configuration management.
"""

from pathlib import Path

# Experiment configuration
COMET_WORKSPACE: str = "outlier-detection"
FINAL_EXPERIMENT_FLAG: bool = False
SEED_COUNT: int = 20

# Filter strategy configurations
FILTER_STRATEGY_NAMES = ["HDBScanFilter LocalOutlierFactorFilter IsolationForestFilter SimpleDSM SemanticAE SimpleSS"]
FILTER_STRATEGY_NAMES_FINAL = [
    "HDBScanFilter",
    "LocalOutlierFactorFilter",
    "IsolationForestFilter",
    "SimpleDSM",
    "SemanticAE",
    "SimpleSS",
]

# Clean names mapping for filter strategies
FILTER_NAMES_CLEAN = {
    "LoserFilter_Plain": "Simple DSM",
    "AutoFilter_Chen_Like": "Semantic AE",
    "SingleStepEntropy_SimplePseudo": "Simple SS",
    "HDBScanFilter": "HDBScan",
    "IsolationForestFilter": "IsolationForest",
    "LocalOutlierFactorFilter": "LocalOutlierFactor",
}

# Metric definitions with their expected types
METRICS = {
    "AutoFilter_Chen_Like_HTL Count": float,
    "AutoFilter_Chen_Like_avg_duration": float,
    "AutoFilter_Chen_Like_medF1 (No HTL)": float,
    "AutoFilter_Chen_Like_medF1 (With HTL)": float,
    "AutoFilter_Chen_Like_avgF1 (random replacement)": float,
    "AutoFilter_Chen_Like_avgF1 (No HTL)": float,
    "AutoFilter_Chen_Like_avgF1 (With HTL)": float,
    "AutoFilter_Chen_Like_medF1 (random replacement)": float,
    ###############################################################
    "HDBScanFilter_HTL Count": float,
    "HDBScanFilter_avg_duration": float,
    "HDBScanFilter_medF1 (No HTL)": float,
    "HDBScanFilter_medF1 (With HTL)": float,
    "HDBScanFilter_medF1 (random replacement)": float,
    "HDBScanFilter_avgF1 (No HTL)": float,
    "HDBScanFilter_avgF1 (With HTL)": float,
    "HDBScanFilter_avgF1 (random replacement)": float,
    ###############################################################
    "IsolationForestFilter_HTL Count": float,
    "IsolationForestFilter_avg_duration": float,
    "IsolationForestFilter_avgF1 (No HTL)": float,
    "IsolationForestFilter_avgF1 (With HTL)": float,
    "IsolationForestFilter_avgF1 (random replacement)": float,
    "IsolationForestFilter_medF1 (No HTL)": float,
    "IsolationForestFilter_medF1 (With HTL)": float,
    "IsolationForestFilter_medF1 (random replacement)": float,
    ###############################################################
    "LocalOutlierFactorFilter_HTL Count": float,
    "LocalOutlierFactorFilter_avg_duration": float,
    "LocalOutlierFactorFilter_avgF1 (No HTL)": float,
    "LocalOutlierFactorFilter_avgF1 (With HTL)": float,
    "LocalOutlierFactorFilter_avgF1 (random replacement)": float,
    "LocalOutlierFactorFilter_medF1 (No HTL)": float,
    "LocalOutlierFactorFilter_medF1 (With HTL)": float,
    "LocalOutlierFactorFilter_medF1 (random replacement)": float,
    ###############################################################
    "LoserFilter_Plain_HTL Count": float,
    "LoserFilter_Plain_avg_duration": float,
    "LoserFilter_Plain_avgF1 (No HTL)": float,
    "LoserFilter_Plain_avgF1 (With HTL)": float,
    "LoserFilter_Plain_avgF1 (random replacement)": float,
    "LoserFilter_Plain_medF1 (No HTL)": float,
    "LoserFilter_Plain_medF1 (With HTL)": float,
    "LoserFilter_Plain_medF1 (random replacement)": float,
    ###############################################################
    "SingleStepEntropy_SimplePseudo_HTL Count": float,
    "SingleStepEntropy_SimplePseudo_avg_duration": float,
    "SingleStepEntropy_SimplePseudo_avgF1 (No HTL)": float,
    "SingleStepEntropy_SimplePseudo_avgF1 (With HTL)": float,
    "SingleStepEntropy_SimplePseudo_avgF1 (random replacement)": float,
    "SingleStepEntropy_SimplePseudo_medF1 (No HTL)": float,
    "SingleStepEntropy_SimplePseudo_medF1 (With HTL)": float,
    "SingleStepEntropy_SimplePseudo_medF1 (random replacement)": float,
}

# Parameter definitions with their expected types
PARAMETERS = {
    "strategy_name": str,
    "filter_strategy_name": str,
    "seed": int,
    "task": str,
}

# Base task names for experiments
TASK_NAMES = [
    "ag-news",
    "dbpedia",
    "fnc1",
    "imdb",
    "qnli",
    "rotten-tomatoes",
    "sst2",
    "trec",
    "wiki-talk",
]

# Version prefix for task names
_version = "x"
TASK_NAMES = [_version + t for t in TASK_NAMES]

# Path configurations
BASE_PATH = Path(__file__).parent
CONFIGS_PATH = BASE_PATH.parent / "Configs" / "Tasks"

# Task configuration file mappings
TASK_CONFIGS = {
    "xAG News": CONFIGS_PATH / "ag_news.json",
    "xDBPedia": CONFIGS_PATH / "dbpedia.json",
    "xFNC1": CONFIGS_PATH / "fnc_one.json",
    "xIMDB": CONFIGS_PATH / "imdb.json",
    "xQNLI": CONFIGS_PATH / "qnli.json",
    "xRotten Tomatoes": CONFIGS_PATH / "rotten_tomatoes.json",
    "xSST2": CONFIGS_PATH / "sst2.json",
    "xTREC": CONFIGS_PATH / "trec.json",
    "xWiki Talk": CONFIGS_PATH / "wiki_talk.json",
}
