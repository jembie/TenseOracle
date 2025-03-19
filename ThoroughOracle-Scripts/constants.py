from pathlib import Path

COMET_WORKSPACE: str = "final-experiment-all-filters"
FINAL_EXPERIMENT_FLAG: bool = False
SEED_COUNT: int = 20

FILTER_STRATEGY_NAMES = ["HDBScanFilter LocalOutlierFactorFilter IsolationForestFilter SimpleDSM SemanticAE SimpleSS"]
FILTER_STRATEGY_NAMES_FINAL = [
    "HDBScanFilter",
    "LocalOutlierFactorFilter",
    "IsolationForestFilter",
    "SimpleDSM",
    "SemanticAE",
    "SimpleSS",
]


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

PARAMETERS = {
    "strategy_name": str,
    "filter_strategy_name": str,
    "seed": int,
    "task": str,
}

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

_version = "x"
TASK_NAMES = [_version + t for t in TASK_NAMES]


BASE_PATH = Path(__file__).parent
CONFIGS_PATH = BASE_PATH.parent / "Configs" / "Tasks"

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
