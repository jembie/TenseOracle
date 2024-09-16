from Strategies.AE_filters import (
    AutoFilter_Chen_Like,
    AutoFilter_LSTM,
    AutoFilter_LSTM_SIMPLE,
)
from Strategies.dsm_filters import (
    LoserFilter_Optimized_Pseudo_Labels,
    LoserFilter_Plain,
    LoserFilter_SSL_Variety,
)
from Strategies.filters import RandomFilter
from Strategies.LE_filters import (
    TeachingFilter,
    TeachingFilter_Smooth,
    TeachingFilter_WOW,
)
from Strategies.other_filters import SingleStepEntropy, SingleStepEntropy_SimplePseudo
from Strategies.sklearn_filters import IsolationForestFilter, LocalOutlierFactorFilter, HDBScanFilter

# ======= Only use these filters for experiments =======

# Create synonymous meaning for filters
SimpleDSM = LoserFilter_Plain

# LSTM = Long Short Term Memory, AE = Auto Encoder
SyntacticAE = AutoFilter_LSTM_SIMPLE  # DO NOT USE FOR EVALUATION

SemanticAE = AutoFilter_Chen_Like

SimpleSS = SingleStepEntropy_SimplePseudo
