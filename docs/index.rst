.. TenseOracle documentation master file, created by
   sphinx-quickstart on Sun Feb 23 15:23:04 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TenseOracle documentation
=========================

This project contains the code utilized in the research paper "..."[\<cite>] and is made publicly available to ensure reproducibility.
In this project we motivate a filtering-based approach for mitigating the impact of 'Too Hard to Learn Samples' (i.e., outliers) during Active Learning (AL) training.

It has been shown [\<cite>], that outlier samples provide minimal value to machine learning algorithms, yet they are frequently chosen by AL strategies. AL strategies prioritize labeling instances where the model is currently performing the weakest.
The reason for that is, that the model assumes that it can gain the most information from samples on which it currently performs poorly compared to samples where the model is already performing well.
However, this assumption does not hold for outliers, as these samples often remain 'unlearnable' and thus lead to an inefficient allocation of labeling resources.


.. toctree::
   :maxdepth: 2
   :caption: Project Analysis Code:

   modules
   usage