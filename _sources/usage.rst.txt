Getting Started
===============

This page is organized into four sections. We begin by outlining the experimental setup and deployment prerequisites. Next, we present the **Minimal Differences** and **Endperformance** experiments, including interpretations of the results.
Lastly, we offer a guide on for extending this framework on strategies and datasets which have not been convered by us.


.. admonition:: Caution
    :class: caution

    We have conducted our experiments on Python version 3.10.4; newer versions of Python might run this framework without complications. However we cannot guarantee that it will due to dependencies between the external libraries.
    Additionally, it is **heavily** recommended to have a CUDA-compatible GPU available, else the computation would take even longer than it already does.


General Setup
-------------
This section provides an overview of the essential preparations which are needed to execute the framework, as well as a description on how the codebase is structured.

First, assure that you create a Comet account, for details on how to do that see: |comet_link|.

The Comet account is needed as we have used this *(by the time of this writing)* free and open-source variant for logging and storing experimental measurements.

.. |comet_link| raw:: html

    <a href="https://www.comet.com/login" target="_blank">Comet Login</a>

Cloning the Repository
^^^^^^^^^^^^^^^^^^^^^^

Next, clone our repository from the following domain:

.. code-block:: bash

    git clone https://github.com/JP-SystemsX/TenseOracle.git

After cloning you should find the **TenseOracle** folder. Change into that directory within your terminal and then execute the following command inside the terminal to list all existing branches:

.. code-block:: bash

    git branch -a

If the cloning was successful, then you should have the following output:

.. code-block:: bash

    * master
    remotes/origin/HEAD -> origin/master
    remotes/origin/Rainbow-Analysis
    remotes/origin/master
    remotes/origin/standard-analysis

Setting up the Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming that the cloning was successful, we now need to setup a virtual environment and install the required external libraries. This can be done as follows:

.. code-block:: bash

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

.. note::
    For deploying our framework we always assume that you are within the root directory of the project, which means one should have the following folder structure within the terminal before deployment:

    .. code-block:: bash

        .
        ├── Configs/
        ├── Datasets/
        ├── Strategies/
        ├── ThoroughOracle-Scripts/
        ├── Utilities/
        ├── docs/
        ...
        └── main.py


Hyperparameter Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most hyperparameters that should remain constant across multiple runs (to ensure comparability) are specified in a separate configuration file. 
By default, the system utilizes the `Configs/standard.json` file. If one wants to choose another configuration file, then this can achieved through adding the ``--experiment_config`` argument when executing.


.. |br| raw:: html

   <br />

.. admonition:: Important
    :class: danger

    Altering the `Configs/standard.json` file would result in alternation of the experimental results and thus any results cannot be compared with our findings.



Minimal Difference Experiment
-----------------------------

In this section we assume that you reside within the **master**'s branch, as the other branch has slight modifications and thus cannot used for the analysis of this experiment.

<Short explanation of what the experiment does>

Now, in order to execute the code one


.. code-block:: bash
    :linenos:
    :emphasize-lines: 5

    python main.py \
        --task_config ./Configs/Tasks/rotten_tomatoes.json \
        --experiment_config ./Configs/standard.json \
        --filter_strategy_name LocalOutlierFactorFilter HDBScanFilter IsolationForestFilter SimpleSS SimpleDSM SemanticAE \
        --comet_api_key COMET_KEY  \ # Replace COMET_KEY with your actual API Key
        --comet_workspace COMET_WORKSPACE # Replace COMET_WORKSPACE with the name of your comet workspace

.. admonition:: Hint
    :class: hint

    The available filtering strategies that work with the ``--filter_strategy_name`` argument can be found inside the ``Strategies/`` directory;
    viewing the ``__init__.py`` file within that directory lists all available filters strategies.
    If no GPU is available, the ``--gpu_optional`` flag **must be set**. If not set, then the execution will terminate immediately.
    GPU usage is **highly recommended** for analysis.


Endperformance Experiment
-------------------------

...

.. code-block:: bash

    python main.py --task_config ./Configs/Tasks/<task_config>.json --filter_strategy_name <filter_class_name>



Example Execution
-----------------

- with GPU available

.. code-block:: bash

    python main.py --task_config ./Configs/Tasks/dbpedia.json --filter_strategy_name AutoFilter_Chen_Like


- without GPU available
