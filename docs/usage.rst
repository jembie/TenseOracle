Getting Started
===============

This page is organized into four sections. We begin by outlining the experimental setup and deployment prerequisites. Next, we present the **Minimal Differences** and **Endperformance** experiments, including interpretations of the results.
Lastly, we offer a guide on for extending this framework on strategies and datasets which have not been convered by us.


.. admonition:: Important: Read Me
    :class: important

    We have conducted our experiments on Python version 3.10.4; newer versions of Python might run this framework without complications. However we cannot guarantee that it will due to dependencies between the external libraries.
    Additionally, it is **heavily** recommended to have a CUDA-compatible GPU available, else the computation would take even longer than it already does.


General Setup
-------------

In this section we detail the preliminary steps needed for deploying our framework. We begin by detailing the setup configuration and then give examples on deployment.

Firstly, clone our repository from the following domain:

.. code-block:: bash

    git clone https://github.com/JP-SystemsX/TenseOracle.git

After cloning you should find the **TenseOracle** folder. Change into that directory within your terminal and then execute the following command to list all existing branches:

.. code-block:: bash

    git branch -a

If the clone was successful, then you should have the following output:

.. code-block:: bash

    * master
    remotes/origin/HEAD -> origin/master
    remotes/origin/Rainbow-Analysis
    remotes/origin/master
    remotes/origin/standard-analysis

Assuming that the cloning was successful, we now need to setup a virtual environment and install the required external libraries. This can be done as follows:

.. code-block:: bash

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

.. note::
    For deploying our framework we always assume that you are within the root directory of the project, which means one has the following folder structure:

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

.. admonition:: Note
    :class: note

    The available filtering strategies for ``<filter_class_name>`` can be found in the ``Strategies`` directory.
    The ``Strategies/__init__.py`` file contains a list of all available filters.
    If no GPU is available, the ``--gpu_optional`` flag must be set. Otherwise, execution will terminate immediately,
    because GPU usage is **highly recommended** for most datasets.


Endperformance Experiment
-------------------------

...

.. code-block:: bash

    python main.py --task_config ./Configs/Tasks/<task_config>.json --filter_strategy_name <filter_class_name>



Configurations
--------------

Most hyperparameters that should remain constant across multiple runs (to ensure comparability) are specified in a separate configuration file. 
By default, the system utilizes the `./Configs/standard.json` file. If one wants to choose another configuration file, then this can achieved through adding `--experiment_config <config>` in the execution.



Example Execution
-----------------

.. note::
    These examples assume that you are in the root directory of the project, which means one has the following folder structure:

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


- with GPU available

.. code-block:: bash

    python main.py --task_config ./Configs/Tasks/dbpedia.json --filter_strategy_name AutoFilter_Chen_Like


- without GPU available
