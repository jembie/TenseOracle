Getting Started
=====

After Cloning the project one can run a trial via the following command:


.. code-block:: python
    python main.py --task_config ./Configs/Tasks/<task_config>.json --filter_strategy_name <filter_class_name>

.. important::

   The available filtering strategies for ``<filter_class_name>`` can be found in the ``Strategies`` directory.
   The ``Strategies/__init__.py`` file contains a list of all available filters.
   If no GPU is available, the ``--gpu_optional`` flag must be set. Otherwise, execution will terminate immediately,
   as GPU usage is highly recommended for most datasets.


Example Execution
-----------------

- with GPU available
.. code-block:: python
    python main.py --task_config ./Configs/Tasks/dbpedia.json --filter_strategy_name AutoFilter_Chen_Like


- without GPU available
.. code-block:: python
    python main.py --task_config ./Configs/Tasks/dbpedia.json --filter_strategy_name AutoFilter_Chen_Like --gpu_optional



Configurations
--------------

Most hyperparameters that should remain constant across multiple runs (to ensure comparability) are specified in a separate configuration file. By default, the system utilizes the `./Configs/standard.json` file. If one wants to choose another configuration file, then this can achieved through adding `--experiment_config <config>` in the execution.
