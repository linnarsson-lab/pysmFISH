.. _cluster:

Cluster setup
=============


Introduction
------------------

Initially the code was designed as a set of disconnected scripts to run on a `HPC <https://www.nsc.liu.se/systems/triolith/>`_ 
with queue system managed by `slurm <https://slurm.schedmd.com/>`_ and parallel processing by `MPICH <https://www.mpich.org/>`_.
In order to improve the flow of the pipeline, make it more portable and easier to run we decided to re-write the whole pipeline
in order to use `dask-distributed <https://distributed.readthedocs.io/en/latest/>`_ to manage not only the cluster ``scheduler/workers`` 
but also the parallel handling of the processing. The current pipeline  can be run locally on your computer, 
remotely on a cluster or a combination of both.


Run the process locally
-----------------------
All the scripts that form the pipeline accept a `scheduler` tcp address as input. If not specified the script
will run locally using  `number of CPUs - 1`.



