.. _cluster:

Cluster setup
=============


Introduction
-------------

Initially the code was designed as a set of disconnected scripts to run on a `HPC <https://www.nsc.liu.se/systems/triolith/>`_ 
with queue system managed by `slurm <https://slurm.schedmd.com/>`_ and parallel processing by `MPICH <https://www.mpich.org/>`_.
In order to improve the flow of the pipeline, make it more portable and easier to run we decided to re-write the whole pipeline
in order to use `dask-distributed <https://distributed.readthedocs.io/en/latest/>`_ to manage not only the cluster ``scheduler/workers`` 
but also the parallel handling of the processing. The current pipeline  can be run locally on your computer, 
remotely on a cluster or a combination of both.


Run the process locally
------------------------
All the scripts that form the pipeline accept a `scheduler` tcp address as input. If not specified the script
will run locally using  ``number of CPUs - 1``.

Run the process on a cluster
----------------------------

**Start scheduler**

- If you want to use the master node as scheduler ``ssh`` into it in tunneling mode  ``ssh -L 7000:localhost:7000 user@cluster.ext``.
- If you want to use another node as scheduler ssh into it in tunneling mode ``ssh -L 7002:localhost:7002 user@node.cluster.ext``.
- From a terminal window start the ``dask-scheduler``  (ex. ``dask-scheduler --port 7001 --bokeh`` ).

**Start workers**

- For each node ssh into it in tunneling mode: 'ssh -L 7003:localhost:7003 user@node2.cluster.ext``.
- From a termial window start ``dask-worker`` (ex. ``dask-worker tcp://130.237.132.207:7001 --nprocs 10 --local-directory /tmp --memory-limit=220e9``).

**Start bokeh server**

From a terminal window ``ssh -L 8787:localhost:8787 user@cluster.ext`` (8787 is the port assigned by defalult).

.. tip::
    It is possible to use a couple of simple scripts to launch/kill a cluster. It works with a small number of nodes.
    The advantage is that you can start a conda env in all nodes and run dask-distributed without a 
    worload managing software like SLURM or PBS.
    For larger cluster look at the `dask-distributed <https://distributed.readthedocs.io/en/latest/>`_ manual.  
    
    **Example**
    We have a cluster with few nodes. The main node is called monod and you need to ssh into it in order to access the
    nodes called monod01, monod02...etc.
    in the launching bash script below I will use monod01 as scheduler and monod09 and monod10 as workers.

    **launch_cluster.sh**
    ::
        #!/bin/bash
        CONDAENV="source activate testing_speed" 
        SCHEDULERON="dask-scheduler"
        SCHEDULER="monod01"
        NPROCS="10"
        NTHREDS="1"
        MEM="220e9"
        LOCDIR="/tmp"
        WORKERON="dask-worker $SCHEDULER:8786 --nprocs $NPROCS --nthreads $NTHREDS --memory-limit $MEM --local-directory $LOCDIR"
        WORKERS="monod10 monod09"

        ssh $SCHEDULER "$CONDAENV; $SCHEDULERON" & 

        for WORKER in $WORKERS
        do
            ssh $WORKER "$CONDAENV; $WORKERON" & 
        done
        exit   
    
    **kill_cluster.sh**
    ::
        #!/bin/bash
        SCHEDULER="monod01"
        WORKERS="monod10 monod09"
        KILLSCHEDULER="killall -INT dask-scheduler"
        KILLWORKER="killall -INT dask-worker"


        ssh $SCHEDULER "$KILLSCHEDULER" & 

        for WORKER in $WORKERS
        do
            ssh $WORKER "$KILLWORKER" &
        done
        exit  


