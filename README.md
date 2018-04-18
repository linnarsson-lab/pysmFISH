# smFISH_Analysis
Re-organized code for running batch analysis of large smFISH datasets

# Introduction
This version of the code rely on `dask.sdistributed` for handling the 
parallelism. We decided to use dask because it allows better deployment
and better profiling of the code. This code can run on a cluster or on 
a local machine using the cores available.

# Dask installation and cluster set up

## Installation
- Install dask and distributed: `pip install dask distributed --upgrade`  
- If you are planning to use the dask-ssh utility you need `pip install paramiko`
- In order to use the web interface you need to install bokeh `conda install bokeh -c bokeh`

## Start cluster
__Start scheduler__
- `ssh` into monod master node in tunneling mode: `'ssh -L 7000:localhost:7000 simone@monod.mbb.ki.se'`
- if you want to use another node as scheduler ssh into it `ssh -L 7002:localhost:7002 monod01`  
- start the dask-scheduler from a terminal `dask-scheduler` (ex. `dask-scheduler --port 7001 --bokeh` )

__Start worker__
- `ssh` into monod node in tunneling mode: `'ssh -L 7000:localhost:7000 monod03'`  
- start worker from terminal `dask-worker  tcp://130.237.132.207:7001 --nprocs 10 --local-directory /tmp --memory-limit=220e9`  

__Start bokeh server__
- From terminal `ssh -L 8787:localhost:8787 simone@monod.mbb.ki.se` (8787 is the port assigned)

Example script for setting up cluster with specific options

```bash
#!/bin/bash  
USERNAME=someUser  
HOSTS="monod02 monod03 monod04"  
SCRIPT="pwd; ls" # edit script with dask code for setting it up  
for HOSTNAME in ${HOSTS} ; do  
    ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "${SCRIPT}"  
done  
```
