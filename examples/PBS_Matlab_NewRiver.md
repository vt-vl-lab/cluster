#### A Matlab example with sample PBS script
1. Write a shell script for submission of jobs on NewRiver. This is a .sh file Chen uses. You can modify it appropriately.
```
#!/bin/bash
#
# Annotated example for submission of jobs on NewRiver
#
# Syntax
# '#' denotes a comment
# '#PBS' denotes a PBS directive that is applied during execution
#
# More info
# https://secure.hosting.vt.edu/www.arc.vt.edu/computing/newriver/#examples
#
# Chen Gao
# Aug 16, 2017
#

# Account under which to run the job
#PBS -A vllab_2017

# Access group. Do not change this line.
#PBS -W group_list=newriver

# Set some system parameters (Resource Request)
#
# NewRiver has the following hardware:
#   a. 100 24-core, 128 GB Intel Haswell nodes
#   b.  16 24-core, 512 GB Intel Haswell nodes
#   c.   8 24-core, 512 GB Intel Haswell nodes with 1 Nvidia K80 GPU
#   d.   2 60-core,   3 TB Intel Ivy Bridge nodes
#   e.  39 28-core, 512 GB Intel Broadwell nodes with 2 Nvidia P100 GPU
#
# Resources can be requested by specifying the number of nodes, cores, memory, GPUs, etc
# Examples:
#   Request 2 nodes with 24 cores each
#   #PBS -l nodes=1:ppn=24
#   Request 4 cores (on any number of nodes)
#   #PBS -l procs=4
#   Request 12 cores with 20gb memory per core
# 	#PBS -l procs=12,pmem=20gb
#   Request 2 nodes with 24 cores each and 20gb memory per core (will give two 512gb nodes)
#   #PBS -l nodes=2:ppn=24,pmem=20gb
#   Request 2 nodes with 24 cores per node and 1 gpu per node
#   #PBS -l nodes=2:ppn=24:gpus=1
#   Request 2 cores with 1 gpu each
#   #PBS -l procs=2,gpus=1
#PBS -l procs=1,pmem=16gb,walltime=2:20:00:00

# Set Queue name
#   normal_q        for production jobs on all Haswell nodes (nr003-nr126)
#   largemem_q      for jobs on the two 3TB, 60-core Ivy Bridge servers (nr001-nr002)
#   dev_q           for development/debugging jobs on Haswell nodes. These jobs must be short but can be large.
#   vis_q           for visualization jobs on K80 GPU nodes (nr019-nr027). These jobs must be both short and small.
#   open_q          for jobs not requiring an allocation. These jobs must be both short and small.
#   p100_normal_q   for production jobs on P100 GPU nodes
#   p100_dev_q      for development/debugging jobs on P100 GPU nodes. These jobs must be short but can be large.
# For more on queues as policies, see http://www.arc.vt.edu/newriver#policy
#PBS -q normal_q

# Send emails to -M when
# a : a job aborts
# b : a job begins
# e : a job ends
#PBS -M <PID>@vt.edu
#PBS -m bea

# Add any modules you might require. This example adds matlab module.
# Use 'module avail' command to see a list of available modules.
#
module load matlab

# Navigate to the directory from which this script was executed
cd /home/chengao/BIrdDetection/Chen_code

# Below here enter the commands to start your job. A few examples are provided below.
# Some useful variables set by the job:
#  $PBS_O_WORKDIR    Directory from which the job was submitted
#  $PBS_NODEFILE     File containing list of cores available to the job
#  $PBS_GPUFILE      File containing list of GPUs available to the job
#  $PBS_JOBID        Job ID (e.g., 107619.master.cluster)
#  $PBS_NP           Number of cores allocated to the job

### If run Matlab job ###
#
# Open a MATLAB instance and call Rich_new()
#matlab -nodisplay -r "addpath('Chen_code'); Rich_new;exit"

### If run Tensorflow job ###
#
```
2. To submit your job to the queuing system, use the command `qsub`. For example, if your script is in "JobScript.qsub", the command would be:
```
qsub ./JobScript.qsub
```
3. This will return your job name of the form. xxxxxx is the job number
```
xxxxxx.master.cluster
```
4. To check a job’s status, use the checkjob command:
```
checkjob -v xxxxxx
```
5. To check resource usage on the nodes available to a running job, use:
```
jobload xxxxxx
```
6. To remove a job from the queue, or stop a running job, use the command qdel
```
qdel xxxxxx
```
