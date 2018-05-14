# Virginia Tech Vision and Learning Lab Computing Resources

<img src="https://filebox.ece.vt.edu/~jbhuang/images/vt-logo.png" width="240" align="right">

Instructions for using clusters at Virginia Tech

## Table of Contents
- [Common](#common) 
- [CVMLP](#cvmlp) 
- [NewRiver](#newriver)
- [Huckleberry](#huckleberry-powerai)
- [Amazon AWS](#amazon-aws)

## Common
### Use customized python kernel in Jupyter Notebook:
1. Install anaconda/miniconda of your choice
2. Create an environment `conda create --name myenv`
3. Install ipykernel `pip install ipykernel`
4. Open the your environment `source activate myenv`
5. Install a Python (myenv) Kernel in the environment by `python -m ipykernel install --user --name myenv --display-name "Python (myenv)"`
5. Open Notebook, go to Kernel/Change kernel, you will see a kernel called “Python (myenv)” besides the default kernel. Select it.
6. Done. Now you can use your anaconda python and packages.

### Synchronize files from CVMLP to NewRiver/Huckleberry (Manually)
1. Create a exclude list in your directory, to specify the files/folders you don't want to synchronize (e.g. checkpoints)
2. Do this
```
rsync -avz --exclude-from="project/exclude-list.txt" project/ ylzou@newriver1.arc.vt.edu:/home/ylzou/research/project
```
Now you synchronize all the files under `project` directory with `/home/ylzou/research/project`.

**TODO**: Set up automatical synchronization

### Restore deleted files in NewRiver/Huckleberry
Check `~/.snapshot`, it keeps an hourly backup for 4 days.

## CVMLP
**NOTE:** Currently, slurm is not working on CVMLP.

### Wiki
Wiki page: https://mlp.ece.vt.edu/wiki/doku.php/computing

### root access (root users only)
#### get a su bash shell
```
sudo bash
```
#### reboot
```
sudo reboot
```
#### restart slurm for rebooted machine
e.g. after fukushima reboot, you need the followings on the slurm master machine (marr). 
```
sudo munged
sudo service slurm restart
sudo scontrol update node=fukushima state=resume
```

## NewRiver
### Submit Jobs
Access to all compute engines (aside from interactive nodes) is controlled via the job scheduler. You can follow the instructions [here](https://secure.hosting.vt.edu/www.arc.vt.edu/computing/newriver/#examples)

#### A Matlab example wiht sample PBS script
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
### Interactive GPU Jobs

```
interact -q p100_dev_q -lnodes=1:ppn=10:gpus=1 -A vllab_2017 -l walltime=2:00:00
```

NOTE: You can also use `p100_normal_q` and set longer walltime.

### Switching between Huckleberry and Newriver

```
serv_name=$(hostname)
if [[ $serv_name == *"hu"* ]];
 then
   # Set up Huckleberry Dependencies
   export PATH="/home/user_name/miniconda2/bin:$PATH"
 else
   # Set up Newriver Dependencies
   export PATH="/home/user_name/anaconda2/bin:$PATH"
 fi
```

#### Remote Editing Environment
You can set up a remote editing environment using sftp connect. This example is using Atom + Remote FTP, but you can do similar things for other editors + sftp plug-ins.
1. First setup your password-less ssh environment. Follow the instructions in 2.
2. On your local machine, choose a project directory to sync your source codes.
3. Install `Remote-ftp`. Go to Setting->Install, type RemoteFTP, Install it.
4. Write a `.ftpconfig` file in the chosen directory as follows.
```
{
    "protocol": "sftp",
    "host": "newriver1.arc.vt.edu", // string - Hostname or IP address of the server. Default: 'localhost'
    "port": 22, // integer - Port number of the server. Default: 22
    "user": "jinchoi", // string - Username for authentication. Default: (none)
    "remote": "/home/jinchoi/src/",
    "privatekey": "/Users/jwC/.ssh/id_rsa" // string - Path to the private key file (in OpenSSH format). Default: (none)
}
```
For the “User”, “remote”, “privatekey” fields, you should modify them for your own settings. You may use VPN client if you are off-campus and want to use PowerAI. If you are off-campus and want to use CVMLP clusters, you can simply use port number 2222.
5. Connect to the server using "Packages->RemoteFTP->Connect"
6. Enjoy!

## Huckleberry (PowerAI)
### General Rule of Thumb: DO NOT SKIP THIS!
Please fully utilize all the GPUs when you are submitting jobs to PowerAI. Each gpu node on PowerAI consists of 4 gpus. If you just submit a job naively, it will only use one GPU but it will block other people to use that node. It is too inefficient. So please run 4 jobs per GPU node. It is important as people outside the lab started to use PowerAI. 

So when you have 4 different models to train, please DO NOT `sbatch model1.sh`, `sbatch model2.sh`, `sbatch model3.sh`, `sbatch model4.sh` unless each of your job requires GPU memory more than 16GB.
Please do `sbatch model1.sh`, ssh to the node your model1 is assigned, then run your three other models in a background of that node using nohup, screen, or whatever your choice.
As far as we know, this is the best way to submit multiple jobs on a single GPU node. If you have more elegant way to submit 4 different jobs on a single GPU node, please let us know.

### Administrator
You can ask [James McClure](mcclurej@vt.edu) if you have questions. Or you can ask [Jinwoo](jinchoi@vt.edu).
### Help Ticket
When there is a problem, e.g. particular node down when you cancel a job by either ctrl + c or scancel command, it would probably be good to submit a help ticket from ARC website if nodes are offline for this reason and also to email. Check the box for huckleberry. This should help to ensure that multiple people see the request. 
https://vt4help.service-now.com/sp?id=sc_cat_item&sys_id=4c7b8c4e0f712280d3254b9ce1050e3c

### Access
#### SSH
General instructions for how to access unix systems, you can check this [link](https://secure.hosting.vt.edu/www.arc.vt.edu/accessing-unix-system/)
#### On Campus
1. Make an account. You may ask [Jia-Bin](https://filebox.ece.vt.edu/~jbhuang/) to do this.
2. Just ssh to the `huckleberry1.arc.vt.edu` with your pid
`ssh jinchoi@huckleberry1.arc.vt.edu`
3. Enjoy!  
#### Off Campus
There are two ways to access off campus.
##### Using VPN
1. Install Pulse (VPN client) from [here](https://vt4help.service-now.com/kb_view_customer.do?sysparm_article=KB0010740)
2. Turn on Pulse
3. SSH to the huckleberry as if you are on campus. (refer to 1)
4. Enjoy!        
##### Connect to CVMLP servers, then connect to PowerAI
1. ssh to one of the CVMLP clusters with a port number 2222 (You need to use port 2222 to access CVMLP clusters off campus) e.g. `ssh -p 2222 <your_pid>@marr.ece.vt.edu`
2. ssh to the PowerAI as if you are on campus. (refer to 1)

#### Public Key
1. From personal computer:
`$ scp -r <userid>@godel.ece.vt.edu:/srv/share/lab_helpful_files/ ~/`
2. Change <userid> to your CVL account username in the ~/lab_helpful_files/config file and move it to ~/.ssh
3. `$ mv ~/lab_helpful_files/config ~/.ssh/`
4. Add the following lines to the “config” file
```
Host huck
	Hostname huckleberry1.arc.vt.edu
	Port 22
	User jinchoi
```
You should change User to <your_pid>. You may change the huck to whatever name you want to use.

5. `$ ssh-keygen -t rsa`

6. Enter a bunch - Make sure ~/ on sever has .ssh folder
login, does `$ cd ~/.ssh` work? if not, type
```
$ mkdir .ssh
$ scp ~/.ssh/id_rsa.pub <userid>@huckleberry1.arc.vt.edu:~/.ssh/
```
7. On the PowerAI server (huckleberry):
```
$ cd ~/.ssh/
$ cat id_rsa.pub >> authorized_keys2
$ chmod 700 ~/.ssh
$ chmod 600 ~/.ssh/authorized_keys2
```
8. Now you can type the following to connect to PowerAI from your PC (If you are off-campus, you need to use VPN)
`$ ssh huck`

#### Remote Editing Environment --SFTP currently not working, use NewRiver, 01/19/2018-- 
You can set up a remote editing environment using sftp connect. This example is using Atom + Remote FTP, but you can do similar things for other editors + sftp plug-ins.
1. First setup your password-less ssh environment. Follow the instructions in 2.
2. On your local machine, choose a project directory to sync your source codes.
3. Install `RemoteFTP`. Go to Setting->Install, type RemoteFTP, Install it.
4. Write a `.ftpconfig` file in the chosen directory as follows.
```
{
    "protocol": "sftp",
    "host": "huckleberry1.arc.vt.edu", // string - Hostname or IP address of the server. Default: 'localhost'
    "port": 22, // integer - Port number of the server. Default: 22
    "user": "jinchoi", // string - Username for authentication. Default: (none)
    "remote": "/home/jinchoi/src/",
    "privatekey": "/Users/jwC/.ssh/id_rsa" // string - Path to the private key file (in OpenSSH format). Default: (none)
}
```
For the “User”, “remote”, “privatekey” fields, you should modify them for your own settings. You may use VPN client if you are off-campus and want to use PowerAI. If you are off-campus and want to use CVMLP clusters, you can simply use port number 2222.
5. Connect to the server using "Packages->RemoteFTP->Connect"
6. Enjoy!

#### Jupyter Notebook
The current PowerAI GPU nodes do not support internet access. Not sure how to setup an Jupyter Notebook environment. This [link](https://secure.hosting.vt.edu/www.arc.vt.edu/digits-user-guide/) might be helpful, but not tested yet.

##### Updates from Subhashree: Maybe not verified on GPU nodes?
The below instructions can be followed when connected to CVMLP/university wifi.
1. Ssh into PowerAI cluster
2. Launch the notebook using:
`$jupyter notebook --no-browser --port=7777`
You should get an output. The Jupyter Notebook is running at: http://localhost:7777/?token=104f6f1af5b7fdd761f28f5746c35b47f89d00698157ce85
3. Open a new terminal in the local machine and use the following port forwarding command.
`$ssh -N -L localhost:7777:localhost:7777 subha@huckleberry1.arc.vt.edu`
4. Now open the browser and visit localhost:7777 and enter the token key mentioned in the link where your notebook is running

### Using GPUs
You should submit GPU jobs only using slurm. You can follow the instructions [here](https://secure.hosting.vt.edu/www.arc.vt.edu/slurm-user-guide/). 

In addition to the instructions, [here](https://www.rc.fas.harvard.edu/resources/documentation/convenient-slurm-commands/) are some more useful information. 

### Queue
Our group memebers can use priority_q when submitting either interactive or batch jobs on PowerAI. Instead of submitting jobs to "normal_q" (crowded and limited walltime), we can submit jobs to "priory_q". In "priority_q", we will have relaxed walltime restriction and ensure at least 40% of the computation cycles of PowerAI are allocated for priority_q. 

#### Debugging a slurm job ID
`scontrol show jobid -dd <jobid>`
It will show you what .sh file you used for the jobid. Sometimes you need this information.

#### A sample slurm batch script using TF
This is a train.sh file Jinwoo uses. You can modify it appropriately.
```
#!/bin/bash -l
#SBATCH -p normal_q   # you can also use priority_q 
#SBATCH -N 1
#SBATCH -t 144:00:00
#SBATCH -J c3d-full
#SBATCH -o ./log/C3D-RGB-Full_Training_UCF101_lr_0.001_batch_256_320k_full_img_size.log

hostname
echo $CUDA_VISIBLE_DEVICES
module load cuda
source /opt/DL/tensorflow/bin/tensorflow-activate
export PYTHONHOME="/home/jinchoi/pkg/miniconda2/envs/tensorflow"

srun python ./tools/train_net.py --device gpu --device_id 0 --imdb UCF101_RGB_1_split_0_TRAIN --cfg experiments/cfgs/c3d_rgb_detect_lr_0.001.yml --network C3D_detect_train --iters 3200000
```
#### Submission of multiple GPU jobs per one GPU node
Each GPU node on PowerAI consists of 4 GPUs. But there is no instruction regarding how to submit multiple jobs (e.g. 4 different jobs) per one GPU node.
[James](mcclurej@vt.edu) says you can use `CUDA_VISIBLE_DEVICES` to do this, but it has not tested yet.


## Amazon AWS
The following instruction is written for VT-Vison group member specifically, who obtains access to VT-Vison AWS account. Please contact Prof. Jia-Bin Huang if you want to use this resource.

Console login: https://194474529881.signin.aws.amazon.com/console

Name: xxx

Password: xxx

### Use Caffe2 and Detectron:
0. If you are a VT-Vison group member, please go to setp 5 directly because Chen has already set up the enviroment.
1. Select US West (Oregon) Region. (Important because different region has different setting)
2. Launch a p3.2xlarge AWS instance with Deep Learning Base AMI (Ubuntu) Version 3.0 - ami-38c87440
3. Put [this](https://gist.github.com/matsui528/6d223d17241842c84d5882a9afa0453a) on `/home/ubuntu/`
4. Run `cd ~ && source install_caffe2_detectron.sh`
5. You can view the dashboard [here](https://us-west-2.console.aws.amazon.com/ec2/v2/home?region=us-west-2#Instances:)
6. SSH to the VM via `ssh -i /PATH/TO/.pem ubuntu@YOUR.VM'S.IP`
7. Caffe2 and Detectron are installed under `~/Project/detectron`
