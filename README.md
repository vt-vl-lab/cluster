# Virginia Tech Vision and Learning Lab Computing Resources

<img src="https://filebox.ece.vt.edu/~jbhuang/images/vt-logo.png" width="240" align="right">

Instructions for using clusters at Virginia Tech

## Table of Contents
- [Common](#common) 
- [CVMLP](#cvmlp) 
- [NewRiver](#newriver)
- [Cascades](#cascades)
- [Huckleberry](#huckleberry-powerai)
- [Amazon AWS](#amazon-aws)

## Common
### Switch between different ARC clusters
Note that all ARC clusters (e.g., NewRiver, Cascades, Huckleberry) share exactly **the same** file system (i.e., every file modifications you do in one cluster will affect all your clusters!). You should set up your environment in separate spaces for each cluster. And you can use the following scipts to automatically choose the correct environment when you log in.

```
serv_name=$(hostname)
if [[ $serv_name == *"hu"* ]]; then
    # Set up Huckleberry Dependencies
    export PATH="/home/user_name/miniconda2/bin:$PATH"
elif [[ $serv_name == *"nr"* ]]; then
    # Set up Newriver Dependencies
    export PATH="/home/user_name/anaconda2/bin:$PATH"
else
    # Set up Cascades Dependencies
    # Note that Cascades and NewRiver both use CentOS, 
    # you might sometimes use dependencies from NewRiver directly.
    # But they have different types of GPU (P100 v.s. V100),
    # this might cause some issues.
fi
```

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
Cannot log into this anymore.

### Computing Resources
1. GPU machines
- Fukushima(up): k80 x 16, 32 Cores Intel, 396GB Ram
- Werbos(up): k80 x 16, 32 Cores Intel, 396GB Ram
- Tesla(gpu down): k40 x 8, 32 Cores Intel, 3s96GB Ram
- Hebb(down): Titan Black x 7 
- Rosenblatt(down): Titan Black x 3 (?) 

2. CPU-only machines
- marr(slurm server node):  
- vapnik: 64 Cores Intel, 528GB Ram
- minsky: 64 Cores Intel, 528GB Ram
- mccarthy: 64 Cores Intel, 528GB Ram
- turing(down): 64 Cores Intel, 528GB Ram
- godel: 64 Cores Intel, 264GB Ram

### Install
Check [INSTALL_CVMLP.md](https://github.com/vt-vl-lab/cluster/blob/master/INSTALL_CVMLP.md)

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
### Job Submission
Access to all compute engines (aside from interactive nodes) is controlled via the job scheduler. You can follow the instructions [here](https://secure.hosting.vt.edu/www.arc.vt.edu/computing/newriver/#examples)

Example: [CPU (Matlab) job submission using PBS](https://github.com/vt-vl-lab/cluster/blob/master/examples/PBS_Matlab_NewRiver.md)

### Install
Check [INSTALL_NewRiver.md](https://github.com/vt-vl-lab/cluster/blob/master/INSTALL_NewRiver.md)

### Interactive GPU Jobs

```
interact -q p100_dev_q -lnodes=1:ppn=10:gpus=1 -A vllab_01 -l walltime=2:00:00
```

NOTE: You can also use `p100_normal_q` and set longer walltime.

### Important Commands
```
# Show CPU resources usage
showq

# See how many empty (GPU) nodes
qstat -Q p100_normal_q
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


## Cascades
You can basically follow the instructions for [NewRiver](#newriver).

### Interactive GPU Jobs

```
interact -q v100_dev_q -lnodes=1:ppn=10:gpus=1 -A vllab_01 -l walltime=2:00:00
```


## Huckleberry (PowerAI)
### General Rule of Thumb: DO NOT SKIP THIS!
Please fully utilize all the GPUs when you are submitting jobs to PowerAI. Each gpu node on PowerAI consists of 4 gpus. If you just submit a job naively, it will only use one GPU but it will block other people to use that node. It is too inefficient. So please run 4 jobs per GPU node. It is important as people outside the lab started to use PowerAI. 

So when you have 4 different models to train, please DO NOT `sbatch model1.sh`, `sbatch model2.sh`, `sbatch model3.sh`, `sbatch model4.sh` unless each of your job requires GPU memory more than 16GB.
Please do `sbatch model1.sh`, ssh to the node your model1 is assigned, then run your three other models in a background of that node using nohup, screen, or whatever your choice.
As far as we know, this is the best way to submit multiple jobs on a single GPU node. If you have more elegant way to submit 4 different jobs on a single GPU node, please let us know.


### Install
Check [INSTALL_Huckleberry.md](https://github.com/vt-vl-lab/cluster/blob/master/INSTALL_Huckleberry.md)


### Administrator
You can ask [James McClure](mcclurej@vt.edu) if you have questions. Or you can ask [Jinwoo](jinchoi@vt.edu).
### Help Ticket
When there is a problem, e.g. particular node down when you cancel a job by either ctrl + c or scancel command, it would probably be good to submit a help ticket from ARC website if nodes are offline for this reason and also to email. Check the box for huckleberry. This should help to ensure that multiple people see the request. 
https://vt4help.service-now.com/sp?id=sc_cat_item&sys_id=4c7b8c4e0f712280d3254b9ce1050e3c


### Using GPUs
You should submit GPU jobs only using slurm. You can follow the instructions [here](https://secure.hosting.vt.edu/www.arc.vt.edu/slurm-user-guide/). 

In addition to the instructions, [here](https://www.rc.fas.harvard.edu/resources/documentation/convenient-slurm-commands/) are some more useful information. 

### Queue
Our group memebers can use priority_q when submitting either interactive or batch jobs on PowerAI. Instead of submitting jobs to "normal_q" (crowded and limited walltime), we can submit jobs to "priory_q". In "priority_q", we will have relaxed walltime restriction and ensure at least 40% of the computation cycles of PowerAI are allocated for priority_q. 

#### Interactive GPU job
```
salloc -N1 --mem-per-cpu=8G -p priority_q -t 120:00 --gres=gpu:pascal:1
srun --pty /bin/bash
```
**NOTE:** When you want to exit the interactive mode, you need to do `ctrl+D` twice. 

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
