# Huckleberry

## Table of Contents
- [ARC account](#arc-account) 
- [Access](#access) 
- [Anaconda](#anaconda)
- [Dependencies](#dependencies)
- [CUDA](#cuda)
- [Deep Learning Frameworks](#deep-learning-frameworks)
- [Submit a job](#job)

## ARC Acount
If you don't have an VT ARC account, make one from here: https://vt4help.service-now.com/sp?id=sc_cat_item&sys_id=2d1b7bfa0ff52680d3254b9ce1050e46
  - Advisor: When requesting an account, don't forget to mention that Jia-Bin Huang is your supervisor so that you can have "allocations" for faster job execution without pending forever in job queues. 
  - Allocation: To use the "allocations" you should let Jia-Bin knows that you want to use it. Just tell him your VT pid, then he wiil add you. 
  - Slurm queue: You can also use `ece_priority_q` instead of `normal_q` in slurm to use more GPUs and submit higher priority jobs.

## Access
### SSH
General instructions for how to access unix systems, you can check this [link](https://secure.hosting.vt.edu/www.arc.vt.edu/accessing-unix-system/)
### On Campus
1. Make an account. You may ask [Jia-Bin](https://filebox.ece.vt.edu/~jbhuang/) to do this.
2. Just ssh to the `huckleberry1.arc.vt.edu` with your pid
`ssh jinchoi@huckleberry1.arc.vt.edu`
3. Enjoy!  
### Off Campus
There are two ways to access off campus.
#### Using VPN
1. Install Pulse (VPN client) from [here](https://vt4help.service-now.com/kb_view_customer.do?sysparm_article=KB0010740)
2. Turn on Pulse
3. SSH to the huckleberry as if you are on campus. (refer to 1)
4. Enjoy!        
#### Connect to CVMLP servers, then connect to PowerAI
1. ssh to one of the CVMLP clusters with a port number 2222 (You need to use port 2222 to access CVMLP clusters off campus) e.g. `ssh -p 2222 <your_pid>@marr.ece.vt.edu`
2. ssh to the PowerAI as if you are on campus. (refer to 1)

### Public Key (password-less ssh)
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

### Remote Editing Environment
You can set up a remote editing environment using sftp connect. This example is using Atom + Remote FTP, but you can do similar things for other editors + sftp plug-ins.
1. First setup your password-less ssh environment. Follow the instructions in 2.
2. On your local machine, choose a project directory to sync your source codes.
3. Install `RemoteFTP`. Go to Setting->Install, type RemoteFTP, Install it.
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


## Anaconda
Download and install miniconda (minimum version of the Anaconda) for IBM architecture (ppc64le) from here: https://docs.conda.io/en/latest/miniconda.html

You can download and install either [Python3](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh) or [Python2](https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-ppc64le.sh). Actually it does not matter too much as you will set up a few different virtual environments and you can specify which python (2 or 3) you will use for the environment. For example,
You can do `conda create -n pytorch_1.1_python_2.7 python=2.7` in order to use python 2.7 
or You can do `conda create -n pytorch_1.1_python_3.7 python=3.7` in order to use python 3.7.

Then, you can install other modules using `conda install …` or `pip install ...`.
Open your `.bashrc` file.
$ vi .bashrc
And add the following `export PATH="/home/jinchoi/miniconda3_huckleberry/bin:$PATH"`.


## Dependencies
### OpenCV
1. Install miniconda and make your conda environment
2. Create and activate your virtual environment e.g. `conda activate pytorch_1.1_python_3.7`
3. Install OpenCV `conda install opencv`

### FFmpeg
If you already installed OpenCV, then you might already have FFmpeg. If you don't have OpenCV installed, do the followings.
1. Install miniconda and make your conda environment
2. Create and activate your virtual environment e.g. `conda activate pytorch_1.1_python_3.7`
3. Install OpenCV `conda install ffmpeg`

## CUDA
You can load cuda library from a shell using this:
`module load cuda/10.1.105`
Then you can verify whether cuda is loaded by this:
`nvcc -V`

## Deep Learning Frameworks
### Pytorch
#### From source
Pytorch installation is quite simple. Clone the sources , fulfill the dependencies and there you go!

1. `git clone https://github.com/pytorch/pytorch.git`
2. (optional) If you want to install a specific version, checkout that version. e.g. `git checkout v0.2.0`
3. `export CMAKE_PREFIX_PATH=[anaconda virtual env root directory]`
4. `conda install numpy pyyaml setuptools cmake cffi`
5. `NO_CAFFE2_OPS=1 python setup.py install`
Done!

Possible error for v1.1.0: Build fails because it cannot find a gfortran library
1. Check if gfortran is in your conda environment (skip to 3 if YES)
`ls /home/USERNAME/.conda/envs/CONDA_ENVIRONMENT/lib/ | grep gfortran`
2. Install gfortran
`conda install libgfortran`
3. Add the path to gfortran and build pytorch 
`USE_MKLDNN=0 BUILD_CAFFE2_OPS=0 LD_LIBRARY_PATH="/home/USERNAME/.conda/envs/CONDA_ENVIRONMEN/lib/" python setup.py install`

#### From conda repo
Or alternatively, you can follow [this](https://stackoverflow.com/questions/52750622/how-to-install-pytorch-on-power-8-or-ppc64-machine) to not build from source. But currently it only supports PyTorch 0.4.0.
1. `conda install -c anaconda ninja`
2. `conda install -c jjhelmus pytorch`
3. `conda install -c engility torchvision`
Done!

### TensorFlow
#### Installing TF 1.10 with your own anaconda3
1. Download [this](https://filebox.ece.vt.edu/~jinchoi/files/TF_huckleberry/TF_Utility-20181012T162146Z-001.zip) and unzip it on huckleberry
2. Make an virtual env with your own anaconda3, and activate it
3. Install TF using pip. i.e. pip install tensorflow-1.10.1-cp36-cp36m-linux_ppc64le.whl
4. Test the installation. i.e. python tfenv_testGPUs.py

#### Pre-installed TensorFlow with pre-installed anaconda3
You can load the pre-installed TensorFlow as follows.
1. module load gcc cuda anaconda3
2. You can use TF 1.10 from python

### Using a pre-setup PowerAI environment (v1.0.1)
```
$ module load gcc/7.3.0 cuda/10.1.105 jdk/8.0.5.31 Anaconda3/2019.03
$ source activate powerai16_ibm
```

## Job
Valid allocations: vllab_05, vllab_07

### Interactive jobs
`salloc -n2 -t 144:00:00 --gres=gpu:1 --mem=125g -p ece_priority_q`
This command will assign you a node with a node with 1 gpu, 2 cpus, 125GB cpu memory for 144 hours on ece_priority_q. And you can use bash shell to do whatever you want.

### Batch jobs
A sample batch script (`train.sh`) to submit a job. You should modify this script accordingly.

```
#!/bin/bash -l
#SBATCH --partition=ece_priority_q
#SBATCH -A vllab_07
#SBATCH -t 144:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=500G
#SBATCH -J test_this
#SBATCH -o /work/huckleberry/jinchoi/temporary_work/action/3dresnet/log/train.log

hostname
echo $CUDA_VISIBLE_DEVICES
module load cuda/10.1.105
source activate powerai36

which python
cd /home/jinchoi/src/3D-ResNets-PyTorch/

python train.py # your script to run
```

You submit a job by this:
`sbatch train.sh`

You can also submit a job array. See this: https://slurm.schedmd.com/job_array.html


## Custom Caffe
No one has been successfully installed a custom Caffe on PowerAI. There are some problems installing the dependencies such as glog, gflags, google protobuf.


## Instruction from ARC on using Huckleberry
### Installation
reference here: https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.0/navigation/pai_install.htm
first make sure you have logined into hulogin1/hulogin2

```
module load gcc cuda Anaconda3 jdk
java -version
conda create -n powerai36 python==3.6 # create a virtual environment
source activate powerai36             # activate virtual environment
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
```
if things don't work, add two channels and run commands showing below
```
conda config --add default_channels https://repo.anaconda.com/pkgs/main
conda config --add default_channels https://repo.anaconda.com/pkgs/r
```
install ibm powerai meta-package via conda
```
conda install powerai
```
keep type 'enter' and then enter 1 for license acceptance
```export IBM_POWERAI_LICENSE_ACCEPT=yes```

### DL library usage
step 1: request for GPU nodes
```salloc -N 1 --gres=gpu:pascal:1 --partition=normal_q --account=openpower```
step 2: load all necessary modules
```module load gcc cuda Anaconda3 jdk```
step 3: activate the virtual environment
```source activate powerai36```
step 4: test with simple code examples
```
python test_pytorch.py
python test_TF_multiGPUs.py
python test_keras.py
```
test with your own codes and begin your AI projects!

Reference: https://drive.google.com/drive/u/1/folders/1rMveCdGoO_TMX2u7wPqmtRHXERHvpsTD
