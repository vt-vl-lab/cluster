# Virginia Tech Vision and Learning Lab Computing Resources

<img src="https://filebox.ece.vt.edu/~jbhuang/images/vt-logo.png" width="240" align="right">

Instructions for using clusters at Virginia Tech

## Table of Contents
- [CVMLP](#cvmlp) 
- [NewRiver](#newriver)
- [Huckleberry](#huckleberry-powerai)


## CVMLP
### Wiki
Wiki page: https://mlp.ece.vt.edu/wiki/doku.php/computing
### Basic Setup
1. Download [Anaconda X86 installer](https://www.continuum.io/downloads), and install it. If you use python2, choose Anaconda2; Anaconda3 is for python3. (You can choose not to append the line to `.bashrc`)
2. Download cudnn package [cuDNN v6.0 Library for Linux](https://developer.nvidia.com/cudnn), you will need to sign up an account.
3. Set up initialization bash files (following is Yuliang's setup)
4. (Optional) Set up PyCharm, follow the instruction [here](https://drive.google.com/file/d/0B1c7NV1MfZatd2JZQjhXOVg4MzA/view?usp=sharing)

- /home/ylzou/.bashrc:
```bash
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
   . /etc/bashrc
fi
  
# User specific aliases and functions
source /home/ylzou/install/init.sh
```

- /home/ylzou/install/init.sh:
```bash
#!/bin/bash
HOST_NAME=`hostname | sed -e 's/\..*$//'`

# Python3.6
ANACONDA_BIN=/home/ylzou/anaconda3/bin
ANACONDA_LIB=/home/ylzou/anaconda3/lib
NVCC=/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

CMAKE_PATH=/home/ylzou/anaconda3/bin/cmake
# Add paths for python
export PATH=$CMAKE_PATH:$NVCC:$ANACONDA_BIN:$PATH

# cudnn-8.0-v6.0
export CUDA_ROOT=/home/ylzou/install/cuda
export LD_LIBRARY_PATH=/home/ylzou/install/cuda/lib64:$LD_LIBRARY_PATH

# Add gpu_lock package
export PATH=$PATH:/srv/share/gpu_lock
export PYTHONPATH=$PYTHONPATH:/srv/share/gpu_lock
```
(Remember to re-connect the cluster after you setup the init scripts)

**NOTE:** I use Python3.6, cuda-8.0, and cudnn-8.0-6.0. You should modify the files above according to your choice.

### PyTorch
1. Check http://pytorch.org/, and choose the conda ones. (Yuliang's choice: conda install pytorch torchvision cuda80 -c soumith)
2. Check if you can run the following scripts in Python
```python
import torch
Import torchvision
A = torch.Tensor(1)
A.cuda()
torch.backends.cudnn.version()
```
Reference: https://discuss.pytorch.org/t/error-when-using-cudnn/577/3


### TensorFlow
#### Virtual Environment (Not yet tested)
Follow the instructions [here](https://www.tensorflow.org/install/install_linux#installing_with_anaconda)
#### Install From Source
1. Download a [distribution archive of bazel (bazel-[VERSION]-dist.zip)](https://github.com/bazelbuild/bazel/releases), and unzip it in a empty folder you create.
2. Run  `bash ./compile.sh` ,this will create a bazel binary in output/bazel. Notice that we **cannot** move it to `/usr/local/bin directory`, since there already has an old-version one.
3. Clone TensorFlow repo, `git clone https://github.com/tensorflow/tensorflow`
4. Modify configure file in TensorFlow folder: Commet line166~line174. So that this file will not detect the version of default bazel, which is too old to use.
5. Run `./configure`, use default setting most of the time, except for the cuda and cudnn relevant ones, which you should check your initialize bash file for reference. (Yuliang's setting: `/home/ylzou/install/init.sh`)
6. Build a pip package
```
[path/to/your/bazel] build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
```
7. Get whl file
Run
```
bazel-bin/tensorflow/tools/pip_package/build_pip_package [/directory/to/store/this/file]
```
8. Install the pip package
```
pip install [/directory/to/your/whl/file]
```
9. Leave the TensorFlow directory, and test if you can import the package

### OpenCV 3.1
(Only for Python3.6, other version should be able to use Anaconda to install)
1. Clone repo, `git clone https://github.com/Itseez/opencv.git`
2. Checkout 
```
cd opencv 
git checkout 3.1.0 && git format-patch -1 10896129b39655e19e4e7c529153cb5c2191a1db && git am < 0001-GraphCut-deprecated-in-CUDA-7.5-and-removed-in-8.0.patch
```
3. Setup build directory
```
mkdir build
cd build
cmake -DBUILD_TIFF=ON -DBUILD_opencv_java=OFF -DWITH_CUDA=OFF -DENABLE_AVX=ON -DWITH_OPENGL=ON -DWITH_OPENCL=ON -DWITH_IPP=ON -DWITH_TBB=ON -DWITH_EIGEN=ON -DWITH_V4L=ON -DWITH_VTK=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_opencv_python2=OFF -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") -DPYTHON3_EXECUTABLE=$(which python3) -DPYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") ..
```
4. Make
```
make -j32
make install
```
5. Check
```
python
Import cv2
```
Reference:
- Jinwoo's script
- https://www.scivision.co/anaconda-python-opencv3/

### FFmpeg
```
conda install -c menpo ffmpeg=3.1.3
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
#PBS -l procs=12,pmem=16gb,walltime=2:20:00:00

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
interact -q p100_dev_q -lnodes=1:ppn=28:gpus=2 -A vllab_2017
```

## Huckleberry (PowerAI)
### Administrator
You can ask [James McClure](mcclurej@vt.edu) if you have questions. Or you can ask [Jinwoo](jinchoi@vt.edu).
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

#### Remote Editing Environment
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
5. Connect to the server using Packages->RemoteFTP->Connect
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
You should submit GPU jobs only using slurm. You can follow the instructions [here](https://secure.hosting.vt.edu/www.arc.vt.edu/slurm-user-guide/)
#### A sample slurm batch script using TF
This is a train.sh file Jinwoo uses. You can modify it appropriately.
```
#!/bin/bash -l
#SBATCH -p normal_q
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


### Platforms
#### Anaconda
PowerAI architecture is ppc64le: we don’t have standard anaconda installer for this architecture. Instead, install miniconda which contains only conda and python. You can find one [here](https://repo.continuum.io/miniconda/). Download it, and install on your custom directory on PowerAI clusters. Then, you can install other modules using `conda install …` or `pip install ...` and so on.
Open your `.bashrc` file.
$ vi .bashrc
And add the following `export PATH="/home/chengao/miniconda2/bin:$PATH"`.

#### FFmpeg
1. Basically follow this [instruction](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu). Howerver, I didn't get the dependencies with apt-get as I don't have a sudo permission. I'm describing the setting I used and it was successful installation.
2. Do make ffmpeg source dir
`mkdir ~/pkg/ffmpeg_sources`
3. Compile required dependencies
yasm -> no need
libx264
libx265
libfdk-aac
libmp3lame -> no need
libopus
libvpx
4. Do ffmpeg build and install
```
$cd ~/ffmpeg_sources
$wget http://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
$tar xjvf ffmpeg-snapshot.tar.bz2
$cd ffmpeg
$PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/pkg/ffmpeg_build/lib/pkgconfig" ./configure --prefix="$HOME/pkg/ffmpeg_build" --pkg-config-flags="--static" --extra-cflags="-I$HOME/pkg/ffmpeg_build/include" --extra-ldflags="-L$HOME/pkg/ffmpeg_build/lib" --bindir="$HOME/bin" --enable-gpl --enable-libfdk-aac   --enable-libfreetype --enable-libopus --enable-libvpx --enable-libx264 --enable-libx265 --enable-nonfree
$PATH="$HOME/bin:$PATH" make
$make install
$hash -r
```
5. Installation is now complete and ffmpeg is now ready for use. Your newly compiled FFmpeg programs are in `~/bin`. add your `~/bin` to `.bashrc` `$PATH` variable.
6. Enjoy!

References:
[1] https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu

#### OpenCV
1. Install miniconda and make your conda environment
2. Install from source (ver 3.1.0)
`$git clone https://github.com/Itseez/opencv.git`
This approach (installation from the source not from a zip file) is to avoid an knwon [issue](https://github.com/opencv/opencv/issues/6677)
3. Checkout and patch
`$cd opencv`
`$git checkout 3.1.0 && git format-patch -1 10896129b39655e19e4e7c529153cb5c2191a1db && git am < 0001-GraphCut-deprecated-in-CUDA-7.5-and-removed-in-8.0.patch`
4. Manually update two source files according to [this](https://github.com/opencv/opencv/pull/6982/commits/0df9cbc954c61fca0993b563c2686f9710978b08)
This step is to avoid "_FPU_SINGLE declaration error"
5. Go to the opencv root dir
`$cd opencv`
6. `mkdir build`
7. `cd build`
8. Do cmake. The following is the cmake command I used. You may want to change the PATH variables according to your miniconda installation path.
```
$cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=~/pkg/opencv_3.1.0_build/ \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/pkg/opencv_contrib-3.1.0/modules \
    -D PYTHON_EXECUTABLE=/home/jinchoi/pkg/miniconda2/envs/tensorflow/bin/python \
    -D PYTHON_PACKAGES_PATH=/home/jinchoi/pkg/miniconda2/envs/tensorflow/lib \
    -D BUILD_EXAMPLES=ON ..
```
9. Do make
`$make -j32`
10. Setup your path in .bashrc file. The following is my path in .bashrc file.
```
export LD_LIBRARY_PATH=/home/jinchoi/pkg/opencv/build/lib/:$LD_LIBRARY_PATH
export INCLUDE_PATH=/home/jinchoi/pkg/opencv/include:$INCLUDE_PATH
export PYTHONPATH=/home/jinchoi/pkg/opencv/build/lib:$PYTHONPATH
export PYTHONPATH=/home/jinchoi/pkg/opencv/include:$PYTHONPATH
```
11. Enjoy!
```
$python
>>import cv2
```
If you don't see any errors, you are good to go.

References:
[1] https://github.com/opencv/opencv/issues/6677     
[2] https://github.com/opencv/opencv/pull/6982/commits/0df9cbc954c61fca0993b563c2686f9710978b08
[3] http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/

#### TensorFlow
You can load the pre-installed TensorFlow as follows.
1. `“module load cuda”`
2. `“source /opt/DL/tensorflow/bin/tensorflow-activate”`
3. `Enjoy!`


#### Pytorch
Pytorch installation is quite simple. Clone the sources , fulfill the dependencies and there you go!

1. `git clone https://github.com/pytorch/pytorch.git`
2. `export CMAKE_PREFIX_PATH=[anaconda root directory]`
3. `conda install numpy pyyaml setuptools cmake cffi`
4. `python setup.py install`

Done!

#### Custom Caffe
No one has been successfully installed a custom Caffe on PowerAI. There are some problems installing the dependencies such as glog, gflags, google protobuf.
