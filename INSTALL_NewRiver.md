## NewRiver

### TensorFlow
0. Connect to a GPU node.
1. Install Anaconda python of your choice.
2. `module purge`
3. `module load cuda/8.0.44`
4. Download cudnn from [here](https://developer.nvidia.com/cudnn).
5. Add `LD_LIBRARY_PATH` to your `.bashrc` file.
Jinwoo's example:
```
# set the server name
serv_name=$(hostname)

if [[ $serv_name == *"hu"* ]];
then
    # for PowerAI (huckleberry)
    # added by Miniconda2 4.3.14 installer
    export PATH="/home/jinchoi/pkg/miniconda2/bin:$PATH"
else
    # for newriver
    # added by Anaconda2 4.4.0 installer
    export PATH="/home/jinchoi/pkg/anaconda2_nr/bin:$PATH"
    export LD_LIBRARY_PATH=/home/jinchoi/lib/cuda/lib64:$LD_LIBRARY_PATH
fi
```
6. `source ~/.bashrc` so that the os can locate your cudnn directory.
7. Follow the official TensorFlow installation procedure provided [here](https://www.tensorflow.org/install/install_linux#InstallingAnaconda).
8. Enjoy!

#### Install Custom TensorFlow
0. (Optional) Create a virtual environment, assume that you have installed Anaconda/Miniconda
```
# Create the environment
conda create python=$VERSION -n $NAME

# Enter the environment
source activate $NAME
```
replace `$VERSION` with `2.7` or `3.6`, and set a `$NAME` for your environment

1. Add the following lines to your `~/.bashrc` file to load CUDA and CUDNN
```
#TF version >= 1.5.0
NVCC=/opt/apps/cuda/9.0.176/bin
export LD_LIBRARY_PATH=/opt/apps/cuda/9.0.176/lib64:$LD_LIBRARY_PATH
CUDA_PATH=/opt/apps/cudnn/7.1
export LD_LIBRARY_PATH=/opt/apps/cudnn/7.1/lib64:$LD_LIBRARY_PATH

## If you want to use older version, you can use the CUDA 8.0
#NVCC=/opt/apps/cuda/8.0.61/bin
#export LD_LIBRARY_PATH=/opt/apps/cuda/8.0.61/lib64:$LD_LIBRARY_PATH
## I am not sure if this cudnn corresponds to CUDA 8.0. If not, you can download it and set the path
#CUDA_PATH=/opt/apps/cudnn/6.0
#export LD_LIBRARY_PATH=/opt/apps/cudnn/6.0/lib64:$LD_LIBRARY_PATH
```

2. Install TensorFlow in the login (CPU) node
```
pip install tensorflow-gpu
```
if you want other version than 1.11.0, just specify the version

3. Sanity check
```
# Log in to the GPU node via interactive mode
interact -q p100_dev_q -lnodes=1:ppn=10:gpus=1 -A vllab_01 -l walltime=2:00:00

source ~/.bashrc

python
import tensorflow as tf
```

### Caffe2 and Detectron
- This instruction requires Anaconda/Miniconda, please install it first
- Compiling may need GPU, while GPU node cannot get access to the Internet. So please open another terminal when you need to git clone or pip install or conda install; while use the GPU node when you are building/compiling GPU related things
- The [Caffe2](https://github.com/pytorch/pytorch) has been merged to the [Pytorch](https://github.com/pytorch/pytorch), the installation instruction with the [old Caffe2](https://github.com/caffe2/caffe2) repository does not work anymore. Here is the installation instruction of the new Caffe2 without any sudo access: [New Caffe2 installation instruction](https://github.com/vt-vl-lab/cluster/blob/master/install_Caffe2_NewRiver.md)

### OpenCV
1. Install Anaconda and make your conda environment
2. Just do `pip install opencv-python`
```
$python
>>import cv2
```
If you don't see any errors, you are good to go.
