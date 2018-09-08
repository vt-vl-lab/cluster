The [Caffe2](https://github.com/pytorch/pytorch) has been merged to the [Pytorch](https://github.com/pytorch/pytorch), the installation instruction with the [old Caffe2](https://github.com/caffe2/caffe2) repository does not work anymore. Here is the installation instruction of the new Caffe2 without any sudo access. 

Addtionally, you can install [Non-local Neural Network](https://github.com/facebookresearch/video-nonlocal-net), built on top of the Caffe2 by following this instruction.
The installation instruction of Caffe2 and Non-Local Neural Network is based on [this](https://github.com/facebookresearch/video-nonlocal-net/blob/master/INSTALL.md).

# Caffe2 installation
Note that if you miss a single dependency installation, you might not build the Caffe2 successfully. Please follow this instruction closely and make sure you install every single dependency before building the Caffe2.

1) Create a new virtual env using conda
`jinchoi@nrlogin1:~$ conda create -n cf2_ffmp python=2.7`

2) Activate the virtual env
`jinchoi@nrlogin1:~$ source activate cf2_ffmp`

3) Install dependencies
```(cf2_ffmp) jinchoi@nrlogin1:~$ conda install --yes cmake && \
 conda install --yes git && \
 conda install --yes glog && \
 conda install --yes gflags && \
 conda install --yes gcc
```

4) Install other dependencies
`(cf2_ffmp) jinchoi@nrlogin1:~$ conda install --yes networkx && conda install --yes cython && conda install --yes libpng && conda install --yes protobuf && conda install --yes flask && conda install --yes future`

5) Install graphviz
`(cf2_ffmp) jinchoi@nrlogin1:~$ conda install --yes graphviz`

6) Install hypothesis
`(cf2_ffmp) jinchoi@nrlogin1:~$ pip install hypothesis`

7) Install other dependencies
```(cf2_ffmp) jinchoi@nrlogin1:~$ conda install --yes pydot && > conda install --yes lmdb && \
 conda install --yes pyyaml  && \
 conda install --yes matplotlib  && \
 conda install --yes requests  && \
 conda install --yes scipy  && \
 conda install --yes setuptools  && \
 conda install --yes six  && \
 conda install --yes tornado
```

8) Install opencv 3.4.1
`(cf2_ffmp) jinchoi@nrlogin1:~$ conda install -c conda-forge opencv`

9) Install setup tools 38.1.0
`(cf2_ffmp) jinchoi@nrlogin1:~/src$ pip install setuptools==38.1.0`

10) Install typing
`(cf2_ffmp) jinchoi@nrlogin1:~/src$ conda install typing`

11) Install bzip2 using conda-forge
`(cf2_ffmp) jinchoi@nrlogin1:~/src$  conda install -c conda-forge bzip2`

13) Upgrade cmake version to 3.12.2
`(cf2_ffmp) jinchoi@nrlogin1:~/src$ conda install -c conda-forge cmake`

14) Reinstall opencv to 3.4.1 as previous steps have downgraded the opencv to 2.4.x
`(cf2_ffmp) jinchoi@nrlogin1:~/src$ conda install -c conda-forge opencv`

15) Add paths
```
export CUDA_HOME="/usr/local/cuda-8.0"
export CUDNN_LIB_DIR="<path_to_your_cudnn_dir>"
export CUDNN_INCLUDE_DIR="<path_to_your_cudnn_dir>/include"
export CUDNN_LIBARY="<path_to_your_cudnn_dir>/lib64/libcudnn.so"
export PATH=/home/USERNAME/anaconda2/envs/caffe2/bin:/usr/local/bin:/usr/local/cuda-8.0/bin:$PATH
export C_INCLUDE_PATH=/home/USERNAME/anaconda2/envs/caffe2/include:/usr/local/cuda-8.0/include:$C_INLCUDE_PATH
export CPLUS_INCLUDE_PATH=/home/USERNAME/anaconda2/envs/caffe2/include:/usr/local/cuda-8.0/include:$CPLUS_INLCUDE_PATH
export LD_LIBRARY_PATH=/home/USERNAME/anaconda2/envs/caffe2/lib:/usr/lib64:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/USERNAME/anaconda2/envs/caffe2/lib:/usr/lib64:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib:$LIBRARY_PATH
conda activate caffe2
```

16) Building Caffe2 should work!!! 
1. `git clone https://github.com/pytorch/pytorch.git && cd pytorch`
2. `git submodule update --init --recursive`
3. Log in to a gpu node
4. `module load cuda`
    1. `nvcc —version` to ensure the cuda version is 8.0.61
5. Also modify and set the option "USE_FFMPEG" in pytorch/CMakeLists.txt "ON"
```
option(USE_FFMPEG "Use ffmpeg" ON)
```
6. Build it
`USE_FFMPEG=1 FULL_CAFFE2=1 python setup.py install` 

17) Test your installation
1. `cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"`
2. `python caffe2/python/operator_test/activation_ops_test.py`


# Non-local Neural Network installation
1. Clone Non-local Neural Network `git clone --recursive https://github.com/facebookresearch/video-nonlocal-net.git`
2. Before build, replace video ops of the original Caffe2 with the Non-local NN video ops
    1. `rm -rf pytorch/caffe2/video`
    2. `cp -r video-nonlocal-net/caffe2_customized_ops/video pytorch/caffe2/`
3. Build the Caffe2 following the [Caffe2 installation instruction](#caffe2-installation)
