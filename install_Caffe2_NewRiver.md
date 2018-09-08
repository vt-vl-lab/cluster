The [Caffe2](https://github.com/pytorch/pytorch) has been merged to the [Pytorch](https://github.com/pytorch/pytorch), the installation instruction with the [old Caffe2](https://github.com/caffe2/caffe2) repository does not work anymore. Here is the installation instruction of the new Caffe2 without any sudo access. 

Addtionally, you can install [Non-local Neural Network](https://github.com/facebookresearch/video-nonlocal-net), built on top of the Caffe2 by following this instruction.
The installation instruction of Caffe2 and Non-Local Neural Network is based on [this](https://github.com/facebookresearch/video-nonlocal-net/blob/master/INSTALL.md).

# Caffe2 installation
1) create a new virtual env using conda
`jinchoi@nrlogin1:~$ conda create -n cf2_ffmp python=2.7`

2) activate the virtual env
`jinchoi@nrlogin1:~$ source activate cf2_ffmp`

3) install dependencies
```(cf2_ffmp) jinchoi@nrlogin1:~$ conda install --yes cmake && \
 conda install --yes git && \
 conda install --yes glog && \
 conda install --yes gflags && \
 conda install --yes gcc
```

4) install other dependencies
`(cf2_ffmp) jinchoi@nrlogin1:~$ conda install --yes networkx && conda install --yes cython && conda install --yes libpng && conda install --yes protobuf && conda install --yes flask && conda install --yes future`

5) install graphviz
`(cf2_ffmp) jinchoi@nrlogin1:~$ conda install --yes graphviz`

6) install hypothesis
`(cf2_ffmp) jinchoi@nrlogin1:~$ pip install hypothesis`

7) install other dependencies
```(cf2_ffmp) jinchoi@nrlogin1:~$ conda install --yes pydot && > conda install --yes lmdb && \
 conda install --yes pyyaml  && \
 conda install --yes matplotlib  && \
 conda install --yes requests  && \
 conda install --yes scipy  && \
 conda install --yes setuptools  && \
 conda install --yes six  && \
 conda install --yes tornado
```

8) install opencv 3.4.1
`(cf2_ffmp) jinchoi@nrlogin1:~$ conda install -c conda-forge opencv`

9) install setup tools 38.1.0
`(cf2_ffmp) jinchoi@nrlogin1:~/src$ pip install setuptools==38.1.0`

10) install typing
`(cf2_ffmp) jinchoi@nrlogin1:~/src$ conda install typing`

11) install bzip2 using conda-forge
`(cf2_ffmp) jinchoi@nrlogin1:~/src$  conda install -c conda-forge bzip2`

13) upgrade make version to 3.12.2
`(cf2_ffmp) jinchoi@nrlogin1:~/src$ conda install -c conda-forge cmake`

14) reinstall opencv to 3.4.1 as previous steps have downgraded the opencv to 2.4.x
`(cf2_ffmp) jinchoi@nrlogin1:~/src$ conda install -c conda-forge opencv`

15) build should work!!! 
1. `git clone https://github.com/pytorch/pytorch.git && cd pytorch`
2. `git submodule update --init --recursive`
3. Log in to a gpu node
4. `module load cuda`
    1. `nvcc —version` to ensure the cuda version is 8.0.61
5. Also modify and set the option "USE_FFMPEG" in pytorch/CMakeLists.txt "ON"
```
option(USE_FFMPEG "Use ffmpeg" ON)
```
6. build
    1. `USE_FFMPEG=1 FULL_CAFFE2=1 python setup.py install` 

16) Test your installation
1. `cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"`
2. `python caffe2/python/operator_test/activation_ops_test.py`


# Non-local Neural Network installation
1. Clone Non-local Neural Network `git clone --recursive https://github.com/facebookresearch/video-nonlocal-net.git`
2. Before build, replace video ops of the original Caffe2 with the Non-local NN video ops
    1. `rm -rf pytorch/caffe2/video`
    2. `cp -r video-nonlocal-net/caffe2_customized_ops/video pytorch/caffe2/`
3. Build the Caffe2 following the [Caffe2 installation instruction](#caffe2-installation)
