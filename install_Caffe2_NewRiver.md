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

10) Install typing and packaging
```
(cf2_ffmp) jinchoi@nrlogin1:~/src$ conda install --yes typing && \
conda install --yes packaging 
```

11) Install bzip2 using conda-forge
`(cf2_ffmp) jinchoi@nrlogin1:~/src$  conda install -c conda-forge bzip2`

13) Upgrade cmake version to 3.12.2
`(cf2_ffmp) jinchoi@nrlogin1:~/src$ conda install -c conda-forge cmake`

14) Reinstall opencv to 3.4.1 as previous steps have downgraded the opencv to 2.4.x
`(cf2_ffmp) jinchoi@nrlogin1:~/src$ conda install -c conda-forge opencv`

15) Make sure your gcc is 4.8.5 and libgcc is 5.2.0, cmake is 3.12.2, opencv is 3.4.1
If you don't have these versions, you may face the following errors when you run `cmake` or when you run `import cv2` in your python code:
```
libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by cmake)
libstdc++.so.6: version `CXXABI_1.3.9' not found (required by cmake)
```

So, if you don't have these versions, do the followings so that you have the appropriate versions of gcc, libgcc, cmake and OpenCV.
```
conda install gcc=4.8.5
conda install libgcc=5.2.0
conda install -c conda-forge cmake
conda install -c conda-forge opencv
```

16) Add paths
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

17) Building Caffe2 should work!!! 
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

18) Test your installation
1. `cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"`
2. `python caffe2/python/operator_test/activation_ops_test.py`

# Non-local Neural Network installation
It does not work right now...

~1. Clone Non-local Neural Network `git clone --recursive https://github.com/facebookresearch/video-nonlocal-net.git`~
~2. Before build, replace video ops of the original Caffe2 with the Non-local NN video ops~
    ~1. `rm -rf pytorch/caffe2/video`~
    ~2. `cp -r video-nonlocal-net/caffe2_customized_ops/video pytorch/caffe2/`~
~3. Build the Caffe2 following the [Caffe2 installation instruction](#caffe2-installation)~

# FYI: Jinwoo's Conda Virtual Environment

I am posting the conda environment information I used for installing Caffe2 here:
```
attrs                     18.2.0                    <pip>
backports                 1.0                      py27_0
backports.functools_lru_cache 1.5                        py_1    conda-forge
backports_abc             0.5                      py27_0
blas                      1.1                    openblas    conda-forge
bzip2                     1.0.6                h470a237_2    conda-forge
ca-certificates           2018.8.24            ha4d7672_0    conda-forge
cairo                     1.14.12              he56eebe_3    conda-forge
certifi                   2016.2.28                py27_0
click                     6.7                      py27_0
cloog                     0.18.0                        0
cmake                     3.12.2               h011004d_0    conda-forge
curl                      7.61.0               h93b3f91_2    conda-forge
cycler                    0.10.0                   py27_0
cython                    0.26                     py27_0
dbus                      1.13.0               h3a4f0e9_0    conda-forge
decorator                 4.1.2                    py27_0
enum34                    1.1.6                     <pip>
expat                     2.2.5                hfc679d8_2    conda-forge
ffmpeg                    4.0.2                ha6a6e2b_0    conda-forge
flask                     0.12.2                   py27_0
fontconfig                2.13.0                        1    conda-forge
freetype                  2.8.1                hfa320df_1    conda-forge
functools32               3.2.3.2                  py27_0
future                    0.16.0                   py27_1
gcc                       4.8.5                         7
gettext                   0.19.8.1             h5e8e0c9_1    conda-forge
gflags                    2.2.0                         1
giflib                    5.1.4                h470a237_1    conda-forge
git                       2.11.1                        0
glib                      2.55.0               h464dc38_2    conda-forge
glog                      0.3.5                         0
gmp                       6.1.0                         0
gnutls                    3.5.17                        0    conda-forge
graphite2                 1.3.12               hfc679d8_1    conda-forge
graphviz                  2.38.0                        7    conda-forge
gst-plugins-base          1.12.5               hde13a9d_0    conda-forge
gstreamer                 1.12.5               h61a6719_0    conda-forge
harfbuzz                  1.8.5                h2bb21d5_0    conda-forge
hdf5                      1.10.2               hc401514_2    conda-forge
hypothesis                3.71.0                    <pip>
icu                       58.2                 hfc679d8_0    conda-forge
isl                       0.12.2                        0
itsdangerous              0.24                     py27_0
jasper                    1.900.1              hff1ad4c_5    conda-forge
jbig                      2.1                           0
jinja2                    2.9.6                    py27_0
jpeg                      9c                   h470a237_1    conda-forge
kiwisolver                1.0.1            py27h2d50403_2    conda-forge
krb5                      1.14.6                        0    conda-forge
libffi                    3.2.1                         1
libgcc                    5.2.0                         0
libgcc-ng                 7.2.0                hdf63c60_3    conda-forge
libgfortran               3.0.0                         1
libiconv                  1.15                 h470a237_3    conda-forge
libidn11                  1.33                          0    conda-forge
libpng                    1.6.34               ha92aebf_2    conda-forge
libprotobuf               3.4.0                         0
libssh2                   1.8.0                         0
libstdcxx-ng              7.2.0                hdf63c60_3    conda-forge
libtiff                   4.0.9                he6b73bb_2    conda-forge
libtool                   2.4.2                         0
libuuid                   2.32.1               h470a237_2    conda-forge
libuv                     1.22.0               h470a237_1    conda-forge
libwebp                   0.5.2                         7    conda-forge
libxcb                    1.12                          1
libxml2                   2.9.4                         0
lmdb                      0.9.21                        0
markupsafe                1.0                      py27_0
matplotlib                2.2.2                    py27_1    conda-forge
mkl                       2017.0.3                      0
mpc                       1.0.3                         0
mpfr                      3.1.5                         0
ncurses                   6.1                  hfc679d8_1    conda-forge
nettle                    3.3                           0    conda-forge
networkx                  1.11                     py27_0
numpy                     1.15.1          py27_blas_openblashd3ea46f_1  [blas_openblas]  conda-forge
openblas                  0.2.20                        8    conda-forge
opencv                    3.4.1           py27_blas_openblash829a850_201  [blas_openblas]  conda-forge
openh264                  1.7.0                         0    conda-forge
openssl                   1.0.2p               h470a237_0    conda-forge
packaging                 16.8                     py27_0
pango                     1.40.14              h9105a7a_2    conda-forge
pcre                      8.39                          1
pip                       9.0.1                    py27_1
pixman                    0.34.0                        0
protobuf                  3.4.0                    py27_0
pycairo                   1.10.0                   py27_0
pydot                     1.0.28                   py27_0
pyparsing                 1.5.6                    py27_0
pyqt                      5.6.0                    py27_2
python                    2.7.15               h9fef7bc_0    conda-forge
python-dateutil           2.6.1                    py27_0
pytz                      2017.2                   py27_0
pyyaml                    3.12                     py27_0
qt                        5.6.2                h50c60fd_8    conda-forge
readline                  7.0                  haf1bffa_1    conda-forge
requests                  2.14.2                   py27_0
rhash                     1.3.4                         0    conda-forge
scipy                     1.1.0           py27_blas_openblash7943236_201  [blas_openblas]  conda-forge
setuptools                36.4.0                   py27_1
setuptools                38.1.0                    <pip>
singledispatch            3.4.0.3                  py27_0
sip                       4.18                     py27_0
six                       1.10.0                   py27_0
sqlite                    3.24.0               h2f33b56_1    conda-forge
ssl_match_hostname        3.5.0.1                  py27_0
subprocess32              3.2.7                    py27_0
tk                        8.6.8                         0    conda-forge
torch                     0.5.0a0+802380a           <pip>
tornado                   4.5.2                    py27_0
typing                    3.6.2                    py27_0
werkzeug                  0.12.2                   py27_0
wheel                     0.29.0                   py27_0
x264                      1!152.20180717       h470a237_0    conda-forge
xorg-kbproto              1.0.7                h470a237_2    conda-forge
xorg-libice               1.0.9                h470a237_4    conda-forge
xorg-libsm                1.2.2                h8c8a85c_6    conda-forge
xorg-libx11               1.6.6                h470a237_0    conda-forge
xorg-libxext              1.3.3                h470a237_4    conda-forge
xorg-libxrender           0.9.10               h470a237_2    conda-forge
xorg-renderproto          0.11.1               h470a237_2    conda-forge
xorg-xextproto            7.3.0                h470a237_2    conda-forge
xorg-xproto               7.0.31               h470a237_7    conda-forge
xz                        5.2.4                h470a237_1    conda-forge
yaml                      0.1.6                         0
zlib                      1.2.11                        0
```
