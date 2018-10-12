## CVMLP

### Basic Setup
1. Download [Anaconda X86 installer](https://www.continuum.io/downloads), and install it. If you use python2, choose Anaconda2; Anaconda3 is for python3. (You can choose not to append the line to `.bashrc`)
2. Download cudnn package [cuDNN v6.0 Library for Linux](https://developer.nvidia.com/cudnn), you will need to sign up an account.
3. Set up initialization bash files (following is Yuliang's setup)

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

4. (Optional) Set up remote editing with PyCharm, follow the instruction [here](https://drive.google.com/file/d/0B1c7NV1MfZatd2JZQjhXOVg4MzA/view?usp=sharing)

**NOTE:** I use Python3.6, cuda-8.0, and cudnn-8.0-6.0. You should modify the files above according to your choice.

### PyTorch
1. Check http://pytorch.org/, and choose the conda ones. (Yuliang's choice: conda install pytorch torchvision cuda80 -c soumith)
2. Check if you can run the following scripts in Python
```python
import torch
import torchvision
A = torch.LongTensor(1)
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
