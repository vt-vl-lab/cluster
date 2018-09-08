## NewRiver

### TensorFlow
WARNING: DO NOT follow the instruction from ARC. It's not working because cudnn is not visible on GPU nodes.
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

### Caffe2 and Detectron
- This instruction requires Anaconda/Miniconda, please install it first
- Compiling may need GPU, while GPU node cannot get access to the Internet. So please open another terminal when you need to git clone or pip install or conda install; while use the GPU node when you are building/compiling GPU related things
- The [Caffe2](https://github.com/pytorch/pytorch) has been merged to the [Pytorch](https://github.com/pytorch/pytorch), the installation instruction with the [old Caffe2](https://github.com/caffe2/caffe2) repository does not work anymore. Here is the installation instruction of the new Caffe2 without any sudo access: [New Caffe2 installation instruction](https://github.com/vt-vl-lab/cluster/blob/master/install_Caffe2_NewRiver.md)

1. Make a separate virtual environment for Caffe2 (Must select python2.7 for Detectron and COCO-API)
```
conda create -n $NAME python=2.7
```
**NOTE:** Everytime you open a new terminal, please enter this environment.

2. [Seems not necessary] Build `protobuf` (credit: http://autchen.github.io/guides/2015/04/03/caffe-install.html)
```
git clone https://github.com/google/protobuf.git                                                                                                              
cd protobuf/                                                     
./autogen.sh
./configure --prefix=/home/you/usr
make 
make install
```
and then add this line to your `.bashrc` file, so that the system can use your protobuf
```
export PATH=/home/you/usr/lib:/home/you/usr/bin:$PATH
```

3. Install OpenCV 3.4.1 
```
conda install -c conda-forge opencv
```
**NOTE:** You cannot use lower version!

4. Clone
```
# Clone Caffe2's source code from our Github repository
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
git submodule update --init
```

5. Modify `CMakeList.txt` to use Caffe2 protobuf
In `line 29`, turn on the flag `BUILD_CUSTOM_PROTOBUF`.

6. Build
```
# Create a directory to put Caffe2's build files in
mkdir build && cd build

# Credit: https://github.com/caffe2/caffe2/issues/549
cmake -D CUDA_TOOLKIT_ROOT_DIR=/opt/apps/cuda/8.0.61 -D CUDNN_INCLUDE_DIR=/path/to/your/cudnn/include -D CUDNN_LIBRARY=/path/to/your/cudnn/lib64/libcudnn.so ..
```

7. Modify `cmake_install.cmake`
```
vim cmake_install.cmake
# Then modify line 6
# Replace /usr/local
# with the path to your new created environment
# i.e. /home/ylzou/install/NewRiver/anaconda3/envs/mycaffe2
```

6. Compile, link, and install
```
make install
```
and then add this line in your `.bashrc` file
```
export PYTHONPATH=/path/to/your/virtual/environment:/path/to/caffe2/build
```

8. Test
```
python caffe2/python/operator_test/relu_op_test.py
cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```
You can always go to the `build` to do `from caffe2.python import core` to see more detailed error or warning information. Usually you will need to install some more python packages, which can be done with `pip`.

9. Install Detectron
Follow https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md, you should be able to install it successfully.

### OpenCV
1. Install Anaconda and make your conda environment
2. Install from source (ver 2.4.13). Installing recent OpenCV with Deep Neural Network might be tricky. If you do not use OpenCV DNN, just installl 2.4.13 without DNN.
```
$git clone https://github.com/Itseez/opencv.git
$cd opencv
$git checkout 2.4
```
3. `mkdir build`
4. `cd build`
5. Do cmake. The following is the cmake command I used. You may want to change the PATH variables according to your miniconda installation path.
```
$cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=~/pkg/opencv_2.4.13_build/ \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D PYTHON_EXECUTABLE=/home/jinchoi/pkg/miniconda2/envs/tensorflow/bin/python \
    -D PYTHON_PACKAGES_PATH=/home/jinchoi/pkg/miniconda2/envs/tensorflow/lib \
    -D BUILD_EXAMPLES=ON ..
```
6. Do make
`$make -j32`
7. Setup your path in .bashrc file. The following is my path in .bashrc file.
```
export LD_LIBRARY_PATH=/home/jinchoi/pkg/opencv/build/lib/:$LD_LIBRARY_PATH
export INCLUDE_PATH=/home/jinchoi/pkg/opencv/include:$INCLUDE_PATH
export PYTHONPATH=/home/jinchoi/pkg/opencv/build/lib:$PYTHONPATH
export PYTHONPATH=/home/jinchoi/pkg/opencv/include:$PYTHONPATH
```
8. Enjoy!
```
$python
>>import cv2
```
If you don't see any errors, you are good to go.
