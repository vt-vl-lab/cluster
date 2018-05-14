## Huckleberry

### Anaconda
Update(08/24/2017): Continuum now supports the standard anaconda for ppc64le architecture. Check this out: https://www.continuum.io/downloads#linux.

Then, you can install other modules using `conda install …` or `pip install ...` and so on.
Open your `.bashrc` file.
$ vi .bashrc
And add the following `export PATH="/home/chengao/miniconda2/bin:$PATH"`.


### FFmpeg
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

### OpenCV
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

### TensorFlow
You can load the pre-installed TensorFlow as follows.
1. `“module load cuda”`
2. `“source /opt/DL/tensorflow/bin/tensorflow-activate”`
3. `Enjoy!`


### Pytorch
Pytorch installation is quite simple. Clone the sources , fulfill the dependencies and there you go!

1. `git clone https://github.com/pytorch/pytorch.git`
2. `export CMAKE_PREFIX_PATH=[anaconda root directory]`
3. `conda install numpy pyyaml setuptools cmake cffi`
4. `python setup.py install`

Done!

### Custom Caffe
No one has been successfully installed a custom Caffe on PowerAI. There are some problems installing the dependencies such as glog, gflags, google protobuf.
