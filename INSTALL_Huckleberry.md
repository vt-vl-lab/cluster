## Huckleberry


### Anaconda
Update(08/24/2017): Continuum now supports the standard anaconda for ppc64le architecture. Check this out: https://www.continuum.io/downloads#linux.

Then, you can install other modules using `conda install …` or `pip install ...` and so on.
Open your `.bashrc` file.
$ vi .bashrc
And add the following `export PATH="/home/chengao/miniconda2/bin:$PATH"`.


### New instruction on using Huckleberry
#>>>>>>>>>>>>>>>>>>>>---------Part 1. Installation----------<<<<<<<<<<<<<<<<<<<#
# reference here: https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.0/navigation/pai_install.htm
# first make sure you have logined into hulogin1/hulogin2
module load gcc cuda Anaconda3 jdk
java -version
conda create -n powerai36 python==3.6 # create a virtual environment
source activate powerai36             # activate virtual environment
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
# if things don't work, add two channels and run commands showing below
conda config --add default_channels https://repo.anaconda.com/pkgs/main
conda config --add default_channels https://repo.anaconda.com/pkgs/r
# install ibm powerai meta-package via conda
conda install powerai
# keep type 'enter' and then enter 1 for license acceptance
export IBM_POWERAI_LICENSE_ACCEPT=yes

#>>>>>>>>>>>>>>>>>>>>---------Part 2. DL Library Usages----------<<<<<<<<<<<<<<<<<<<#
# step 1: request for GPU nodes
# salloc -N 1 --gres=gpu:pascal:1 --partition=normal_q --account=openpower
# step 2: load all necessary modules
module load gcc cuda Anaconda3 jdk
# step 3: activate the virtual environment
source activate powerai36
# step 4: test with simple code examples
python test_pytorch.py
python test_TF_multiGPUs.py
python test_keras.py

# test with your own codes and begin your AI projects!

Reference: https://drive.google.com/drive/u/1/folders/1rMveCdGoO_TMX2u7wPqmtRHXERHvpsTD



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
#### Pre-installed TensorFlow with pre-installed anaconda3
You can load the pre-installed TensorFlow as follows.
1. module load gcc cuda anaconda3
2. You can use TF 1.10 from python

#### Installing TF 1.10 with your own anaconda3
1. Download [this](https://filebox.ece.vt.edu/~jinchoi/files/TF_huckleberry/TF_Utility-20181012T162146Z-001.zip) and unzip it on huckleberry
2. Make an virtual env with your own anaconda3, and activate it
3. Install TF using pip. i.e. pip install tensorflow-1.10.1-cp36-cp36m-linux_ppc64le.whl
4. Test the installation. i.e. python tfenv_testGPUs.py

### Pytorch
#### From source
Pytorch installation is quite simple. Clone the sources , fulfill the dependencies and there you go!

1. `git clone https://github.com/pytorch/pytorch.git`
2. (optional) If you want to install a specific version, checkout that version. e.g. `git checkout v0.2.0`
3. `export CMAKE_PREFIX_PATH=[anaconda virtual env root directory]`
4. `conda install numpy pyyaml setuptools cmake cffi`
5. `NO_CAFFE2_OPS=1 python setup.py install`
Done!

#### From conda repo
Or alternatively, you can follow [this](https://stackoverflow.com/questions/52750622/how-to-install-pytorch-on-power-8-or-ppc64-machine) to not build from source. But currently it only supports PyTorch 0.4.0.
1. `conda install -c anaconda ninja`
2. `conda install -c jjhelmus pytorch`
3. `conda install -c engility torchvision`
Done!

### Custom Caffe
No one has been successfully installed a custom Caffe on PowerAI. There are some problems installing the dependencies such as glog, gflags, google protobuf.



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

#### Remote Editing Environment --SFTP currently not working, use NewRiver, 01/19/2018-- 
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
5. Connect to the server using "Packages->RemoteFTP->Connect"
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

