#Cervical cancer cell detection system backend

## introduction 

This backend consists of two parts: faster RCNN and classification net. The faster RCNN needs to be compiled before using.

## prerequisites

* Python 3.*
* Pytorch 0.3.1
* CUDA 8.0 or higher

## Compilation of faster RCNN

```shell
cd fasterRCNN
pip install -r requirements.txt
```

The packages in  requirements.txt are:

* cython
* cffi
* opencv-python
* scipy
* easydict
* matplotlib
* pyyaml

```shell
cd lib
sh make.sh
```

Before compilation, you need to ensure that the line 3 in make.sh is the correct CUDA path in your machine. And you should also add CUDA path in the PATH, LD_LIBRARY_PATH, and C_INCLUDE_PATH before compilation. The following is the example on my machine. You should modify the paths added to the environment for your situation.

```shell
line 3: CUDA_PATH=/opt/cuda/cuda-8.0/
```

```shell
export PATH='/opt/cuda/cuda-8.0/bin:'"$PATH"
export LD_LIBRARY_PATH='/opt/cuda/cuda-8.0/lib64:'"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export C_INCLUDE_PATH='/opt/cuda/cuda-8.0/include:'"${C_INCLUDE_PATH}"
```

