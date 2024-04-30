# Installation

This document contains detailed instructions for installing dependencies for GLAD. The code is tested on an Ubuntu 18.04 system with Nvidia GeForce RTX 3070.

### Requirments
* Conda with Python 3.7
* PyTorch 1.8.1
* CUDA 11.1
* TorchVision 1.9.1
* OpenCV 4.4.0.46
* Tensorrt 7.2.2.3

#### Create environment and activate
```bash
conda create --name GLAD python=3.7
source activate GLAD
```

#### Install numpy/pytorch/opencv/tensorrt
```bash
conda install numpy
conda install pytorch=1.8.1 torchvision cudatoolkit=11.1 -c pytorch
pip install opencv-python==4.4.0.46
pip install tensorrt==7.2.2.3
```

## More Information
About tensorrt please see the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
the YOLOv5 version is v6.0 in this project



