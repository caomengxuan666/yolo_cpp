# 项目概述

## 环境
使用OPENCV4.10.0+CONTIRB+DNN+CUDA，编译器是MSVC，平台是WINDOWS

由于darknet需要pthread,而MSVC没有这个库，我自己编译了一个pthread库，不需要你额外配置

如果你是linux+gcc/g++方案，修改cmake配置也能进行编译

如果你使用的并非CUDA，而是采用CPU推理，也能实现CPU版本的推理，默认是GPU，并不需要修改代码

##使用方式
需要你自己编译OPENCV，版本4.X即可

**请注意**

如果你的OPENCV版本并非4.10.0，请你自行替换opencvworld.dll和opencvworld.lib，用你本机的替换

CMAKE中的OPENCV路径也要根据你的本机进行修改

## 参数设置
你可以调整使用的模型和模型的weight文件，以及NMS和置信度的阈值。

weight文件需要自行下载
[https://pjreddie.com/media/files/yolov3.weights](https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights)
