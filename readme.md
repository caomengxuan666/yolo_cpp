# 项目概述
本项目是一个完全基于OPENCV DNN和DARKNET的yolo的C++模型，darknet框架是完全没有任何依赖的，并且性能极其强大。
为了支持更快的解码，我引入了ffmpeg来直接操作硬解码部分。
本项目准备至少兼容两个大模块

1.模型训练
正在开发当中，目前只支持单线程

2.模型的推理
支持V3 V4 V7等YOLO模型，注意用到V5及其以上的模型，如果没有预训练模型可以去对应仓库去找.pt模型然后转为.weigths模型

* 支持图片推理，你只需要把图片放在图片文件夹下
* 支持视频推理，你只需要把视频放在视频文件夹下
* 未来将支持本地摄像头推理
* 未来将支持远程摄像头推理
* 未来将支持从远程服务器上拉流推理

## 环境
使用OPENCV4.10.0+CONTIRB+DNN+CUDA，编译器是MSVC，平台是WINDOWS

由于darknet需要pthread,而MSVC没有这个库，我自己编译了一个pthread库，不需要你额外配置

如果你是linux+gcc/g++方案，修改cmake配置也能进行编译

如果你使用的并非CUDA，而是采用CPU推理，也能实现CPU版本的推理，默认是GPU，并不需要修改代码

##使用方式

需要你自己编译OPENCV，版本4.X即可

同时，你必须编译opencv_contrib和dnn模块

darknet已经编译了cuda gpu支持，你可以直接调用。

**请注意**

* 如果你的OPENCV版本并非4.10.0，请你自行替换opencvworld.dll和opencvworld.lib，用你本机的替换

* CMAKE中的OPENCV路径也要根据你的本机进行修改

* 如果你使用的并非msvc，所有的库都需要你重新编译

* 如果你使用的是mingw，请注意mingw并不直接支持cuda，也就是说opencv_dnn模块无法在mingw下成功编译，你只能使用gpu推理，如果你是在linux平台使用gnu工具链，那么你并不受影响。不过需要调整架构重新编译。

* 如果确实某些库文件，可以直接在我的relaese中去获取

## 参数设置
你可以调整使用的模型和模型的weight文件，以及NMS和置信度的阈值。

weight文件需要自行下载
[https://pjreddie.com/media/files/yolov3.weights](https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights)

可以下载release直接使用
