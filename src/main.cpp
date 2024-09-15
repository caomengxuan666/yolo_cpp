#include <iostream>
#include "yolo_cmx.h"
#include"utils.h"
#include <thread>

int main() {
    YOLO_CMX yolo_model(yolo_nets[1]);
    std::string folder_path_img = "../dataset/test/img/train2017";
    std::string folder_path_video = "../dataset/test/video";

    std::thread _thread1([&]() {
        detect_folder(folder_path_video, yolo_model,VIDEO);
    });

    _thread1.join();


    return 0;

}
