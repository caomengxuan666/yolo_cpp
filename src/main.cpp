#include <iostream>
#include "yolo_cmx.h"
#include"utils.h"

int main() {
    YOLO_CMX yolo_model(yolo_nets[1]);
    //std::string folder_path = "../dataset/test/img/train2017";
    std::string folder_path = "../dataset/test/video";
    detect_folder(folder_path, yolo_model,VIDEO);

    return 0;
}
