#include <iostream>
#include "yolo_cmx.h"
#include"utils.h"

int main() {
    //这里默认用v3,我只下载了v3的weight
    YOLO_CMX yolo_model(yolo_nets[0]);
    std::string folder_path = "../train2017";

    detect_folder(folder_path, yolo_model);

    return 0;
}
