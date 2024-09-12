#pragma once
#include <string>

struct Net_config {
    float confThreshold = 0.5;
    float nmsThreshold = 0.4;
    int inpWidth;
    int inpHeight;
    std::string classesFile;
    std::string modelConfiguration;
    std::string modelWeights;
    std::string netname;
};


// 使用有效的索引
Net_config yolo_nets[] = {
        {0.5, 0.1, 416, 416, "../cfg/coco.names", "../cfg/yolov3.cfg", "../weight/yolov3.weights", "yolov3"},
        {0.5, 0.1, 416, 416, "../cfg/coco.names", "../cfg/yolov4-tiny.cfg", "../weight/yolov4-tiny.weights", "yolov4-tiny"},
        {0.5, 0.1, 416, 416, "../cfg/coco.names", "../cfg/yolov4.cfg", "../weight/yolov4.weights", "yolov4"}};
