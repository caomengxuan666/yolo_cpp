#pragma once
#include<regex>
#include<filesystem>
#include<opencv.hpp>
#include"yolo_cmx.h"

// 正则表达式匹配常见图像格式
static const std::regex kImageFileRegex("\\.(bmp|gif|jpg|jpeg|png|tiff|webp)$");

static void detect_folder(const std::string& folder_path, YOLO_CMX& yolo_model) {
    namespace fs = std::filesystem;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string imgpath = entry.path().string();

            // 检查文件是否为图像文件
            if (std::regex_search(imgpath, kImageFileRegex)) {
                cv::Mat srcimg = cv::imread(imgpath);

                if (srcimg.empty()) {
                    std::cerr << "Error loading image: " << imgpath << std::endl;
                    continue;
                }

                auto raw_size = srcimg.size();

                yolo_model.detect(srcimg);

                static const std::string kWinName = "检测结果，按q退出";
                cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
                cv::resizeWindow(kWinName, srcimg.cols, srcimg.rows);  // 设置窗口
                cv::imshow(kWinName, srcimg);
                auto q = cv::waitKey();
                if (q == 'q')break;
                cv::destroyAllWindows();
            }
        }
    }
}