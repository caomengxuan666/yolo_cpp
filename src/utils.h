#pragma once
#include "yolo_cmx.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <regex>

static enum detect_mode {
    IMG = 0,
    VIDEO = 1,
    CAMERA = 2
};

// 正则表达式匹配常见图像格式
static const std::regex kImageFileRegex("\\.(bmp|gif|jpg|jpeg|png|tiff|webp)$");

// 匹配视频格式
static const std::regex kVideoFileRegex("\\.(avi|mp4|mov|mkv|wmv|flv|rmvb|rm|webm|mpeg|mpg)$");

static void detect_folder(const std::string &folder_path, YOLO_CMX &yolo_model, detect_mode mode = IMG) {
    namespace fs = std::filesystem;
    for (const auto &entry: fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string imgpath = entry.path().string();

            if (mode == VIDEO) {
                //检查文件是否为视频文件
                if (std::regex_search(imgpath, kVideoFileRegex)) {
                    std::cout << "Detecting video: " << imgpath << std::endl;
                    cv::VideoCapture videoReader(imgpath);
                    //ThreadPool pool(1);
                    yolo_model.detect_video(videoReader);
                }
            } else {
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
                    cv::resizeWindow(kWinName, srcimg.cols, srcimg.rows);// 设置窗口
                    cv::imshow(kWinName, srcimg);
                    auto q = cv::waitKey();
                    if (q == 'q') break;
                    cv::destroyAllWindows();
                }
            }
        }
    }
}