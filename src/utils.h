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

// ������ʽƥ�䳣��ͼ���ʽ
static const std::regex kImageFileRegex("\\.(bmp|gif|jpg|jpeg|png|tiff|webp)$");

// ƥ����Ƶ��ʽ
static const std::regex kVideoFileRegex("\\.(avi|mp4|mov|mkv|wmv|flv|rmvb|rm|webm|mpeg|mpg)$");

static void detect_folder(const std::string &folder_path, YOLO_CMX &yolo_model, detect_mode mode = IMG) {
    namespace fs = std::filesystem;
    for (const auto &entry: fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string imgpath = entry.path().string();

            if (mode == VIDEO) {
                //����ļ��Ƿ�Ϊ��Ƶ�ļ�
                if (std::regex_search(imgpath, kVideoFileRegex)) {
                    std::cout << "Detecting video: " << imgpath << std::endl;
                    cv::VideoCapture videoReader(imgpath);
                    //ThreadPool pool(1);
                    yolo_model.detect_video(videoReader);
                }
            } else {
                // ����ļ��Ƿ�Ϊͼ���ļ�
                if (std::regex_search(imgpath, kImageFileRegex)) {
                    cv::Mat srcimg = cv::imread(imgpath);

                    if (srcimg.empty()) {
                        std::cerr << "Error loading image: " << imgpath << std::endl;
                        continue;
                    }

                    auto raw_size = srcimg.size();

                    yolo_model.detect(srcimg);

                    static const std::string kWinName = "���������q�˳�";
                    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
                    cv::resizeWindow(kWinName, srcimg.cols, srcimg.rows);// ���ô���
                    cv::imshow(kWinName, srcimg);
                    auto q = cv::waitKey();
                    if (q == 'q') break;
                    cv::destroyAllWindows();
                }
            }
        }
    }
}