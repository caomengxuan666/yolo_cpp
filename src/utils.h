#pragma once
#include<regex>
#include<filesystem>
#include<opencv.hpp>
#include"yolo_cmx.h"

// ������ʽƥ�䳣��ͼ���ʽ
static const std::regex kImageFileRegex("\\.(bmp|gif|jpg|jpeg|png|tiff|webp)$");

static void detect_folder(const std::string& folder_path, YOLO_CMX& yolo_model) {
    namespace fs = std::filesystem;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string imgpath = entry.path().string();

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
                cv::resizeWindow(kWinName, srcimg.cols, srcimg.rows);  // ���ô���
                cv::imshow(kWinName, srcimg);
                auto q = cv::waitKey();
                if (q == 'q')break;
                cv::destroyAllWindows();
            }
        }
    }
}