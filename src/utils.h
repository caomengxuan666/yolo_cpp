#pragma once
#include "yolo_cmx.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <regex>
#include "ffmpeg_video.hpp"


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
            //替换'\'
            imgpath = std::regex_replace(imgpath, std::regex("\\\\"), "/");

            if (mode == VIDEO) {
                //检查文件是否为视频文件
                if (std::regex_search(imgpath, kVideoFileRegex)) {
                    std::cout << "Detecting video: " << imgpath <<"\n";

                    AVCodecContext* codec_ctx = nullptr;
                    int video_stream_index = -1;

                    AVFormatContext* format_ctx = cmx::init_ffmpeg(imgpath, &codec_ctx, video_stream_index);
                    if (!format_ctx) {
                        std::cerr << "Failed to initialize FFmpeg context." << std::endl;
                        return;
                    }

                    std::thread video_thread([=]() {
                        cmx::video_convert_to_mat(format_ctx, codec_ctx, video_stream_index);
                    });

                    std::thread read_mat_thread([&](){
                        while(1){
                            if(cmx::video_queue.empty()) continue;
                            cv::Mat frame=cmx::video_queue.back();
                            yolo_model.detect_video(frame);
                        }
                    });
                    video_thread.join();
                    read_mat_thread.join();
                    avcodec_free_context(&codec_ctx);
                    avformat_close_input(&format_ctx);
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