#pragma once

#include "Net_config.h"
#include "ThreadPool.hpp"
#include "darknet.h"
#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>


class YOLO_CMX {

public:
    YOLO_CMX() = delete;
    YOLO_CMX(const Net_config& config) {

        std::cout << "Net use " << config.netname << std::endl;

        // 检查 netname 长度
        if (config.netname.length() >= sizeof(this->netname)) {
            std::cerr << "Error: netname string is too long." << std::endl;
            return;
        }
        strcpy_s(this->netname, config.netname.c_str());

        // 检查文件是否可以打开
        std::ifstream ifs(config.classesFile.c_str());
        if (!ifs.is_open()) {
            std::cerr << "Error: Could not open classes file: " << config.classesFile << std::endl;
            return;
        }

        std::string line;
        while (getline(ifs, line)) this->classes.push_back(line);

        if (!std::ifstream(config.modelConfiguration) || !std::ifstream(config.modelWeights)) {
            std::cerr << "Error: Could not open model configuration or weights file." << std::endl;
            return;
        }

        // 加载网络模型
        this->net = cv::dnn::readNetFromDarknet(config.modelConfiguration, config.modelWeights);
        if (this->net.empty()) {
            std::cerr << "Error: Could not load the network from files." << std::endl;
            return;
        }

        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        device = "CPU";

        //置信度设置
        this->confThreshold = config.confThreshold;
        this->nmsThreshold = config.nmsThreshold;
        this->inpWidth = config.inpWidth;
        this->inpHeight = config.inpHeight;
    }

    //默认使用GPU 如果opencv没有启用CUDA支持，也会自动换回CPU
    inline void detect(cv::Mat frame, bool showRaw = false, bool save = false, bool gpu = false, bool dbg = true) {
        auto raw_size = frame.size();
        //std::cout << "推理图像原始宽度: " << raw_size.width << "\t 推理图像原始高度: " << raw_size.height << std::endl;

        cv::Mat blob;

        // 使用 GPU 推理
        static bool init = false;
        if (!init) {
            std::cout << "初始化成功" << std::endl;
            if (gpu) {
                this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);// 设置后端为 CUDA
                this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);  // 设置目标为 CUDA
                device = "GPU";
            }


            if (this->net.empty()) {
                std::cerr << "Error: Network is empty. Model may not have loaded correctly." << std::endl;
                return;
            }
        }
        init = true;
        //this->inpWidth=416;
        //this->inpHeight=416;
        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(this->inpWidth, this->inpHeight), cv::Scalar(0, 0, 0), true, false);
        // 打印 Blob 形状
        //std::cout << "Blob shape: " << blob.size << std::endl;

        if (showRaw)
            imshow("Raw_Pic", frame);

        //std::cout << "inpWidth: " << inpWidth << std::endl;
        //std::cout << "inpHeight: " << inpHeight << std::endl;

        // 输入 blob 到网络中
        this->net.setInput(blob);
        static std::vector<cv::String> outputLayerNames = this->net.getUnconnectedOutLayersNames();
        if (outputLayerNames.empty()) {
            std::cerr << "Error: Unable to retrieve output layer names." << std::endl;
            return;
        }
        // 执行前向推理
        static std::vector<cv::Mat> outs;
        this->net.forward(outs, outputLayerNames);
        // 进行后处理
        this->postprocess(frame, outs);


        // 将 frame 调整到原始输入大小
        cv::resize(frame, frame, raw_size);

        // 获取推理时间
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("%s Inference time : %.2f ms", this->netname, t);
        //fps
        std::string fps_label = cv::format("FPS: %.2f", 1000 / t);
        putText(frame, label, cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
        putText(frame, fps_label, cv::Point(0, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
        //使用的yolo版本
        putText(frame, netname, cv::Point(0, 90), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
        //使用的硬件
        std::string device_name = "detect_device:" + device;
        putText(frame, device_name, cv::Point(0, 120), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
        // 保存推理结果
        if (save)
            imwrite(cv::format("%s_out.jpg", this->netname), frame);
    }


    void detect_video(cv::VideoCapture &video, bool save = false) {
        //设置video的尺寸
        video.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        video.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        //设置video的fps
        video.set(cv::CAP_PROP_FPS, 30);

        if (!video.isOpened()) {
            std::cerr << "Error: Video capture not opened." << std::endl;
            return;
        }

        std::atomic<bool> stopFlag(false);
        std::queue<cv::Mat> frameQueue;
        std::mutex queueMutex;

        // 使用 lambda 捕获 this 指针

        std::thread displayThread([this, &frameQueue, &stopFlag, &queueMutex]() {
            this->displayVideo(frameQueue, stopFlag, queueMutex);
        });

        cv::Mat frame;
        cv::VideoWriter writer;
        if (save) {
            int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');                                 // 编码格式
            double fps = video.get(cv::CAP_PROP_FPS);                                                // 视频帧率
            cv::Size size(video.get(cv::CAP_PROP_FRAME_WIDTH), video.get(cv::CAP_PROP_FRAME_HEIGHT));// 视频尺寸
            std::string outputFilename = "output.avi";                                               // 输出文件名
            writer.open(outputFilename, codec, fps, size, true);
        }

        while (video.read(frame)) {
            //设置frame 的尺寸
            //cv::resize(frame,frame, cv::Size(640, 480));
            if (frame.empty()) {
                std::cerr << "Error: Frame is empty." << std::endl;
                stopFlag = true;
                break;
            }

            cv::Mat detectFrame = frame.clone();
            detect(detectFrame);// 推理处理

            if (save) {
                writer.write(detectFrame);// 写入帧
            }

            // 将推理结果放入队列供显示线程使用
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                frameQueue.push(detectFrame);
            }

            if (stopFlag) break;
        }

        displayThread.join();

        video.release();
        if (save) {
            writer.release();// 释放视频写入器
        }
        cv::destroyAllWindows();
    }

    inline void detect_video(cv::Mat &frame) {
        cv::Mat detectFrame = frame.clone();
        detect(detectFrame);// 推理处理
        cv::imshow("detect_frame", detectFrame);
        cv::waitKey(1);
    }


    void train(char *weight_file = nullptr, int *using_gpu = (int *) 0, int gpu_num = 1, int batchSize = 64, float thr = 0.5f, float iouThresh = 0.4f, int not_show = 0, int cl = 1, int calcMap = 0, char *save_path = nullptr, int map_epoch = 0, int show_img = 0, int benchmark_layer = 0, int m_jpeg = 0) {
        char *datacfg = const_cast<char *>(config.classesFile.c_str());       // 数据文件路径
        char *cfgfile = const_cast<char *>(config.modelConfiguration.c_str());// 网络配置文件路径
        char *weightfile = weight_file;                                       // 预训练权重文件，可以为 nullptr 表示从头开始训练
        int *gpus = using_gpu;                                                // GPU 设置，可以为 nullptr 使用默认
        int ngpus = gpu_num;                                                  // GPU 数量，0 表示不使用 GPU
        int clear = cl;                                                       // 是否清除上次的训练状态，1 表示清除
        int dont_show = not_show;                                             // 是否不显示训练过程，0 表示显示
        int calc_map = calcMap;                                               // 是否计算 mAP，0 表示不计算
        float thresh = thr;                                                   // 置信度阈值
        float iou_thresh = iouThresh;                                         // IOU 阈值
        int mjpeg_port = mjpeg_port;                                          // MJPEG 端口，0 表示不使用
        int show_imgs = show_img;                                             // 是否显示图像，0 表示不显示
        int benchmark_layers = benchmark_layer;                               // 是否进行层级性能测试，0 表示不进行
        char *chart_path = save_path;                                         // 图表保存路径，可以为 nullptr 不保存
        int mAP_epochs = map_epoch;                                           // mAP 计算周期
        int batch_size = batchSize;                                           //训练批次

        // 启动训练过程
        network *net = load_network(cfgfile, weightfile, clear);
        if (net == nullptr) {
            std::cerr << "Error: Could not load network." << std::endl;
            return;
        }

        srand(time(0));
        set_batch_network(net, batch_size);// 设置 batch 大小

        // 开始训练
        train_detector(datacfg, cfgfile, weightfile, gpus, ngpus, clear, dont_show, calc_map, thresh, iou_thresh, mjpeg_port, show_imgs, benchmark_layers, chart_path, mAP_epochs);

        // 释放资源
        free_network(*net);
    }


private:
    Net_config config;
    float confThreshold;
    float nmsThreshold;
    int inpWidth = 416;
    int inpHeight = 416;
    char netname[20];
    std::vector<std::string> classes;
    cv::dnn::Net net;
    std::string device;


    void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs) {
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (size_t i = 0; i < outs.size(); ++i) {
            // Scan through all the bounding boxes output from the network and keep only the
            // ones with high confidence scores. Assign the box's class label as the class
            // with the highest score for the box.
            float *data = (float *) outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double confidence;
                // Get the value and location of the maximum score
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > this->confThreshold) {
                    int centerX = (int) (data[0] * frame.cols);
                    int centerY = (int) (data[1] * frame.rows);
                    int width = (int) (data[2] * frame.cols);
                    int height = (int) (data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float) confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }

        // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // lower confidences
        std::vector<int> indices;
        //输出NMS的所有信息
        cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
                           box.x + box.width, box.y + box.height, frame);
        }
    }
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame, int dbg = true) {
        // 定义颜色映射
        constexpr size_t default_seed = 20040820;

        static std::default_random_engine engine(default_seed);// 固定随机数种子
        static std::uniform_int_distribution<int> dist(0, 255);

        // 根据类别生成独特的颜色
        cv::Scalar color;
        if (classId >= 0) {// 确保类别 ID 是有效的
            // 使用哈希函数生成独特的颜色
            unsigned int hashValue = std::hash<int>()(classId);
            color = cv::Scalar(
                    (hashValue >> 16) % 256,
                    (hashValue >> 8) % 256,
                    hashValue % 256);
        } else {
            color = cv::Scalar(0, 0, 255);// 如果类别 ID 无效，默认使用红色
        }

        // 绘制边界框
        cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), color, 1);

        // 获取标签文本
        std::string label = cv::format("%.2f", conf);
        if (!this->classes.empty()) {
            CV_Assert(classId < (int) this->classes.size());
            label = this->classes[classId] + ":" + label;
        }

        // 显示标签
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = std::max(top, labelSize.height);
        cv::rectangle(frame, cv::Point(left, top - int(1.6 * labelSize.height)), cv::Point(left + int(1.5 * labelSize.width), top + baseLine), color, cv::FILLED);
        cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_TRIPLEX, 0.75, cv::Scalar(255, 255, 255), 1);// 文本颜色为白色

        //打印所有的识别到的名称，矩形框范围，置信度
        std::cout << "识别到：" << label << std::endl;
    }
    // 显示函数
    void displayVideo(std::queue<cv::Mat> &frameQueue, std::atomic<bool> &stopFlag, std::mutex &queueMutex) {
        static const std::string kWinName = "检测结果，按q退出";

        while (!stopFlag) {
            std::unique_lock<std::mutex> lock(queueMutex);

            if (!frameQueue.empty()) {
                cv::Mat frame = frameQueue.front();
                frameQueue.pop();
                lock.unlock();

                cv::imshow(kWinName, frame);
                if (cv::waitKey(1) == 'q') {
                    stopFlag = true;
                    break;
                }
                if (stopFlag == true) {
                    break;
                }
            } else {
                lock.unlock();
            }
        }
    }
};
