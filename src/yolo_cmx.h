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

        // ��� netname ����
        if (config.netname.length() >= sizeof(this->netname)) {
            std::cerr << "Error: netname string is too long." << std::endl;
            return;
        }
        strcpy_s(this->netname, config.netname.c_str());

        // ����ļ��Ƿ���Դ�
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

        // ��������ģ��
        this->net = cv::dnn::readNetFromDarknet(config.modelConfiguration, config.modelWeights);
        if (this->net.empty()) {
            std::cerr << "Error: Could not load the network from files." << std::endl;
            return;
        }

        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        device = "CPU";

        //���Ŷ�����
        this->confThreshold = config.confThreshold;
        this->nmsThreshold = config.nmsThreshold;
        this->inpWidth = config.inpWidth;
        this->inpHeight = config.inpHeight;
    }

    //Ĭ��ʹ��GPU ���opencvû������CUDA֧�֣�Ҳ���Զ�����CPU
    inline void detect(cv::Mat frame, bool showRaw = false, bool save = false, bool gpu = false, bool dbg = true) {
        auto raw_size = frame.size();
        //std::cout << "����ͼ��ԭʼ���: " << raw_size.width << "\t ����ͼ��ԭʼ�߶�: " << raw_size.height << std::endl;

        cv::Mat blob;

        // ʹ�� GPU ����
        static bool init = false;
        if (!init) {
            std::cout << "��ʼ���ɹ�" << std::endl;
            if (gpu) {
                this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);// ���ú��Ϊ CUDA
                this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);  // ����Ŀ��Ϊ CUDA
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
        // ��ӡ Blob ��״
        //std::cout << "Blob shape: " << blob.size << std::endl;

        if (showRaw)
            imshow("Raw_Pic", frame);

        //std::cout << "inpWidth: " << inpWidth << std::endl;
        //std::cout << "inpHeight: " << inpHeight << std::endl;

        // ���� blob ��������
        this->net.setInput(blob);
        static std::vector<cv::String> outputLayerNames = this->net.getUnconnectedOutLayersNames();
        if (outputLayerNames.empty()) {
            std::cerr << "Error: Unable to retrieve output layer names." << std::endl;
            return;
        }
        // ִ��ǰ������
        static std::vector<cv::Mat> outs;
        this->net.forward(outs, outputLayerNames);
        // ���к���
        this->postprocess(frame, outs);


        // �� frame ������ԭʼ�����С
        cv::resize(frame, frame, raw_size);

        // ��ȡ����ʱ��
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("%s Inference time : %.2f ms", this->netname, t);
        //fps
        std::string fps_label = cv::format("FPS: %.2f", 1000 / t);
        putText(frame, label, cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
        putText(frame, fps_label, cv::Point(0, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
        //ʹ�õ�yolo�汾
        putText(frame, netname, cv::Point(0, 90), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
        //ʹ�õ�Ӳ��
        std::string device_name = "detect_device:" + device;
        putText(frame, device_name, cv::Point(0, 120), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
        // ����������
        if (save)
            imwrite(cv::format("%s_out.jpg", this->netname), frame);
    }


    void detect_video(cv::VideoCapture &video, bool save = false) {
        //����video�ĳߴ�
        video.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        video.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        //����video��fps
        video.set(cv::CAP_PROP_FPS, 30);

        if (!video.isOpened()) {
            std::cerr << "Error: Video capture not opened." << std::endl;
            return;
        }

        std::atomic<bool> stopFlag(false);
        std::queue<cv::Mat> frameQueue;
        std::mutex queueMutex;

        // ʹ�� lambda ���� this ָ��

        std::thread displayThread([this, &frameQueue, &stopFlag, &queueMutex]() {
            this->displayVideo(frameQueue, stopFlag, queueMutex);
        });

        cv::Mat frame;
        cv::VideoWriter writer;
        if (save) {
            int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');                                 // �����ʽ
            double fps = video.get(cv::CAP_PROP_FPS);                                                // ��Ƶ֡��
            cv::Size size(video.get(cv::CAP_PROP_FRAME_WIDTH), video.get(cv::CAP_PROP_FRAME_HEIGHT));// ��Ƶ�ߴ�
            std::string outputFilename = "output.avi";                                               // ����ļ���
            writer.open(outputFilename, codec, fps, size, true);
        }

        while (video.read(frame)) {
            //����frame �ĳߴ�
            //cv::resize(frame,frame, cv::Size(640, 480));
            if (frame.empty()) {
                std::cerr << "Error: Frame is empty." << std::endl;
                stopFlag = true;
                break;
            }

            cv::Mat detectFrame = frame.clone();
            detect(detectFrame);// ������

            if (save) {
                writer.write(detectFrame);// д��֡
            }

            // ��������������й���ʾ�߳�ʹ��
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                frameQueue.push(detectFrame);
            }

            if (stopFlag) break;
        }

        displayThread.join();

        video.release();
        if (save) {
            writer.release();// �ͷ���Ƶд����
        }
        cv::destroyAllWindows();
    }

    inline void detect_video(cv::Mat &frame) {
        cv::Mat detectFrame = frame.clone();
        detect(detectFrame);// ������
        cv::imshow("detect_frame", detectFrame);
        cv::waitKey(1);
    }


    void train(char *weight_file = nullptr, int *using_gpu = (int *) 0, int gpu_num = 1, int batchSize = 64, float thr = 0.5f, float iouThresh = 0.4f, int not_show = 0, int cl = 1, int calcMap = 0, char *save_path = nullptr, int map_epoch = 0, int show_img = 0, int benchmark_layer = 0, int m_jpeg = 0) {
        char *datacfg = const_cast<char *>(config.classesFile.c_str());       // �����ļ�·��
        char *cfgfile = const_cast<char *>(config.modelConfiguration.c_str());// ���������ļ�·��
        char *weightfile = weight_file;                                       // Ԥѵ��Ȩ���ļ�������Ϊ nullptr ��ʾ��ͷ��ʼѵ��
        int *gpus = using_gpu;                                                // GPU ���ã�����Ϊ nullptr ʹ��Ĭ��
        int ngpus = gpu_num;                                                  // GPU ������0 ��ʾ��ʹ�� GPU
        int clear = cl;                                                       // �Ƿ�����ϴε�ѵ��״̬��1 ��ʾ���
        int dont_show = not_show;                                             // �Ƿ���ʾѵ�����̣�0 ��ʾ��ʾ
        int calc_map = calcMap;                                               // �Ƿ���� mAP��0 ��ʾ������
        float thresh = thr;                                                   // ���Ŷ���ֵ
        float iou_thresh = iouThresh;                                         // IOU ��ֵ
        int mjpeg_port = mjpeg_port;                                          // MJPEG �˿ڣ�0 ��ʾ��ʹ��
        int show_imgs = show_img;                                             // �Ƿ���ʾͼ��0 ��ʾ����ʾ
        int benchmark_layers = benchmark_layer;                               // �Ƿ���в㼶���ܲ��ԣ�0 ��ʾ������
        char *chart_path = save_path;                                         // ͼ����·��������Ϊ nullptr ������
        int mAP_epochs = map_epoch;                                           // mAP ��������
        int batch_size = batchSize;                                           //ѵ������

        // ����ѵ������
        network *net = load_network(cfgfile, weightfile, clear);
        if (net == nullptr) {
            std::cerr << "Error: Could not load network." << std::endl;
            return;
        }

        srand(time(0));
        set_batch_network(net, batch_size);// ���� batch ��С

        // ��ʼѵ��
        train_detector(datacfg, cfgfile, weightfile, gpus, ngpus, clear, dont_show, calc_map, thresh, iou_thresh, mjpeg_port, show_imgs, benchmark_layers, chart_path, mAP_epochs);

        // �ͷ���Դ
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
        //���NMS��������Ϣ
        cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
                           box.x + box.width, box.y + box.height, frame);
        }
    }
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame, int dbg = true) {
        // ������ɫӳ��
        constexpr size_t default_seed = 20040820;

        static std::default_random_engine engine(default_seed);// �̶����������
        static std::uniform_int_distribution<int> dist(0, 255);

        // ����������ɶ��ص���ɫ
        cv::Scalar color;
        if (classId >= 0) {// ȷ����� ID ����Ч��
            // ʹ�ù�ϣ�������ɶ��ص���ɫ
            unsigned int hashValue = std::hash<int>()(classId);
            color = cv::Scalar(
                    (hashValue >> 16) % 256,
                    (hashValue >> 8) % 256,
                    hashValue % 256);
        } else {
            color = cv::Scalar(0, 0, 255);// ������ ID ��Ч��Ĭ��ʹ�ú�ɫ
        }

        // ���Ʊ߽��
        cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), color, 1);

        // ��ȡ��ǩ�ı�
        std::string label = cv::format("%.2f", conf);
        if (!this->classes.empty()) {
            CV_Assert(classId < (int) this->classes.size());
            label = this->classes[classId] + ":" + label;
        }

        // ��ʾ��ǩ
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = std::max(top, labelSize.height);
        cv::rectangle(frame, cv::Point(left, top - int(1.6 * labelSize.height)), cv::Point(left + int(1.5 * labelSize.width), top + baseLine), color, cv::FILLED);
        cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_TRIPLEX, 0.75, cv::Scalar(255, 255, 255), 1);// �ı���ɫΪ��ɫ

        //��ӡ���е�ʶ�𵽵����ƣ����ο�Χ�����Ŷ�
        std::cout << "ʶ�𵽣�" << label << std::endl;
    }
    // ��ʾ����
    void displayVideo(std::queue<cv::Mat> &frameQueue, std::atomic<bool> &stopFlag, std::mutex &queueMutex) {
        static const std::string kWinName = "���������q�˳�";

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
