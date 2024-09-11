#pragma once

#include "Net_config.h"
#include "darknet.h"
#include <fstream>
#include <iostream>
#include <opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>


class YOLO_CMX {

public:
    YOLO_CMX() = delete;
    YOLO_CMX(Net_config config) {

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

        //���Ŷ�����
        this->confThreshold = config.confThreshold;
        this->nmsThreshold = config.nmsThreshold;
        this->inpWidth = config.inpWidth;
        this->inpHeight = config.inpHeight;
    }

    //Ĭ��ʹ��GPU ���opencvû������CUDA֧�֣�Ҳ���Զ�����CPU
    void detect(cv::Mat &frame) {
        auto raw_size = frame.size();
        std::cout << "����ͼ��ԭʼ���: " << raw_size.width << "\t ����ͼ��ԭʼ�߶�: " << raw_size.height << std::endl;

        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(this->inpWidth, this->inpHeight), cv::Scalar(0, 0, 0), true, false);
        imshow("Raw_Pic", frame);

        std::cout << "inpWidth: " << inpWidth << std::endl;
        std::cout << "inpHeight: " << inpHeight << std::endl;

        // ʹ�� GPU ����
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);// ���ú��Ϊ CUDA
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);  // ����Ŀ��Ϊ CUDA

        // ���� blob ��������
        this->net.setInput(blob);

        if (this->net.empty()) {
            std::cerr << "Error: Network is empty. Model may not have loaded correctly." << std::endl;
            return;
        }

        std::vector<cv::String> outputLayerNames = this->net.getUnconnectedOutLayersNames();
        if (outputLayerNames.empty()) {
            std::cerr << "Error: Unable to retrieve output layer names." << std::endl;
            return;
        }

        // ִ��ǰ������
        std::vector<cv::Mat> outs;
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
        putText(frame, label, cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);

        // ����������
        imwrite(cv::format("%s_out.jpg", this->netname), frame);
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
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame) {
        //Draw a rectangle displaying the bounding box
        cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 1);

        //Get the label for the class name and its confidence
        std::string label = cv::format("%.2f", conf);
        if (!this->classes.empty()) {
            CV_Assert(classId < (int) this->classes.size());
            label = this->classes[classId] + ":" + label;
        }

        //Display the label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = std::max(top, labelSize.height);
        //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
        putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_TRIPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
    }
};
