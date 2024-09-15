#ifndef YOLO_CPP_FFMPEG_VIDEO_HPP
#define YOLO_CPP_FFMPEG_VIDEO_HPP

#include <LockFreeQueue.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ffmpeg.hpp>

namespace cmx {
    static LockFreeQueue<cv::Mat> video_queue;

    // 初始化视频解码器的函数
    static AVFormatContext* init_ffmpeg(const std::string& filename, AVCodecContext** codec_ctx, int& video_stream_index) {
        const char* input_filename = filename.c_str();

        avformat_network_init();
        AVFormatContext* format_ctx = nullptr;

        if (avformat_open_input(&format_ctx, input_filename, nullptr, nullptr) != 0) {
            std::cerr << "Could not open video file: " << input_filename << std::endl;
            return nullptr;
        }

        if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
            std::cerr << "Could not retrieve stream info from file" << std::endl;
            avformat_close_input(&format_ctx);
            return nullptr;
        }

        video_stream_index = -1;
        AVCodecParameters* codec_params = nullptr;
        const AVCodec* codec = nullptr;

        for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
            if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_index = i;
                codec_params = format_ctx->streams[i]->codecpar;
                codec = avcodec_find_decoder(codec_params->codec_id);
                break;
            }
        }

        if (video_stream_index == -1 || !codec) {
            std::cerr << "Unsupported codec or video stream not found!" << std::endl;
            avformat_close_input(&format_ctx);
            return nullptr;
        }

        *codec_ctx = avcodec_alloc_context3(codec);
        if (!(*codec_ctx) || avcodec_parameters_to_context(*codec_ctx, codec_params) < 0) {
            std::cerr << "Failed to initialize codec context" << std::endl;
            avcodec_free_context(codec_ctx);
            avformat_close_input(&format_ctx);
            return nullptr;
        }

        if (avcodec_open2(*codec_ctx, codec, nullptr) < 0) {
            std::cerr << "Could not open codec" << std::endl;
            avcodec_free_context(codec_ctx);
            avformat_close_input(&format_ctx);
            return nullptr;
        }

        return format_ctx;
    }

    // 视频转Mat帧
    static int video_convert_to_mat(AVFormatContext* format_ctx, AVCodecContext* codec_ctx, int video_stream_index, int height = -1, int width = -1) {
        AVFrame* frame = av_frame_alloc();
        AVPacket* packet = av_packet_alloc();
        struct SwsContext* sws_ctx;
        int fps=codec_ctx->framerate.num/codec_ctx->framerate.den;
        std::cout<<"当前实际视频帧数"<<fps<<std::endl;

        if (height != -1) {
            sws_ctx = sws_getContext(codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt, width, height, AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr, nullptr, nullptr);
        } else {
            sws_ctx = sws_getContext(codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt, codec_ctx->width, codec_ctx->height, AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr, nullptr, nullptr);
        }

        cv::Mat frame_mat = (height == -1) ? cv::Mat(codec_ctx->height, codec_ctx->width, CV_8UC3) : cv::Mat(height, width, CV_8UC3);

        int response = 0;

        while (av_read_frame(format_ctx, packet) >= 0) {
            if (packet->stream_index == video_stream_index) {
                response = avcodec_send_packet(codec_ctx, packet);
                if (response < 0) {
                    std::cerr << "Error sending packet to decoder" << std::endl;
                    continue;
                }

                response = avcodec_receive_frame(codec_ctx, frame);
                if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                    continue;
                } else if (response < 0) {
                    std::cerr << "Error receiving frame from decoder" << std::endl;
                    break;
                }

                uint8_t* dest[4] = { frame_mat.data, nullptr, nullptr, nullptr };
                int dest_linesize[4] = { static_cast<int>(frame_mat.step[0]), 0, 0, 0 };
                sws_scale(sws_ctx, frame->data, frame->linesize, 0, codec_ctx->height, dest, dest_linesize);

                std::this_thread::sleep_for(std::chrono::milliseconds(800/fps));
                video_queue.enqueue(frame_mat);
            }
            av_packet_unref(packet);
        }

        sws_freeContext(sws_ctx);
        av_frame_free(&frame);
        av_packet_free(&packet);

        return 0;
    }
}

#endif // YOLO_CPP_FFMPEG_VIDEO_HPP
