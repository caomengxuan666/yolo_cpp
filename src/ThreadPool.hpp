//
// Created by CaoMengxuan on 2024/9/12.
//

#ifndef YOLO_CPP_THREADPOOL_HPP
#define YOLO_CPP_THREADPOOL_HPP

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>

class ThreadPool {
public:
    ThreadPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            threads.emplace_back([this] { workerThread(); });
        }
    }

    ~ThreadPool() {
        stop = true;
        condition.notify_all();
        for (auto& thread : threads) {
            thread.join();
        }
    }

    template<typename Func, typename... Args>
    void enqueue(Func&& f, Args&&... args) {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.emplace(std::forward<Func>(f), std::forward<Args>(args)...);
        lock.unlock();
        condition.notify_one();
    }

private:
    void workerThread() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                condition.wait(lock, [this] { return stop || !tasks.empty(); });
                if (stop && tasks.empty())
                    return;
                task = std::move(tasks.front());
                tasks.pop();
            }
            task();
        }
    }

    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop = false;
};

#endif // YOLO_CPP_THREADPOOL_HPP
