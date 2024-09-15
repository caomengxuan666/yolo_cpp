//
// Created by CaoMengxuan on 2024/9/13.
//

#ifndef YOLO_CPP_LOCKFREEQUEUE_HPP
#define YOLO_CPP_LOCKFREEQUEUE_HPP
#include <atomic>

namespace cmx{

    /*
     * @brief 无锁队列，采用原子操作，线程安全。并没有依赖std::queue而是自己用链表做了实现，更加快速
     * @note 最好在客户端定义这个队列，不要在函数中定义，否则会因为函数调用栈的销毁而丢失数据，不用堆内存是为了队列的最快速度
     * @param T 参数类型，如果有必要的话可以做特化
     */
    template <typename T>
    class LockFreeQueue {
    private:
        struct Node {
            T data;
            Node* next;

            Node(T value) : data(value), next(nullptr) {}
        };

        std::atomic<Node*> head;
        std::atomic<Node*> tail;

    public:
        LockFreeQueue() {
            Node* dummy = new Node(T());  // 使用哑节点初始化
            head.store(dummy);
            tail.store(dummy);
        }

        ~LockFreeQueue() {
            while (Node* node = head.load()) {
                head.store(node->next);
                delete node;
            }
        }

        // 入队操作
        void enqueue(T value) {
            Node* newNode = new Node(value);
            Node* oldTail = tail.exchange(newNode);
            oldTail->next = newNode;
        }

        // 顺序出队操作，输入一个新节点，返回是否成功出队
        bool dequeue(T& result) {
            Node* oldHead = head.load();
            Node* nextNode = oldHead->next;

            if (nextNode == nullptr) {
                // 队列为空
                return false;
            }

            result = nextNode->data;
            head.store(nextNode);
            delete oldHead;  // 删除哑节点
            return true;
        }

        bool empty(){
            return head.load()->next == nullptr;
        }

        T&front(){
            return head.load()->next->data;
        }

        //实时视频用这个
        T& back(){
            return tail.load()->data;
        }
    };

}

#endif//YOLO_CPP_LOCKFREEQUEUE_HPP
