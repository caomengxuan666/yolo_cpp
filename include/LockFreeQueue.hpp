//
// Created by CaoMengxuan on 2024/9/13.
//

#ifndef YOLO_CPP_LOCKFREEQUEUE_HPP
#define YOLO_CPP_LOCKFREEQUEUE_HPP
#include <atomic>

namespace cmx{

    /*
     * @brief �������У�����ԭ�Ӳ������̰߳�ȫ����û������std::queue�����Լ�����������ʵ�֣����ӿ���
     * @note ����ڿͻ��˶���������У���Ҫ�ں����ж��壬�������Ϊ��������ջ�����ٶ���ʧ���ݣ����ö��ڴ���Ϊ�˶��е�����ٶ�
     * @param T �������ͣ�����б�Ҫ�Ļ��������ػ�
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
            Node* dummy = new Node(T());  // ʹ���ƽڵ��ʼ��
            head.store(dummy);
            tail.store(dummy);
        }

        ~LockFreeQueue() {
            while (Node* node = head.load()) {
                head.store(node->next);
                delete node;
            }
        }

        // ��Ӳ���
        void enqueue(T value) {
            Node* newNode = new Node(value);
            Node* oldTail = tail.exchange(newNode);
            oldTail->next = newNode;
        }

        // ˳����Ӳ���������һ���½ڵ㣬�����Ƿ�ɹ�����
        bool dequeue(T& result) {
            Node* oldHead = head.load();
            Node* nextNode = oldHead->next;

            if (nextNode == nullptr) {
                // ����Ϊ��
                return false;
            }

            result = nextNode->data;
            head.store(nextNode);
            delete oldHead;  // ɾ���ƽڵ�
            return true;
        }

        bool empty(){
            return head.load()->next == nullptr;
        }

        T&front(){
            return head.load()->next->data;
        }

        //ʵʱ��Ƶ�����
        T& back(){
            return tail.load()->data;
        }
    };

}

#endif//YOLO_CPP_LOCKFREEQUEUE_HPP
