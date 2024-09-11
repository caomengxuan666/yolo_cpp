cmake_minimum_required(VERSION 3.27)
project(yolo_cpp)

set(CMAKE_CXX_STANDARD 17)

# 设置头文件搜索路径
include_directories(
        include
        include/opencv2
)

# 查找 pthread 库（适用于 MSVC 的特殊处理）
if (MSVC)
    # Windows 上的 pthread 库
    find_library(PTHREAD_LIBRARY pthreadVC2 PATHS /lib REQUIRED)
else()
    # 其他平台的 pthread 库
    find_package(Threads REQUIRED)
endif()

# 查找 OpenCV 库
find_library(OpenCV_LIBRARIES NAMES opencv_world4100 PATHS /lib REQUIRED)

# 查找 Darknet 导入库（.lib 文件）
find_library(DARKNET_LIBRARY NAMES darknet.lib PATHS /lib REQUIRED)

# 添加可执行文件
add_executable(yolo_cpp src/main.cpp)

# 链接库文件
target_link_libraries(yolo_cpp
        ${OpenCV_LIBRARIES}
        ${DARKNET_LIBRARY}
        ${PTHREAD_LIBRARY}  # 仅在 MSVC 下有用
        ${CMAKE_THREAD_LIBS_INIT}  # 在其他平台上使用
)

# 复制 darknet.dll 到可执行文件输出目录
add_custom_command(TARGET yolo_cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/lib/darknet.dll" $<TARGET_FILE_DIR:yolo_cpp>
)

# 安装规则
install(TARGETS yolo_cpp DESTINATION ../bin)

# 查找并安装所有 DLL 文件
file(GLOB DLL_FILES "${CMAKE_SOURCE_DIR}/lib/*.dll")
install(FILES ${DLL_FILES} DESTINATION ../bin)
