cmake_minimum_required(VERSION 3.0)
project(TensorRTLearn)

set(CMAKE_CXX_STANDARD 17)

# find TesnorRT
set(TENSORRT_DIR /home/wangzijian/Desktop/TensorRT-8.6.0.12)
include_directories(/home/wangzijian/Desktop/TensorRT-8.6.0.12/targets/x86_64-linux-gnu/include)

add_executable(TensorRTLearn main.cpp)

target_link_libraries(TensorRTLearn /home/wangzijian/Desktop/TensorRT-8.6.0.12/targets/x86_64-linux-gnu/lib/libnvinfer.so.8.6.0)
target_link_libraries(TensorRTLearn /home/wangzijian/Desktop/TensorRT-8.6.0.12/targets/x86_64-linux-gnu/lib/libnvonnxparser.so.8.6.0)
