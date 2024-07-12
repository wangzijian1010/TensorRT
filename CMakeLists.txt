cmake_minimum_required(VERSION 3.16.0)
project(yolov5 VERSION 0.1.0)

include(CTest)
enable_testing()

set(DEPRECATION_FLAG "-Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DEPRECATION_FLAG}")

# CUDA
set(CUDA_DIR /usr/local/cuda)
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)

# TensorRT
set(TENSORRT_ROOT /usr/local/TensorRT-8.6.1.6)
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)


# OpenCV
set(Opencv_DIR /home/ai-test1/wangzijian/lite.ai.toolkit/third_party/opencv)
include_directories(${Opencv_DIR}/include/opencv4/)
link_directories(${Opencv_DIR}/lib)

add_executable(model_gen model_gen.cpp logger.h logging.h)
add_executable(model_infer infer.cpp logger.h logging.h)
add_executable(model_infer_yolov8 infer_yolov8.cpp logger.h logging.h)
add_executable(model_infer_yolov8face yoloface_v8.cpp logger.h logging.h)




target_link_libraries(
        model_gen nvinfer nvonnxparser nvinfer_plugin cudart  opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_dnn
)
# Ensure the library path is included
target_link_libraries(model_gen ${TENSORRT_ROOT}/lib/libnvinfer_builder_resource.so.8.6.1)


target_link_libraries(
        model_infer nvinfer nvonnxparser nvinfer_plugin cudart  opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_dnn
)
# Ensure the library path is included
target_link_libraries(model_infer ${TENSORRT_ROOT}/lib/libnvinfer_builder_resource.so.8.6.1)

target_link_libraries(
        model_infer_yolov8 nvinfer nvonnxparser nvinfer_plugin cudart  opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_dnn
)
# Ensure the library path is included
target_link_libraries(model_infer_yolov8 ${TENSORRT_ROOT}/lib/libnvinfer_builder_resource.so.8.6.1)

target_link_libraries(
        model_infer_yolov8face nvinfer nvonnxparser nvinfer_plugin cudart  opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_dnn
)
# Ensure the library path is included
target_link_libraries(model_infer_yolov8face ${TENSORRT_ROOT}/lib/libnvinfer_builder_resource.so.8.6.1)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
