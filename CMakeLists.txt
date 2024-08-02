cmake_minimum_required(VERSION 3.29)
project(clip_cpp)

set(CMAKE_CXX_STANDARD 14)

set(THIRDPARTY /workspace/clip.cpp/thirdparty)

set(ONNXRUNTIME_DIR ${THIRDPARTY}/onnxruntime)
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)


add_executable(clip_cpp
        clip.cpp
        clip.h
        tokenizer.h
        main.cpp
        vocab.h
        )

target_link_libraries(clip_cpp onnxruntime)
