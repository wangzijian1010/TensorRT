//
// Created by wangzijian on 8/2/24.
//

#ifndef CLIP_CPP_CLIP_H
#define CLIP_CPP_CLIP_H
#include "iostream"
#include "vector"
#include "onnxruntime_cxx_api.h"
#include "ort_config.h"

class clip {
public:
    clip(const std::string _onnx_path, unsigned int _num_threads = 1);

public:
    std::vector<const char *> input_node_names;
    std::vector<std::string> input_node_names_;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::size_t input_tensor_size = 1;
    std::vector<std::vector<int64_t>> input_node_dims; // >=1 inputs.
    Ort::Session *ort_session = nullptr;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::AllocatorWithDefaultOptions allocator;

public:
    void inference(std::vector<int> input,std::vector<float> &output);



};


#endif //CLIP_CPP_CLIP_H
