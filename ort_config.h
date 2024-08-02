//
// Created by root on 8/2/24.
//

#ifndef CLIP_CPP_ORT_CONFIG_H
#define CLIP_CPP_ORT_CONFIG_H
#include "onnxruntime_cxx_api.h"
#include "iostream"

inline static std::string OrtCompatiableGetInputName(size_t index, OrtAllocator* allocator,
                                                     Ort::Session *ort_session) {
#if ORT_API_VERSION >= 14
    return std::string(ort_session->GetInputNameAllocated(index, allocator).get());
#else
    return std::string(ort_session->GetInputName(i, allocator));
#endif
}

inline static std::string OrtCompatiableGetOutputName(size_t index, OrtAllocator* allocator,
                                                      Ort::Session *ort_session) {
#if ORT_API_VERSION >= 14
    return std::string(ort_session->GetOutputNameAllocated(index, allocator).get());
#else
    return std::string(ort_session->GetOutputName(i, allocator));
#endif
}

#endif //CLIP_CPP_ORT_CONFIG_H
