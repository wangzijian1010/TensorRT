//
// Created by root on 8/2/24.
//

#include "iostream"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "clip.h"
#include "tokenizer.h"
#include "vocab.h"

int main()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeTest");
    std::cout<<"hello"<<std::endl;
    std::string onnx_path = "/workspace/clip.cpp/models/clip_text_model_vitb32.onnx";

    clip clip_test(onnx_path,1);

    std::string text = "i am not good at cpp";

    CLIPTokenizer tokenizer(VERSION_1_x);
    std::string str(reinterpret_cast<char*>(merges_utf8_c_str),sizeof(merges_utf8_c_str));
    tokenizer.load_from_merges(str);

    auto on_new_token_cb = [](std::string& str, std::vector<int32_t>& tokens) -> bool {
        // 可以在这里进行自定义处理，返回 true 可以跳过该 token 的处理
        return false;
    };

    std::vector<int> encoded_tokens = tokenizer.tokenize(text, on_new_token_cb);

    encoded_tokens.push_back(49407);

    std::cout<<"hello"<<std::endl;

    std::vector<float> output;

    clip_test.inference(encoded_tokens,output);


}