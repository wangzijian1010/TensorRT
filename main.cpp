//
// Created by root on 8/2/24.
//

#include "iostream"
#include "onnxruntime_cxx_api.h"
#include "clip.h"
#include "text_encode.h"

int main()
{

    std::string onnx_path = "/workspace/clip.cpp/models/clip_text_model_vitb32.onnx";
    clip clip_test(onnx_path,1);

    std::string text = "i am not good at cpp";
    std::vector<std::string> input_text;
    input_text.push_back(text);
    std::vector<int> output_token;
    encode_text(input_text,output_token);

    std::vector<float> output;
    clip_test.inference(output_token,output);

    std::cout<<"infer done!"<<std::endl;

}