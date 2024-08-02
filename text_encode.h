//
// Created by root on 8/2/24.
//

#ifndef CLIP_CPP_TEXT_ENCODE_H
#define CLIP_CPP_TEXT_ENCODE_H
#include "tokenizer.h"
#include "vocab.h"

void encode_text(std::vector<std::string> input_text,std::vector<int>& output)
{
    CLIPTokenizer tokenizer(VERSION_1_x);
    std::string str(reinterpret_cast<char*>(merges_utf8_c_str),sizeof(merges_utf8_c_str));
    tokenizer.load_from_merges(str);
    auto on_new_token_cb = [](std::string& str, std::vector<int32_t>& tokens) -> bool {
        // 可以在这里进行自定义处理，返回 true 可以跳过该 token 的处理
        return false;
    };
    output = tokenizer.tokenize(input_text[0], on_new_token_cb);
    output.push_back(49407);
}


#endif //CLIP_CPP_TEXT_ENCODE_H
