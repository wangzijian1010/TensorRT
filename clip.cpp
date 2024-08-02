//
// Created by wangzijian on 8/2/24.
//

#include "clip.h"
#include "fstream"



using namespace std;
using namespace Ort;

clip::clip(const std::string _onnx_path, unsigned int _num_threads) {

    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Face Detect");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
    std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    ort_session = new Ort::Session(env, _onnx_path.c_str(), sessionOptions); ////linux写法
    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();

    for (int i = 0; i < numInputNodes; i++)
    {

        Ort::AllocatedStringPtr input_name_Ptr = ort_session->GetInputNameAllocated(i, allocator);  /// 高版本onnxruntime的接口函数
        input_names.push_back(input_name_Ptr.get()); /// 高版本onnxruntime的接口函数
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }

    for (int i = 0; i < numOutputNodes; i++)
    {
        std::string outputname= "TEXT_EMBEDDING";
        output_names.push_back(outputname); /// 高版本onnxruntime的接口函数
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }

}



void save_tensor_to_file(const Ort::Value& tensor, const std::string& filename) {
    // 获取张量的类型和形状信息
    auto type_and_shape_info = tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = type_and_shape_info.GetShape();
    size_t element_count = type_and_shape_info.GetElementCount();

    ONNXTensorElementDataType type = type_and_shape_info.GetElementType();
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        std::cerr << "Unsupported tensor data type. Only int32 tensors are supported." << std::endl;
        return;
    }

    const int32_t* pdata = tensor.GetTensorData<int32_t>();

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < element_count; ++i) {
        file << pdata[i] << "\n";
    }
    file.close();
}




void clip::inference(std::vector<int> input, std::vector<float>& output) {
    int token_length = 77;
    std::vector<int32_t> text_features_input(token_length);

    // 将 input 中的元素转换为 int32_t 并存储在 text_features_input 中
    for (int i = 0; i < input.size(); ++i) {
        text_features_input[i] = static_cast<int32_t>(input[i]);
    }

    // 定义输入的形状（此处假设形状为 {1, 77}）
    std::vector<int64_t> input_node_dims1 = {1, 77};

    // 创建内存信息
    Ort::MemoryInfo allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 创建输入张量
    auto inputTensor = Ort::Value::CreateTensor<int32_t>(
            allocator_info,                    // 分配器信息
            text_features_input.data(),        // 数据指针
            text_features_input.size(),        // 数据元素数量
            input_node_dims1.data(),           // 形状指针
            input_node_dims1.size()            // 维度数量
    );


    save_tensor_to_file(inputTensor,"/workspace/clip.cpp/input_text.txt");


    // 创建运行选项
    Ort::RunOptions runOptions;

    // 定义输入和输出节点的名称
    std::vector<std::string> input_names1 = {"TEXT"};
    std::vector<std::string> output_names1 = {"TEXT_EMBEDDING"};

    // 转换输入和输出名称为 const char* 数组
    std::vector<const char*> input_names_cstr(input_names1.size());
    std::vector<const char*> output_names_cstr(output_names1.size());

    std::transform(input_names1.begin(), input_names1.end(), input_names_cstr.begin(),
                   [](const std::string& str) { return str.c_str(); });

    std::transform(output_names1.begin(), output_names1.end(), output_names_cstr.begin(),
                   [](const std::string& str) { return str.c_str(); });

    // 运行推理
    std::vector<Ort::Value> ort_outputs = ort_session->Run(
            runOptions,                         // 运行选项
            input_names_cstr.data(),            // 输入名称
            &inputTensor,                       // 输入张量
            1,                                  // 输入张量数量
            output_names_cstr.data(),           // 输出名称
            output_names_cstr.size()            // 输出张量数量
    );

    // 处理输出...
    // 可以根据需要从 ort_outputs 中提取数据并填充 output 向量


    const float *text_feature_ptr = ort_outputs[0].GetTensorMutableData<float>();

    for (int i = 0 ; i < 512 ; ++i)
    {
        std::cout<<text_feature_ptr[i]<<std::endl;
        output.push_back(text_feature_ptr[i]);
    }


}

