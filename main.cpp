#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
using namespace nvinfer1;
using namespace nvonnxparser;


// 打印日志文件
// 如果重要程度小于Kwarning的话 就打印出来
class Logger : public ILogger
{
    void log(Severity severity,const char* msg) noexcept override
    {
        if(severity<=Severity::kWARNING){
            std::cout<<msg<<std::endl;
        }
    }
} logger;

int main() {

    // 创建一个指向Ibuilder类型的指针builder
    // 1.创建builder
    // createInferBuilder 函数是 NVIDIA TensorRT 库中的一个函数，
    // 用于创建一个 IBuilder 接口实例
    // logger只是记载输出 这个无所谓的
    // 下面的代码就是其实builder拥有创建推理引擎的所有方法
    IBuilder* builder = createInferBuilder(logger);


    // 2.创建builder之后 优化模型的第一部就是创建网络的定义

    // 它表示一个标志，用于指示网络定义应该使用显式批处理
    uint32_t flag = 1U<<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    // 创建一个名为network的指针指向网络的定义
    // createNetworkV2方法的作用是创建一个新的网络定义对象，该对象将用于构建一个深度神经网络
    INetworkDefinition* network = builder->createNetworkV2(flag);

    // 这句代码的含义是将序列化文件加载到内存进行推理 更直白的说就是读入onnx文件和trt文件进行推理
    IParser* parser = createParser(*network,logger);

    // 输入onnx文件的路径
    std::string modelFile = "./mnist.onnx";
    // 注意这里传入的时候要给modelFile变量加入.c_str()后缀 因为她期望的是字符串格式
    //使用这个parser之前记得链接libnvparser
    // 这个函数用于从磁盘上的ONNX模型文件中解析出模型的网络结构和权重参数
    // 并将它们存储在TensorRT的内部数据结构中，以便后续的推理过程中使用
    parser->parseFromFile(modelFile.c_str(),
                          static_cast<int32_t>(ILogger::Severity::kWARNING));

    //这个循环语句的作用是检查在解析ONNX模型文件时是否出现了错误，并输出错误信息
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }


    // 下一步是创建build configuration特指TensorRT该如何优化模型
    // 这里还是builder的方法
    // 创建config指针来进行builderconfig的设置
    IBuilderConfig* config = builder->createBuilderConfig();

    // 这里的config可以设置很多属性 其中docs中提到最重要的属性是sizememory
    // 这里举个例子 设置为1G 其中1U <<30 应该是30次方的意思
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,1U << 30);

    // 将上面定义的network和config传入
    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network,*config);

    delete parser;
    delete network;
    delete config;
    delete builder;
    delete serializedModel;

    // 创建运行时
    IRuntime* runtime = createInferRuntime(logger);
    // 这里是反序列化
    // 使用TensorRT运行时库中的deserializeCudaEngine函数将存储在modelData中的模型数据反序列化为ICudaEngine对象
    // 反序列化后的ICudaEngine对象可以用于执行推理操作，即输入一组数据，输出预测结果
    ICudaEngine* engine = runtime->deserializeCudaEngine(modelData,modelSize);


    // 创建上下文
    IExecutionContext *context = engine->createExecutionContext();

    // 定义输入输出
    //通过setTensorAddress函数将输入和输出张量的地址设置为inputBuffer和outputBuffer
    //其中INPUT_NAME和OUTPUT_NAME是输入和输出张量的名称。
    context->setTensorAddress(INPUT_NAME, inputBuffer);
    context->setTensorAddress(OUTPUT_NAME, outputBuffer);


    // 最后，通过enqueueV3函数将推理操作放入stream流中异步执行
    // 提高推理效率
    context->enqueueV3(stream);
    return 0;
}
