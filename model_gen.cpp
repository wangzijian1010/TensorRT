#include <fstream>
#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "logger.h"
using namespace nvinfer1;
using namespace sample;

const char* IN_NAME = "input";
const char* OUTPUT_NAME = "output";

static const int IN_H = 224;
static const int IN_W = 224;
static const int BATCH_SIZE = 1;

static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

int main(int argc, char** argv) {
    // 创建构建器和配置器
    Logger m_logger;
    IBuilder* builder = createInferBuilder(m_logger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // 创建网络定义
    INetworkDefinition* network = builder->createNetworkV2(EXPLICIT_BATCH);

    // 添加输入和池化层
    ITensor* input_tensor = network->addInput(IN_NAME, DataType::kFLOAT, Dims4{ BATCH_SIZE, 3, IN_H, IN_W });
    IPoolingLayer* pool = network->addPoolingNd(*input_tensor, PoolingType::kMAX, DimsHW{ 2, 2 });

    // 设置池化层的步幅
    pool->setStrideNd(DimsHW{ 2, 2 });
    pool->getOutput(0)->setName(OUTPUT_NAME);
    network->markOutput(*pool->getOutput(0));

    // 设置优化配置文件
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(IN_NAME, OptProfileSelector::kMIN, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
    profile->setDimensions(IN_NAME, OptProfileSelector::kOPT, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
    profile->setDimensions(IN_NAME, OptProfileSelector::kMAX, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
    config->addOptimizationProfile(profile); // 添加优化配置文件到配置器

    // 设置最大工作空间大小
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    IHostMemory* modelStream{ nullptr };
    assert(engine != nullptr);
    modelStream = engine->serialize();

    std::ofstream p("../model.engine", std::ios::binary);

    if (!p) {
        std::cerr << "could not open output file to save model" << std::endl;
        return -1;
    }

    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    std::cout << "generating file done!" << std::endl;

    // 释放资源
    modelStream->destroy();
    network->destroy();
    engine->destroy();
    config->destroy();
    builder->destroy();

    return 0;
}