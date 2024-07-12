//
// Created by wangzijian on 24-7-12.
//
#include <iostream>
#include <fstream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"

using namespace nvinfer1;

// Logger for TensorRT info/warning/errors
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// Function to preprocess the input image
cv::Mat preprocessImage(const cv::Mat& img, int inputH, int inputW) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(inputW, inputH));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    return resized;
}

// Function to perform inference
void doInference(IExecutionContext& context, float* input, float* output, int batchSize, int inputSize, int outputSize) {
    const ICudaEngine& engine = context.getEngine();
    assert(engine.getNbBindings() == 2);

    void* buffers[2];
    const int inputIndex = engine.getBindingIndex("images");
    const int outputIndex = engine.getBindingIndex("output0");

    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * inputSize * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * outputSize * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * inputSize * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

// Function to apply non-maximum suppression
std::vector<int> nonMaximumSuppression(const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, float nmsThreshold) {
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, nmsThreshold, indices);
    return indices;
}

// Function to postprocess the output
void postprocessOutput(float* output, int outputSize, int inputH, int inputW, float confThreshold, float nmsThreshold, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector<int>& classIds) {
    for (int i = 0; i < outputSize; i += 85) {
        float confidence = output[i + 4];
        if (confidence >= confThreshold) {
            float* classesScores = output + i + 5;
            int classId;
            float maxClassScore = -1;
            for (int j = 0; j < 80; ++j) {
                if (classesScores[j] > maxClassScore) {
                    maxClassScore = classesScores[j];
                    classId = j;
                }
            }

            if (maxClassScore >= confThreshold) {
                float centerX = output[i];
                float centerY = output[i + 1];
                float width = output[i + 2];
                float height = output[i + 3];
                int left = static_cast<int>((centerX - width / 2) * inputW);
                int top = static_cast<int>((centerY - height / 2) * inputH);

                boxes.push_back(cv::Rect(left, top, static_cast<int>(width * inputW), static_cast<int>(height * inputH)));
                confidences.push_back(confidence);
                classIds.push_back(classId);
            }
        }
    }

    std::vector<int> indices = nonMaximumSuppression(boxes, confidences, nmsThreshold);
    std::vector<cv::Rect> nmsBoxes;
    std::vector<float> nmsConfidences;
    std::vector<int> nmsClassIds;
    for (int idx : indices) {
        nmsBoxes.push_back(boxes[idx]);
        nmsConfidences.push_back(confidences[idx]);
        nmsClassIds.push_back(classIds[idx]);
    }
    boxes = nmsBoxes;
    confidences = nmsConfidences;
    classIds = nmsClassIds;
}

int main(int argc, char** argv) {


    std::string engineFile = "/home/ai-test1/wangzijian/yolov5-trt/yolov5s.engine";
    std::string imageFile = "/home/ai-test1/wangzijian/yolov5-trt/test_lite_yolov5_1.jpg";

    // Load engine
    char* trtModelStream{ nullptr };
    size_t size{ 0 };
    std::ifstream file(engineFile, std::ios::binary);

    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    Logger logger;
    IRuntime* runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    delete[] trtModelStream;

    // Load input image
    cv::Mat img = cv::imread(imageFile);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imageFile << std::endl;
        return -1;
    }

    int inputH = 640; // Input height (change based on your model)
    int inputW = 640; // Input width (change based on your model)
    int outputSize = 25200 * 85; // Output size (change based on your model)

    cv::Mat preprocessed = preprocessImage(img, inputH, inputW);

    float* input = new float[inputH * inputW * 3];
    std::memcpy(input, preprocessed.ptr<float>(), inputH * inputW * 3 * sizeof(float));

    float* output = new float[outputSize];

    doInference(*context, input, output, 1, inputH * inputW * 3, outputSize);
    // Postprocess output
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    postprocessOutput(output, outputSize, inputH, inputW, 0.0, 0.1, boxes, confidences, classIds);

    // Draw the detection results
    for (size_t i = 0; i < boxes.size(); ++i) {
        cv::rectangle(img, boxes[i], cv::Scalar(0, 255, 0), 2);
        std::string label = std::to_string(classIds[i]) + ": " + std::to_string(confidences[i]);
        cv::putText(img, label, cv::Point(boxes[i].x, boxes[i].y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    // Save the result image
    cv::imwrite("result.jpg", img);

    // Cleanup
    delete[] input;
    delete[] output;
    context->destroy();
    engine->destroy();
    runtime->destroy();

    std::cout << "Inference done!" << std::endl;
    return 0;
}