//
// Created by wangzijian on 24-7-12.
//

// include the basic header
#include <iostream>
#include <fstream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "filesystem"
#include "NvInferRuntimeCommon.h"

using namespace nvinfer1;
using std::vector;
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
} Bbox;

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

void save_txt(float* data,std::string filename, int element_count)
{
    std::ofstream file_var(filename);
    for (size_t i = 0; i < element_count; ++i) {
        file_var << data[i] << "\n";
    }
    file_var.close();
}


void hwc_to_chw(const cv::Mat& img, float* data) {
    int channels = img.channels();
    int height = img.rows;
    int width = img.cols;

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                data[c * height * width + h * width + w] = img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
}

// Function to preprocess the input image
cv::Mat preprocessImage(const cv::Mat& srcimg) {
    cv::Mat resized;
    const int height = srcimg.rows;
    const int width = srcimg.cols;
    cv::Mat temp_image = srcimg.clone();
    int input_height = 640;
    int input_width = 640;

    if (height > input_height || width > input_width)
    {
        const float scale = std::min((float)input_height / height, (float)input_width / width);
        cv::Size new_size = cv::Size(int(width * scale), int(height * scale));
        cv::resize(srcimg, temp_image, new_size);
    }

    float ratio_height = (float)height / temp_image.rows;
    float ratio_width = (float)width / temp_image.cols;

    cv::Mat input_img;
    cv::copyMakeBorder(temp_image, input_img, 0, input_height - temp_image.rows,
                       0, input_width - temp_image.cols, cv::BORDER_CONSTANT, 0);

    std::vector<cv::Mat> bgrChannels(3);
    cv::split(input_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 128.0, -127.5 / 128.0);
    }
    cv::Mat normalized_image;
    cv::merge(bgrChannels,normalized_image);
    return normalized_image;

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


float get_iou(const Bbox box1, const Bbox box2) {
    float x1 = std::max(box1.xmin, box2.xmin);
    float y1 = std::max(box1.ymin, box2.ymin);
    float x2 = std::min(box1.xmax, box2.xmax);
    float y2 = std::min(box1.ymax, box2.ymax);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    float over_area = w * h;
    if (over_area == 0)
        return 0.0;
    float union_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin) + (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin) - over_area;
    return over_area / union_area;
}



std::vector<int> nms(std::vector<Bbox> boxes, std::vector<float> confidences, const float nms_thresh) {
    sort(confidences.begin(), confidences.end(), [&confidences](size_t index_1, size_t index_2)
    { return confidences[index_1] > confidences[index_2]; });
    const int num_box = confidences.size();
    std::vector<bool> isSuppressed(num_box, false);
    for (int i = 0; i < num_box; ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < num_box; ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }

            float ovr = get_iou(boxes[i], boxes[j]);
            if (ovr > nms_thresh)
            {
                isSuppressed[j] = true;
            }
        }
    }

    std::vector<int> keep_inds;
    for (int i = 0; i < isSuppressed.size(); i++)
    {
        if (!isSuppressed[i])
        {
            keep_inds.emplace_back(i);
        }
    }
    return keep_inds;
}


void draw_bboxes(cv::Mat& image, const std::vector<Bbox>& bboxes) {
    for (const auto& bbox : bboxes) {
        cv::rectangle(image, cv::Point(bbox.xmin, bbox.ymin), cv::Point(bbox.xmax, bbox.ymax), cv::Scalar(0, 255, 0), 2);
    }
}

int main(){
    std::string engineFile = "/home/ai-test1/wangzijian/yolov5-trt/yoloface_8n.engine";
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
    int outputSize = 1 * 20 * 8400 ; // Output size (change based on your model)

    cv::Mat preprocessed = preprocessImage(img);

    float* input = new float [inputH * inputW * 3];
//    std::memcpy(input, preprocessed.ptr<float>(), inputH * inputW * 3 * sizeof(float));
    hwc_to_chw(preprocessed,input);
    std::string filename = "/home/ai-test1/wangzijian/yolov5-trt/input-1750.txt";

    save_txt(input,filename,inputH * inputW * 3);



    float* output = new float[outputSize];


    doInference(*context, input, output, 1, inputH * inputW * 3, outputSize);
    std::string output_filename = "/home/ai-test1/wangzijian/yolov5-trt/output-1756.txt";
    save_txt(output,output_filename,20 * 8400);

    // 这里输出的output就是输出的向量

    for (int i = 0; i < 10; i++)
        std::cout<<output[i]<<std::endl;

    int num_box = 8400;
    const float conf_threshold = 0.5f;
    const float iou_threshold = 0.4f;
    const float ratio_width  =1.6875;
    const float ratio_height  =1.6875;
    vector<float> score_raw;
    vector<Bbox> bounding_box_raw;
    for (int i = 0; i < num_box; i++)
    {
        const float score = output[4 * num_box + i];
        if (score > conf_threshold)
        {
            float xmin = (output[i] - 0.5 * output[2 * num_box + i]) * ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymin = (output[num_box + i] - 0.5 * output[3 * num_box + i]) * ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float xmax = (output[i] + 0.5 * output[2 * num_box + i]) * ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymax = (output[num_box + i] + 0.5 * output[3 * num_box + i]) * ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            ////坐标的越界检查保护，可以添加一下
            bounding_box_raw.emplace_back(Bbox{xmin, ymin, xmax, ymax});
            score_raw.emplace_back(score);
            /// 剩下的5个关键点坐标的计算,暂时不写,因为在下游的模块里没有用到5个关键点坐标信息
        }
    }

    vector<int> keep_inds = nms(bounding_box_raw, score_raw, iou_threshold);
    const int keep_num = keep_inds.size();
    std::vector<Bbox> boxes;
    boxes.clear();
    boxes.resize(keep_num);
    for (int i = 0; i < keep_num; i++)
    {
        const int ind = keep_inds[i];
        boxes[i] = bounding_box_raw[ind];
    }

    draw_bboxes(img,boxes);

    cv::imwrite("/home/ai-test1/wangzijian/yolov5-trt/output.jpg",img);

}


