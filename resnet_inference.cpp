#include <iostream>
#include <fstream>
#include <numeric> // needed for accumulate()

#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#define WIDTH 32
#define HEIGHT 32
#define NUM_CLASSES 10

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


std::vector<std::string> readLabels(std::string& labelFilepath)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);

    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}

ONNXTensorElementDataType getDataType(Ort::TypeInfo info) {

    auto tensorInfo = info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType dtype = tensorInfo.GetElementType();
    return dtype;
}

std::vector<int64_t> getDataShape(Ort::TypeInfo info) {

    auto tensorInfo = info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensorInfo.GetShape();
    return shape;
}

int main(int argc, const char* argv[]) {

    if (argc != 3) {
        std::cerr << "Usage: ./classifier <path-to-onnx-model> <path-to-test-img>" << std::endl;
        return -1;
    }


    std::string instanceName{"resnet-cifar-classification"};
    std::string modelPath{argv[1]};
    std::string imgPath{argv[2]};
    
    std::string labelPath{"labels.txt"};
    std::vector<std::string> labels{readLabels(labelPath)};

    // define an environment that has options related to logging levels,
    // execution providers, memory allocators, etc.
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                instanceName.c_str());

    // define the options for the session
    Ort::SessionOptions options;

    // [Relevant only to CPU EP] If onnx runtime finds an op that can be 
    // run parallelly across multiple threads, it allocates multiple threads 
    // to run different portions of the operator's computation in parallel
    // Upto how many threads to use is being specified here. 
    options.SetIntraOpNumThreads(1);

    options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED
    );

    // create the sesion to run the model
    Ort::Session session{env, modelPath.c_str(), options};

    // memory allocator that manages all the memory allocation related 
    // to content extracted from the model file through the session
    Ort::AllocatorWithDefaultOptions allocator;

    // get the name of the input & output for the model
    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);

    // Get the data type & shape of the input & output tensors (only 1 here so the 0th input & 0th output)
    ONNXTensorElementDataType inputType = getDataType(session.GetInputTypeInfo(0));
    ONNXTensorElementDataType outputType = getDataType(session.GetOutputTypeInfo(0));
    
    std::vector<int64_t> inputShape = getDataShape(session.GetInputTypeInfo(0));
    std::vector<int64_t> outputShape = getDataShape(session.GetOutputTypeInfo(0));

    const int64_t batchSize = 1;

    // Set the batch sizes of the input & output tensors
    if (inputShape.at(0) == -1) {
        std::cout << "Got dynamic batch size. Setting input batch size to "
                << batchSize << "." << std::endl;

        inputShape.at(0) = batchSize;
    }
    
    if (outputShape.at(0) == -1) {
        std::cout << "Got dynamic batch size. Setting output batch size to "
                << batchSize << "." << std::endl;
        
        outputShape.at(0) = batchSize;
    }

    std::cout << "Input Name: " << inputName.get() << std::endl;
    std::cout << "Input Type: " << inputType << std::endl;
    std::cout << "Input Dims: " << inputShape << std::endl;
    
    std::cout << "Output Name: " << outputName.get() << std::endl;
    std::cout << "Output Type: " << outputType << std::endl;
    std::cout << "Output Dims: " << outputShape << std::endl;

    // Read in the test image
    cv::Mat imageBGR = cv::imread(imgPath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;

    // resize the image according to the input shape required by the model
    cv::resize(imageBGR, resizedImageBGR, 
                cv::Size(inputShape.at(3), inputShape.at(2)), 
                cv::InterpolationFlags::INTER_CUBIC
                );
    
    // change channel order BGR to RGB
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                    cv::ColorConversionCodes::COLOR_BGR2RGB);

    // scale the image to 0, 1 then change the data type to float32 
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0/255);

    // separate out the 3 channels to apply channel-wise normalizations
    cv::Mat channels[3];
    cv::split(resizedImage, channels);

    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    
    // merge the channels back together
    cv::merge(channels, 3, resizedImage);

    // change the shape to channels-first 
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    // get the size of the input & output tensors i.e.,
    // for a tensor of (n, h, w, c) this returns n*h*w*c
    size_t inputTensorSize = vectorProduct(inputShape);
    size_t outputTensorSize = vectorProduct(outputShape);
    assert(("Output tensor size should equal to the label set size.",
            labels.size() * batchSize == outputTensorSize));
    
    // create a flattened vector to hold an entire tensor
    std::vector<float> inputTensorValues(inputTensorSize);
    std::vector<float> outputTensorValues(outputTensorSize);
    
    // load the values from cv::Mat to a float vector
    for (int64_t i = 0; i < batchSize; ++i)
    {
        std::copy(preprocessedImage.begin<float>(),
                  preprocessedImage.end<float>(),
                  inputTensorValues.begin() + i * inputTensorSize / batchSize);
    }

    std::vector<const char*> inputNames{inputName.get()};
    std::vector<const char*> outputNames{outputName.get()};
    
    // create an allocator object. this will be used to create Ort::Value 
    // tensors for input and output 
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    // push a tensor into the vector of input Tensors
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputShape.data(),
        inputShape.size()));
    
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputShape.data(), outputShape.size()));

    // run the session to get model output
    session.Run(Ort::RunOptions{nullptr}, 
                inputNames.data(), inputTensors.data(), 1,  // Number of inputs
                outputNames.data(), outputTensors.data(), 1 // Number of outputs
                );

    // get the results in this vector
    std::vector<int> predIds(batchSize, 0);

    // get the text labels for the predictions in this vector
    std::vector<std::string> predLabels(batchSize);

    // get the confidence values in thie vector
    std::vector<float> confidences(batchSize, 0.0f);
    
    for (int64_t b = 0; b < batchSize; ++b)
    {
        float activation = 0;
        float maxActivation = std::numeric_limits<float>::lowest();
        float expSum = 0;
        for (int i = 0; i < labels.size(); i++)
        {
            activation = outputTensorValues.at(i + b * labels.size());
            expSum += std::exp(activation);
            if (activation > maxActivation)
            {
                predIds.at(b) = i;
                maxActivation = activation;
            }
        }
        predLabels.at(b) = labels.at(predIds.at(b));
        confidences.at(b) = std::exp(maxActivation) / expSum;
    }

    std::cout << "Predicted labels: " << predLabels << std::endl;
    std::cout << "Confidence scores: " << confidences << std::endl;
}