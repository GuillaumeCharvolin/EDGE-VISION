#include "argParser.h"
#include "draw.h"
#include "model.h"
#include "postprocess.h"
#include "serialize.h"
#include "preprocess.h"

#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    std::optional<AppConfig> configOpt = parseArguments(argc, argv);
    if (!configOpt) {
        return 1;
    }
    const AppConfig& config = *configOpt;

    std::optional<Model> modelOpt = Model::load(config.modelFile);
    if (!modelOpt) {
        return 1;
    }
    Model model = std::move(*modelOpt);

    std::optional<cv::Mat> matOpt = loadImage(config.inputFile);
    if (!matOpt) {
        return 1;
    }
    cv::Mat mat = std::move(*matOpt);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_node_dims = model.session().GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::optional<PreprocessResult> preprocessOpt = preprocessImage(mat, input_node_dims, allocator);
    if (!preprocessOpt) {
        return 1;
    }
    Ort::Value input_tensor = std::move(preprocessOpt->tensor);

    std::vector<Ort::Value> outputs;
    try {
        outputs = model.session().Run(Ort::RunOptions{nullptr},
                                      model.inputNamePtrs().data(),
                                      &input_tensor,
                                      1,
                                      model.outputNamePtrs().data(),
                                      model.outputNamePtrs().size());
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Run failed: " << e.what() << std::endl;
        return 1;
    }

    const auto outputInfo = outputs[0].GetTensorTypeAndShapeInfo();
    if (outputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::cerr << "Error: Model output must be a float tensor." << std::endl;
        return 1;
    }

    std::optional<std::vector<Detection>> detectionsOpt =
        decodeOutput(outputs[0].GetTensorData<float>(), outputInfo.GetShape());
    if (!detectionsOpt) {
        return 1;
    }

    std::vector<Detection> detections;
    for (const Detection& letterboxed : *detectionsOpt) {
        detections.push_back(unletterbox(letterboxed, preprocessOpt->letterbox, mat.cols, mat.rows));
    }

    std::cout << "Detections (" << detections.size() << "):\n";
    for (const Detection& det : detections) {
        std::cout << "  class=" << cocoClassName(det.classId) << " score=" << det.score
                  << " x=" << det.box.x << " y=" << det.box.y
                  << " w=" << det.box.width << " h=" << det.box.height << '\n';
    }

    if (!config.outputFile.empty()) {
        drawDetections(mat, detections);
        if (!cv::imwrite(config.outputFile.string(), mat)) {
            std::cerr << "Error: Could not write the output image to: " << config.outputFile.string() << std::endl;
            return 1;
        }
        std::cout << "Saved " << config.outputFile.string() << '\n';
    }

    if (!config.detectionsFile.empty()) {
        std::ofstream csv(config.detectionsFile);
        if (!csv) {
            std::cerr << "Error: Could not open the detections file: " << config.detectionsFile.string() << std::endl;
            return 1;
        }
        writeDetectionsCsv(csv, detections);
        std::cout << "Saved " << config.detectionsFile.string() << '\n';
    }

    return 0;
}
