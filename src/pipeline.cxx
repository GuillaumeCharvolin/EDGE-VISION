#include <iostream>
#include "pipeline.h"
#include "preprocess.h"

std::optional<std::vector<Detection>> detect(Model& model, const cv::Mat& frame) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_node_dims = model.session().GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::optional<PreprocessResult> preprocessOpt = preprocessImage(frame, input_node_dims, allocator);
    if (!preprocessOpt) {
        return std::nullopt;
    }

    std::vector<Ort::Value> outputs;
    try {
        outputs = model.session().Run(Ort::RunOptions{nullptr},
                                      model.inputNamePtrs().data(),
                                      &preprocessOpt->tensor,
                                      1,
                                      model.outputNamePtrs().data(),
                                      model.outputNamePtrs().size());
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Run failed: " << e.what() << std::endl;
        return std::nullopt;
    }

    const auto outputInfo = outputs[0].GetTensorTypeAndShapeInfo();
    if (outputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::cerr << "Error: Model output must be a float tensor." << std::endl;
        return std::nullopt;
    }

    std::optional<std::vector<Detection>> detectionsOpt =
        decodeOutput(outputs[0].GetTensorData<float>(), outputInfo.GetShape());
    if (!detectionsOpt) {
        return std::nullopt;
    }

    std::vector<Detection> detections;
    for (const Detection& letterboxed : *detectionsOpt) {
        detections.push_back(unletterbox(letterboxed, preprocessOpt->letterbox, frame.cols, frame.rows));
    }
    return detections;
}
