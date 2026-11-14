#include <chrono>
#include <iostream>
#include "pipeline.h"
#include "preprocess.h"

namespace {

using Clock = std::chrono::steady_clock;

double millisecondsBetween(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

}  // namespace

std::optional<std::vector<Detection>> detect(Model& model, const cv::Mat& frame, StageTimings* timings) {
    const Clock::time_point preprocessStart = Clock::now();
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_node_dims = model.session().GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::optional<PreprocessResult> preprocessOpt = preprocessImage(frame, input_node_dims, allocator);
    if (!preprocessOpt) {
        return std::nullopt;
    }

    const Clock::time_point inferenceStart = Clock::now();
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

    const Clock::time_point postprocessStart = Clock::now();
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

    if (timings) {
        const Clock::time_point end = Clock::now();
        timings->preprocessMs = millisecondsBetween(preprocessStart, inferenceStart);
        timings->inferenceMs = millisecondsBetween(inferenceStart, postprocessStart);
        timings->postprocessMs = millisecondsBetween(postprocessStart, end);
    }
    return detections;
}
