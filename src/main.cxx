#include "argParser.h"
#include "model.h"
#include "preprocess.h"

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

    std::optional<cv::Mat> matOpt = loadImage(config.imageFile);
    if (!matOpt) {
        return 1;
    }
    cv::Mat mat = std::move(*matOpt);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_node_dims = model.session().GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::optional<Ort::Value> inputTensorOpt = preprocessImage(mat, input_node_dims, allocator);
    if (!inputTensorOpt) {
        return 1;
    }
    Ort::Value input_tensor = std::move(*inputTensorOpt);

    return 0;
}
