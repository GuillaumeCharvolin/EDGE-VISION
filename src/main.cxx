#include "argParser.h"
#include "model.h"
#include "preprocess.h"

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

    std::optional<cv::Mat> matOpt = loadImage(config.imageFile);
    if (!matOpt) {
        return 1;
    }
    cv::Mat mat = std::move(*matOpt);

    // Show the source image (close the window or press any key to continue).
    cv::imshow("edge_vision input", mat);
    std::cout << "Press any key in the image window to run inference...\n";
    cv::waitKey(0);
    cv::destroyAllWindows();

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_node_dims = model.session().GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::optional<Ort::Value> inputTensorOpt = preprocessImage(mat, input_node_dims, allocator);
    if (!inputTensorOpt) {
        return 1;
    }
    Ort::Value input_tensor = std::move(*inputTensorOpt);

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

    std::cout << "Raw outputs (" << outputs.size() << "):\n";
    for (size_t o = 0; o < outputs.size(); ++o) {
        const auto info = outputs[o].GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        const size_t count = info.GetElementCount();

        std::cout << "  [" << o << "] name=" << model.outputNames()[o] << " shape=[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) {
                std::cout << ',';
            }
            std::cout << shape[i];
        }
        std::cout << "] values(" << count << ")=";

        if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float* data = outputs[o].GetTensorData<float>();
            for (size_t i = 0; i < count; ++i) {
                if (i > 0) {
                    std::cout << ' ';
                }
                std::cout << data[i];
            }
        } else {
            std::cout << "<non-float tensor, type=" << static_cast<int>(info.GetElementType())
                      << ">";
        }
        std::cout << '\n';
    }

    return 0;
}
