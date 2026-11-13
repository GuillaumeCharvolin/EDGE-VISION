#include "argParser.h"
#include "draw.h"
#include "model.h"
#include "pipeline.h"
#include "preprocess.h"
#include "serialize.h"

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

    std::optional<std::vector<Detection>> detectionsOpt = detect(model, mat);
    if (!detectionsOpt) {
        return 1;
    }
    const std::vector<Detection>& detections = *detectionsOpt;

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
