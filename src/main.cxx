#include "argParser.h"
#include "draw.h"
#include "model.h"
#include "pipeline.h"
#include "preprocess.h"
#include "serialize.h"

#include <chrono>
#include <fstream>
#include <iostream>

namespace {

int runImage(Model& model, const AppConfig& config) {
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

int runVideo(Model& model, const AppConfig& config) {
    using Clock = std::chrono::steady_clock;

    cv::VideoCapture capture(config.inputFile.string());
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open the video at: " << config.inputFile.string() << std::endl;
        return 1;
    }

    // Write the output at the same rate as the input, so it plays at normal speed.
    double inputFps = capture.get(cv::CAP_PROP_FPS);
    if (inputFps <= 0) {
        inputFps = 30.0;
    }
    cv::VideoWriter writer;

    cv::Mat frame;
    int frameCount = 0;
    double smoothedFps = 0.0;
    const Clock::time_point start = Clock::now();
    Clock::time_point previous = start;

    while (capture.read(frame)) {
        std::optional<std::vector<Detection>> detectionsOpt = detect(model, frame);
        if (!detectionsOpt) {
            return 1;
        }
        drawDetections(frame, *detectionsOpt);

        // FPS from the time between two frames (read + detect + draw + write),
        // smoothed so the number does not jump at every frame.
        const Clock::time_point now = Clock::now();
        const double frameSeconds = std::chrono::duration<double>(now - previous).count();
        previous = now;
        const double instantFps = frameSeconds > 0 ? 1.0 / frameSeconds : 0.0;
        smoothedFps = frameCount == 0 ? instantFps : 0.9 * smoothedFps + 0.1 * instantFps;
        drawFps(frame, smoothedFps);

        if (!config.outputFile.empty()) {
            // Opened on the first frame, once the frame size is known.
            if (!writer.isOpened()) {
                writer.open(config.outputFile.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            inputFps, frame.size());
                if (!writer.isOpened()) {
                    std::cerr << "Error: Could not write the output video to: " << config.outputFile.string() << std::endl;
                    return 1;
                }
            }
            writer.write(frame);
        }

        ++frameCount;
        if (frameCount % 100 == 0) {
            std::cout << "  " << frameCount << " frames, FPS " << smoothedFps << '\n';
        }
    }

    if (frameCount == 0) {
        std::cerr << "Error: No frame could be read from: " << config.inputFile.string() << std::endl;
        return 1;
    }

    const double totalSeconds = std::chrono::duration<double>(Clock::now() - start).count();
    std::cout << "Processed " << frameCount << " frames, mean FPS " << frameCount / totalSeconds << '\n';
    if (!config.outputFile.empty()) {
        std::cout << "Saved " << config.outputFile.string() << '\n';
    }
    return 0;
}

}  // namespace

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

    if (isVideoFile(config.inputFile)) {
        return runVideo(model, config);
    }
    return runImage(model, config);
}
