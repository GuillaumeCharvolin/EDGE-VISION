#include <CLI/CLI.hpp>
#include "argParser.h"

#include <algorithm>
#include <iostream>
#include <vector>

namespace {

// Check for a file the program writes: right extension, and its directory must exist.
std::function<std::string(const std::string&)> outputFileCheck(const std::vector<std::string>& extensions) {
    return [extensions](const std::string& filename) {
        const std::filesystem::path path = std::filesystem::absolute(filename);
        if (std::find(extensions.begin(), extensions.end(), path.extension()) == extensions.end()) {
            std::string message = "Output file must have one of these extensions:";
            for (const std::string& extension : extensions) {
                message += " " + extension;
            }
            return message;
        }
        if (!std::filesystem::is_directory(path.parent_path())) {
            return std::string("Output directory does not exist: ") + path.parent_path().string();
        }
        return std::string();
    };
}

}  // namespace

bool isVideoFile(const std::filesystem::path& file) {
    return file.extension() == ".mp4" || file.extension() == ".avi";
}

std::optional<AppConfig> parseArguments(int argc, char** argv) {
    AppConfig config;
    auto app = CLI::App{"EDGE-VISION"};

    app.add_option("--model", config.modelFile, "Path to onnx model weights file")
        ->required()
        ->check(CLI::ExistingFile)
        ->check([](const std::string& filename) {
            if (std::filesystem::path(filename).extension() != ".onnx") {
                return std::string("Model weights file must have an .onnx extension");
            };
            return std::string();
        });

    app.add_option("--input", config.inputFile, "Path to the input jpg image, or mp4/avi video")
        ->required()
        ->check(CLI::ExistingFile)
        ->check([](const std::string& filename) {
            const std::filesystem::path path(filename);
            if (path.extension() != ".jpg" && !isVideoFile(path)) {
                return std::string("Input file must have a .jpg, .mp4 or .avi extension");
            };
            return std::string();
        });

    app.add_option("--output", config.outputFile,
                   "Path where the image (jpg) or video (mp4) with drawn boxes is saved")
        ->check(outputFileCheck({".jpg", ".mp4"}));

    app.add_option("--detections", config.detectionsFile, "Path where the detections are saved as csv")
        ->check(outputFileCheck({".csv"}));

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        app.exit(e);
        return std::nullopt;
    }

    // Checks that involve several options together.
    const bool videoInput = isVideoFile(config.inputFile);
    if (!config.outputFile.empty()) {
        const std::string expected = videoInput ? ".mp4" : ".jpg";
        if (config.outputFile.extension() != expected) {
            std::cerr << "--output: must be a " << expected << " file when --input is "
                      << (videoInput ? "a video" : "an image") << std::endl;
            return std::nullopt;
        }
    }
    if (videoInput && !config.detectionsFile.empty()) {
        std::cerr << "--detections: not supported for video input yet" << std::endl;
        return std::nullopt;
    }

    config.inputFile = std::filesystem::absolute(config.inputFile);
    config.modelFile = std::filesystem::absolute(config.modelFile);
    if (!config.outputFile.empty()) {
        config.outputFile = std::filesystem::absolute(config.outputFile);
    }
    if (!config.detectionsFile.empty()) {
        config.detectionsFile = std::filesystem::absolute(config.detectionsFile);
    }

    return config;
}
