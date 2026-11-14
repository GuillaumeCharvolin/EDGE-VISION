#include <CLI/CLI.hpp>
#include "argParser.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <vector>

namespace {

// CLI11 check functions return an empty string when the value is valid, an error message otherwise.
using Check = std::function<std::string(const std::string&)>;

// A camera is given to --input as its index: a plain non-negative number like "0".
bool isCameraIndex(const std::string& input) {
    return !input.empty() && std::all_of(input.begin(), input.end(), ::isdigit);
}

// --model: an .onnx file (CLI::ExistingFile checks that it exists).
Check modelFileCheck() {
    return [](const std::string& filename) {
        if (std::filesystem::path(filename).extension() != ".onnx") {
            return std::string("Model weights file must have an .onnx extension");
        }
        return std::string();
    };
}

// --input: a camera index, or an existing .jpg / .mp4 / .avi file.
Check inputCheck() {
    return [](const std::string& input) {
        if (isCameraIndex(input)) {
            return std::string();
        }
        const std::filesystem::path path(input);
        if (!std::filesystem::is_regular_file(path)) {
            return "Input file does not exist: " + input;
        }
        if (path.extension() != ".jpg" && !isVideoFile(path)) {
            return std::string("Input file must have a .jpg, .mp4 or .avi extension");
        }
        return std::string();
    };
}

// A file the program writes: right extension, and its directory must exist.
Check outputFileCheck(const std::vector<std::string>& extensions) {
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
            return "Output directory does not exist: " + path.parent_path().string();
        }
        return std::string();
    };
}

// Rules that involve several options together. A camera follows the same rules as a video.
// Returns false (with an error message) if a rule is broken.
bool checkOptionsTogether(const AppConfig& config) {
    const bool videoInput = config.cameraIndex.has_value() || isVideoFile(config.inputFile);

    if (!config.outputFile.empty()) {
        const std::string expected = videoInput ? ".mp4" : ".jpg";
        if (config.outputFile.extension() != expected) {
            std::cerr << "--output: must be a " << expected << " file when --input is "
                      << (videoInput ? "a video or a camera" : "an image") << std::endl;
            return false;
        }
    }
    if (videoInput && !config.detectionsFile.empty()) {
        std::cerr << "--detections: not supported for video or camera input yet" << std::endl;
        return false;
    }
    if (!videoInput && !config.timingsFile.empty()) {
        std::cerr << "--timings: only for video or camera input" << std::endl;
        return false;
    }
    return true;
}

// Make an optional path absolute, leaving it empty if it was not given.
void makeAbsolute(std::filesystem::path& path) {
    if (!path.empty()) {
        path = std::filesystem::absolute(path);
    }
}

}  // namespace

bool isVideoFile(const std::filesystem::path& file) {
    return file.extension() == ".mp4" || file.extension() == ".avi";
}

std::optional<AppConfig> parseArguments(int argc, char** argv) {
    AppConfig config;
    std::string input;
    CLI::App app{"EDGE-VISION"};

    app.add_option("--model", config.modelFile, "Path to onnx model weights file")
        ->required()
        ->check(CLI::ExistingFile)
        ->check(modelFileCheck());
    app.add_option("--input", input, "Path to the input jpg image or mp4/avi video, or a camera index (e.g. 0)")
        ->required()
        ->check(inputCheck());
    app.add_option("--output", config.outputFile,
                   "Path where the image (jpg) or video (mp4) with drawn boxes is saved")
        ->check(outputFileCheck({".jpg", ".mp4"}));
    app.add_option("--detections", config.detectionsFile, "Path where the detections are saved as csv")
        ->check(outputFileCheck({".csv"}));
    app.add_option("--timings", config.timingsFile,
                   "Path where per-frame stage timings are saved as csv (video or camera only)")
        ->check(outputFileCheck({".csv"}));
    app.add_option("--warmup", config.warmupFrames,
                   "Number of first frames left out of the timings csv (default 10)")
        ->check(CLI::NonNegativeNumber);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        app.exit(e);
        return std::nullopt;
    }

    if (isCameraIndex(input)) {
        config.cameraIndex = std::stoi(input);
    } else {
        config.inputFile = input;
    }

    if (!checkOptionsTogether(config)) {
        return std::nullopt;
    }

    for (std::filesystem::path* path : {&config.modelFile, &config.inputFile, &config.outputFile,
                                        &config.detectionsFile, &config.timingsFile}) {
        makeAbsolute(*path);
    }
    return config;
}
