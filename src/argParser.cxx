#include <CLI/CLI.hpp>
#include "argParser.h"

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

    app.add_option("--input", config.inputFile, "Path to the input jpg image file")
        ->required()
        ->check(CLI::ExistingFile)
        ->check([](const std::string& filename) {
            if (std::filesystem::path(filename).extension() != ".jpg") {
                return std::string("Input file must have a .jpg extension");
            };
            return std::string();
        });

    app.add_option("--output", config.outputFile, "Path where the jpg image with drawn boxes is saved")
        ->check([](const std::string& filename) {
            const std::filesystem::path path = std::filesystem::absolute(filename);
            if (path.extension() != ".jpg") {
                return std::string("Output file must have a .jpg extension");
            }
            if (!std::filesystem::is_directory(path.parent_path())) {
                return std::string("Output directory does not exist: ") + path.parent_path().string();
            }
            return std::string();
        });

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        app.exit(e);
        return std::nullopt;
    }

    config.inputFile = std::filesystem::absolute(config.inputFile);
    config.modelFile = std::filesystem::absolute(config.modelFile);
    if (!config.outputFile.empty()) {
        config.outputFile = std::filesystem::absolute(config.outputFile);
    }

    return config;
}
