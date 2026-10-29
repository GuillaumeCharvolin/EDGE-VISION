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

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        app.exit(e);
        return std::nullopt;
    }

    config.inputFile = std::filesystem::absolute(config.inputFile);
    config.modelFile = std::filesystem::absolute(config.modelFile);

    return config;
}
